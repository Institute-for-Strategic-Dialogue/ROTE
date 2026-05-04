"""
X (Twitter) status checker.

Posts -- single probe:
    publish.twitter.com/oembed
        200 -> live  (returns author + title)
        404 -> gone  (deleted, suspended, or protected -- collapsed by oEmbed)
        429 -> rate-limited
        else -> error

Accounts -- two-stage probe with graceful degradation:

    Stage 1 (always): syndication.twitter.com/srv/timeline-profile/screen-name/<user>
        Free, no token. Parse the embedded `__NEXT_DATA__` JSON (and fall back
        to substring sniffing in case the shape mutates).
            hasResults:false, no headerProps   -> NEVER_EXISTED
            hasResults:true, empty entries     -> EXISTS_BUT_INVISIBLE
                (suspended / deactivated / protected / zero-tweet)
            hasResults:true, entries populated -> LIVE

    Stage 2 (best-effort): GraphQL UserByScreenName via api.x.com with a
    guest token. Splits the EXISTS_BUT_INVISIBLE bucket into PROTECTED vs
    SUSPENDED_OR_DEACTIVATED, and adds metadata for live accounts. If the
    doc_id has rotated or the request fails, we keep the syndication verdict.

Doc-id rotation:
    The GraphQL operation hash rotates every ~2-4 weeks. Override via the
    ``X_GRAPHQL_USER_BY_SCREEN_NAME_ID`` env var without code changes. The
    syndication probe is independent and keeps working.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Optional
from urllib.parse import quote, urlparse

import requests

from .base import (
    PlatformChecker,
    StatusResult,
    STATUS_LIVE,
    STATUS_GONE,
    STATUS_PROTECTED,
    STATUS_SUSPENDED_OR_DEACTIVATED,
    STATUS_NOT_FOUND,
    STATUS_UNKNOWN,
    STATUS_ERROR,
    STATUS_INVALID,
    STATUS_RATE_LIMITED,
)


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "sec-ch-ua": '"Chromium";v="124", "Not-A.Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "Upgrade-Insecure-Requests": "1",
    "Referer": "https://platform.twitter.com/",
}

# Public web-app bearer; static for years, used for guest-token activation.
X_PUBLIC_BEARER = (
    "AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D"
    "1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA"
)
X_GRAPHQL_DOC_ID = os.environ.get(
    "X_GRAPHQL_USER_BY_SCREEN_NAME_ID",
    "G3KGOASz96M-Qu0nwmGXNg",
)

OEMBED_URL = "https://publish.twitter.com/oembed"
SYNDICATION_URL = "https://syndication.twitter.com/srv/timeline-profile/screen-name/"

REQUEST_TIMEOUT = 12
SYNDICATION_RETRY_BACKOFF_SEC = 2.5  # one extra try after a 429 / 5xx

USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{1,15}$")
# Reserved x.com path components that look like usernames but aren't.
RESERVED_USERNAMES = frozenset({
    "i", "home", "search", "explore", "hashtag", "messages", "notifications",
    "settings", "compose", "tos", "privacy", "logout", "login", "signup",
    "share", "intent", "about", "help",
})
POST_URL_RE = re.compile(
    r"https?://(?:(?:www|mobile|m)\.)?(?:x\.com|twitter\.com|nitter\.[\w.-]+)/"
    r"(?:[A-Za-z0-9_]+|i/web)/status(?:es)?/(\d+)",
    re.IGNORECASE,
)
PROFILE_URL_RE = re.compile(
    r"https?://(?:(?:www|mobile|m)\.)?(?:x|twitter)\.com/"
    r"([A-Za-z0-9_]{1,15})/?$",
    re.IGNORECASE,
)


class XChecker(PlatformChecker):
    NAME = "x"
    DISPLAY_NAME = "X / Twitter"
    SUPPORTS_POSTS = True
    SUPPORTS_ACCOUNTS = True
    HOST_PATTERNS = [
        re.compile(
            r"https?://(?:(?:www|mobile|m)\.)?(?:x|twitter)\.com/",
            re.IGNORECASE,
        ),
    ]

    def __init__(self) -> None:
        # One session per checker instance; reused across calls in a batch.
        self._session = requests.Session()
        self._guest_token: Optional[str] = None
        self._guest_token_at: float = 0.0
        # Once GraphQL fails (e.g. doc_id rotated), don't keep hammering it
        # for the rest of the batch.
        self._graphql_disabled: bool = False

    # ------------------------------------------------------------------ parsing

    def parse_post(self, raw: str) -> Optional[str]:
        if not raw:
            return None
        s = raw.strip()
        m = POST_URL_RE.search(s)
        if m:
            return m.group(1)
        if s.isdigit() and 1 <= len(s) <= 25:
            return s
        return None

    def parse_account(self, raw: str) -> Optional[str]:
        if not raw:
            return None
        s = raw.strip().lstrip("@")
        if s.startswith("http"):
            m = PROFILE_URL_RE.search(s)
            if m and m.group(1).lower() not in RESERVED_USERNAMES:
                return m.group(1)
            try:
                p = urlparse(s)
                parts = [x for x in (p.path or "").split("/") if x]
                if parts and USERNAME_RE.match(parts[0]) and parts[0].lower() not in RESERVED_USERNAMES:
                    return parts[0]
            except Exception:
                return None
            return None
        if USERNAME_RE.match(s) and s.lower() not in RESERVED_USERNAMES:
            return s
        return None

    # ------------------------------------------------------------------ posts

    def check_post(self, raw: str) -> StatusResult:
        post_id = self.parse_post(raw)
        if not post_id:
            return StatusResult(
                input=raw, mode="post", platform=self.NAME,
                status=STATUS_INVALID, status_label="Invalid X post URL or id",
                error="Could not parse an X post URL or numeric id from the input.",
            )
        # Use the input URL if we can; otherwise rebuild a canonical lookup URL.
        # oEmbed rejects the `/i/web/status/<id>` form (returns 404), but
        # accepts any placeholder username -- it follows the redirect internally.
        canonical = raw.strip() if POST_URL_RE.search(raw or "") else f"https://twitter.com/_/status/{post_id}"

        try:
            r = self._session.get(
                OEMBED_URL,
                params={"url": canonical, "omit_script": "1"},
                headers=DEFAULT_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as e:
            return StatusResult(
                input=raw, mode="post", platform=self.NAME,
                status=STATUS_ERROR, status_label="Network error",
                normalized_url=canonical, identifier=post_id, error=str(e),
            )

        if r.status_code == 200:
            try:
                data = r.json()
            except ValueError:
                data = {}
            return StatusResult(
                input=raw, mode="post", platform=self.NAME,
                status=STATUS_LIVE, status_label="Live",
                normalized_url=canonical, identifier=post_id,
                author_name=data.get("author_name"),
                author_url=data.get("author_url"),
                author_handle=_handle_from_author_url(data.get("author_url")),
                oembed_title=data.get("title"),
            )
        if r.status_code == 404:
            return StatusResult(
                input=raw, mode="post", platform=self.NAME,
                status=STATUS_GONE,
                status_label="Gone (deleted, suspended, or protected)",
                normalized_url=canonical, identifier=post_id,
                detail="oEmbed returned 404 - X collapses deleted, suspended, and protected into a single signal.",
            )
        if r.status_code == 429:
            return StatusResult(
                input=raw, mode="post", platform=self.NAME,
                status=STATUS_RATE_LIMITED, status_label="Rate limited",
                normalized_url=canonical, identifier=post_id,
                detail="oEmbed rate limit. Slow down and retry.",
            )
        return StatusResult(
            input=raw, mode="post", platform=self.NAME,
            status=STATUS_ERROR, status_label=f"HTTP {r.status_code}",
            normalized_url=canonical, identifier=post_id,
            error=f"oEmbed returned HTTP {r.status_code}",
        )

    # ------------------------------------------------------------------ accounts

    def check_account(self, raw: str) -> StatusResult:
        username = self.parse_account(raw)
        if not username:
            return StatusResult(
                input=raw, mode="account", platform=self.NAME,
                status=STATUS_INVALID, status_label="Invalid X username or profile URL",
                error="Could not parse a valid X username from the input.",
            )

        canonical = f"https://x.com/{username}"

        # Stage 1 -- syndication
        s1 = self._probe_syndication(username)
        result = StatusResult(
            input=raw, mode="account", platform=self.NAME,
            status=s1.get("status", STATUS_ERROR),
            status_label=s1.get("label", "Inconclusive"),
            normalized_url=canonical, identifier=username,
            detail=s1.get("detail"),
            author_handle=s1.get("author_handle"),
            author_name=s1.get("author_name"),
            error=s1.get("error"),
        )

        # Skip stage 2 only when stage 1 already gave a definitive verdict.
        # NOT_FOUND is the one case GraphQL can't improve (it conflates that
        # with suspended/deactivated). Rate limit / error on syndication is
        # different -- GraphQL is on independent infra and may still succeed.
        if result.status in (STATUS_NOT_FOUND, STATUS_INVALID):
            return result

        # Stage 2 -- GraphQL UserByScreenName, best-effort.
        s2 = self._probe_graphql_user(username)
        if s2 is None:
            return result

        gql_status = s2.get("status")
        if gql_status in (STATUS_LIVE, STATUS_PROTECTED):
            # Replace ambiguous (or rate-limited) syndication verdict and enrich.
            result.status = gql_status
            result.status_label = s2.get("label", result.status_label)
            result.author_handle = s2.get("author_handle") or result.author_handle
            result.author_name = s2.get("author_name") or result.author_name
            result.followers = s2.get("followers")
            result.following = s2.get("following")
            result.statuses = s2.get("statuses")
            result.verified = s2.get("verified")
            result.profile_created_at = s2.get("profile_created_at")
            result.bio = s2.get("bio")
            result.withheld_in_countries = s2.get("withheld_in_countries")
            # Successful GraphQL supersedes any stale syndication-side note.
            result.detail = s2.get("detail") or "GraphQL UserByScreenName."
            result.error = None
        elif gql_status == STATUS_SUSPENDED_OR_DEACTIVATED:
            # GraphQL alone cannot distinguish "never existed" from "suspended
            # or deactivated" -- both produce empty `data` objects. Use
            # syndication's verdict as the tiebreaker:
            #   - syndication confirmed handle exists  -> it's truly suspended/deactivated
            #   - syndication had no usable signal     -> we genuinely can't tell, report UNKNOWN
            #     (rather than masking it as rate_limited)
            if result.status in (STATUS_RATE_LIMITED, STATUS_ERROR):
                result.status = STATUS_UNKNOWN
                result.status_label = "Unknown -- not confirmed by either probe"
                result.detail = (
                    "GraphQL returned empty user data; syndication probe "
                    f"({s1.get('label', 'unavailable')}) could not confirm whether the handle exists. "
                    "Could be never-existed OR suspended/deactivated."
                )
            else:
                result.status = STATUS_SUSPENDED_OR_DEACTIVATED
                result.status_label = s2.get("label", "Suspended or deactivated")
                if s2.get("detail"):
                    result.detail = s2["detail"]
        return result

    # ------------------------------------------------------------------ probes

    def _probe_syndication(self, username: str) -> dict:
        # One retry on 429 / 5xx -- syndication has a short, narrow rate window
        # and a brief backoff often gets us through.
        url = SYNDICATION_URL + quote(username)
        attempts = 0
        last_status: Optional[int] = None
        last_exc: Optional[str] = None
        body: str = ""
        while attempts < 2:
            attempts += 1
            try:
                r = self._session.get(url, headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT)
            except requests.RequestException as e:
                last_exc = str(e)
                last_status = None
            else:
                last_status = r.status_code
                if r.status_code == 200:
                    body = r.text or ""
                    break
                if r.status_code not in (429, 500, 502, 503, 504):
                    break  # non-retryable HTTP error
            if attempts < 2:
                time.sleep(SYNDICATION_RETRY_BACKOFF_SEC)

        if last_status is None:
            return {"status": STATUS_ERROR, "label": "Network error", "error": last_exc}
        if last_status == 429:
            return {
                "status": STATUS_RATE_LIMITED, "label": "Rate limited",
                "detail": "Syndication endpoint returned 429 (after retry).",
            }
        if last_status != 200:
            return {
                "status": STATUS_ERROR, "label": f"HTTP {last_status}",
                "error": f"Syndication returned HTTP {last_status}",
            }

        next_data = _extract_next_data(body)
        parsed = _parse_syndication_json(next_data) if next_data else {}

        # Defensive substring sniffing in case the JSON path mutates.
        sub_has_results_false = '"hasResults":false' in body or '"hasResults": false' in body
        sub_has_results_true = '"hasResults":true' in body or '"hasResults": true' in body
        sub_header_present = '"headerProps":{' in body and '"headerProps":null' not in body
        sub_empty_entries = bool(re.search(r'"entries"\s*:\s*\[\s*\]', body))

        has_results = parsed.get("has_results")
        if has_results is None:
            if sub_has_results_false:
                has_results = False
            elif sub_has_results_true:
                has_results = True

        has_header = parsed.get("has_header") or sub_header_present
        entries_count = parsed.get("entries_count")
        if entries_count is None and sub_empty_entries:
            entries_count = 0

        if has_results is False and not has_header:
            return {
                "status": STATUS_NOT_FOUND,
                "label": "Not found (never existed)",
                "detail": "Syndication endpoint reports no profile data.",
            }

        if has_header and (entries_count == 0 or not entries_count):
            return {
                "status": STATUS_SUSPENDED_OR_DEACTIVATED,
                "label": "Exists, no public timeline (suspended / deactivated / protected / zero tweets)",
                "detail": "Syndication shows the handle but returns no public timeline entries.",
                "author_handle": parsed.get("screen_name") or username,
                "author_name": parsed.get("name"),
            }

        if (entries_count or 0) > 0 or has_header:
            return {
                "status": STATUS_LIVE,
                "label": "Live",
                "detail": f"Syndication returned {entries_count or 'some'} timeline entries.",
                "author_handle": parsed.get("screen_name") or username,
                "author_name": parsed.get("name"),
            }

        # Couldn't classify -- treat as inconclusive.
        return {
            "status": STATUS_ERROR, "label": "Inconclusive",
            "detail": "Syndication response shape did not match any known signature.",
        }

    def _ensure_guest_token(self) -> Optional[str]:
        # Guest tokens reportedly last 2-4 hours, but X also IP-binds them; rotate aggressively.
        if self._guest_token and (time.time() - self._guest_token_at) < 1800:
            return self._guest_token
        try:
            r = self._session.post(
                "https://api.x.com/1.1/guest/activate.json",
                headers={**DEFAULT_HEADERS, "Authorization": f"Bearer {X_PUBLIC_BEARER}"},
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException:
            return None
        if r.status_code != 200:
            return None
        try:
            tok = r.json().get("guest_token")
        except ValueError:
            return None
        if tok:
            self._guest_token = tok
            self._guest_token_at = time.time()
        return tok

    def _probe_graphql_user(self, username: str) -> Optional[dict]:
        if self._graphql_disabled:
            return None
        token = self._ensure_guest_token()
        if not token:
            return None

        variables = json.dumps(
            {"screen_name": username, "withSafetyModeUserFields": True},
            separators=(",", ":"),
        )
        # Feature flags X requires; missing flags cause 400. List taken from the
        # current public web bundle. New flags appear occasionally; if a 400
        # complains about a missing flag, add it here.
        features = json.dumps(
            {
                "hidden_profile_likes_enabled": True,
                "hidden_profile_subscriptions_enabled": True,
                "responsive_web_graphql_exclude_directive_enabled": True,
                "verified_phone_label_enabled": False,
                "subscriptions_verification_info_is_identity_verified_enabled": True,
                "subscriptions_verification_info_verified_since_enabled": True,
                "highlights_tweets_tab_ui_enabled": True,
                "responsive_web_twitter_article_notes_tab_enabled": True,
                "creator_subscriptions_tweet_preview_api_enabled": True,
                "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False,
                "responsive_web_graphql_timeline_navigation_enabled": True,
            },
            separators=(",", ":"),
        )

        url = f"https://api.x.com/graphql/{X_GRAPHQL_DOC_ID}/UserByScreenName"
        try:
            r = self._session.get(
                url,
                params={"variables": variables, "features": features},
                headers={
                    **DEFAULT_HEADERS,
                    "Authorization": f"Bearer {X_PUBLIC_BEARER}",
                    "x-guest-token": token,
                    "Content-Type": "application/json",
                },
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException:
            return None

        if r.status_code in (401, 403, 404):
            # doc_id likely rotated, or the token is bound to another IP.
            self._graphql_disabled = True
            return None
        if r.status_code != 200:
            return None
        try:
            payload = r.json()
        except ValueError:
            return None

        # GraphQL errors block: if features list is wrong we get 400 normally,
        # but if doc_id has been removed the response carries an `errors` array.
        if isinstance(payload.get("errors"), list) and payload["errors"]:
            self._graphql_disabled = True
            return None

        user = (payload.get("data") or {}).get("user") or {}
        result = user.get("result") if isinstance(user, dict) else None
        if not result:
            # `data: {}` -- collapsed suspend/deactivate/none. Syndication is the
            # tiebreaker for distinguishing these from never-existed.
            return {
                "status": STATUS_SUSPENDED_OR_DEACTIVATED,
                "label": "Suspended or deactivated",
                "detail": "GraphQL returned empty user data.",
            }
        typename = result.get("__typename")
        if typename == "UserUnavailable":
            reason = (result.get("reason") or "").lower()
            return {
                "status": STATUS_SUSPENDED_OR_DEACTIVATED,
                "label": "Suspended" if "suspend" in reason else "Unavailable",
                "detail": f"GraphQL UserUnavailable: {reason or 'no reason given'}",
            }
        legacy = result.get("legacy") or {}
        protected = bool(legacy.get("protected", False))
        return {
            "status": STATUS_PROTECTED if protected else STATUS_LIVE,
            "label": "Protected (locked)" if protected else "Live",
            "author_handle": legacy.get("screen_name") or username,
            "author_name": legacy.get("name"),
            "followers": _coerce_int(legacy.get("followers_count")),
            "following": _coerce_int(legacy.get("friends_count")),
            "statuses": _coerce_int(legacy.get("statuses_count")),
            "verified": bool(legacy.get("verified") or result.get("is_blue_verified")),
            "profile_created_at": legacy.get("created_at"),
            "bio": legacy.get("description"),
            "withheld_in_countries": ",".join(legacy.get("withheld_in_countries") or []) or None,
        }


# -------------------------------------------------------------------- helpers

_NEXT_DATA_RE = re.compile(
    r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>',
    re.DOTALL,
)


def _extract_next_data(html: str) -> Optional[dict]:
    if not html:
        return None
    m = _NEXT_DATA_RE.search(html)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except (ValueError, json.JSONDecodeError):
        return None


def _parse_syndication_json(d: dict) -> dict:
    """Walk the response JSON for known signals. Defensive: keys live in
    different places across schema revisions, so we BFS-search."""
    out: dict = {"has_results": None, "has_header": False, "entries_count": None}
    if not isinstance(d, dict):
        return out

    found_has_results = _find_first_key(d, "hasResults")
    if isinstance(found_has_results, bool):
        out["has_results"] = found_has_results

    header = _find_first_key(d, "headerProps")
    if isinstance(header, dict) and header:
        out["has_header"] = True
        if isinstance(header.get("screenName"), str):
            out["screen_name"] = header["screenName"]
        if isinstance(header.get("name"), str):
            out["name"] = header["name"]
        user = header.get("user") or {}
        if isinstance(user, dict):
            out.setdefault("screen_name", user.get("screen_name"))
            out.setdefault("name", user.get("name"))

    entries = _find_first_key(d, "entries")
    if isinstance(entries, list):
        out["entries_count"] = len(entries)

    return out


def _find_first_key(node, key):
    """BFS search for the first occurrence of ``key`` in nested dict/list."""
    stack = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if key in cur:
                return cur[key]
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return None


def _handle_from_author_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        p = urlparse(url)
        parts = [x for x in (p.path or "").split("/") if x]
        return parts[0] if parts else None
    except Exception:
        return None


def _coerce_int(x):
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None
