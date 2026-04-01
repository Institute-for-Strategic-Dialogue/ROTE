# File: serp_social_blueprint.py
"""
Flask Blueprint: Automated Google searches via SerpAPI to extract social-media profiles.

Features
- Accepts queries from a pasted list or uploaded CSV.
- Or generates preset queries from Countries × Sites using built-in patterns.
- Calls SerpAPI Google engine, paginates, and parses results.
- Extracts platform, normalized profile URL, and handle where possible.
- Carries all input columns forward (including the exact `query`) into results for easy sorting.
- Records per-row status and inline error text for failures.
- Exports results as on-page table, CSV, or Excel.

Environment
- Set SERPAPI_KEY in the environment, or provide an API key in the form field `api_key`.

Template
- Expects a Jinja template at `templates/serp_social.html` that reads `results` dict.

POST Form Fields (suggested UI names)
- mode: "preset" | "paste" | "file"
- api_key: optional (falls back to env)
- output_mode: "table" | "csv" | "excel"
- countries: comma-separated (used in preset mode)
- sites: comma-separated (used in preset mode)
- page_limit: integer (max pages per query; default 3) — 100 results per page
- throttle_ms: integer milliseconds between page calls (default 1000)
- google_domain, gl, hl, location: optional overrides (defaults provided)
- pasted: free-text (CSV or plain), used in paste mode
- file: uploaded CSV file, used in file mode

CSV Input Expectations
- If a `query` column exists, it is used as-is.
- Otherwise, if `country` and `site` columns exist, the blueprint builds a query using the pattern rules below.
- All other columns are carried forward to output rows unchanged.

Output Columns (fixed + dynamic)
- Always includes: query, country, site, status, error, page_start, position, title, link,
  snippet, domain, platform, is_profile, profile_type, profile_handle, normalized_profile_url,
  fetched_at, serpapi_search_url.
- Plus any additional input columns, preserved verbatim.
"""
import csv
import io
import os
import re
import json
import time
import math
import html
import datetime as dt
from collections import OrderedDict
from urllib.parse import urlparse, unquote

import pandas as pd
import requests
from flask import Blueprint, render_template, request, send_file

serp_social_bp = Blueprint("serp_social", __name__, template_folder="templates")

SERPAPI_URL = "https://serpapi.com/search"
REQUEST_TIMEOUT = 45  # seconds
DEFAULT_PAGE_LIMIT = 3  # 3 × 100 = up to 300 results per query
MAX_PAGE_LIMIT = 1000     # hard guardrail
DEFAULT_THROTTLE_MS = 1000

DEFAULT_COUNTRIES = ["China", "Iran", "Russia"]
DEFAULT_SITES = [
    "instagram.com",
    "facebook.com",
    "youtube.com",
    "t.me",
    "tiktok.com",
    "threads.net",
]

# Base Google params. Change-able via form.
BASE_PARAMS = {
    "engine": "google",
    "location": "United States",
    "google_domain": "google.com",
    "gl": "us",
    "hl": "en",
    "start": "0",
    "num": "10",
}

# ----------------------------
# Query construction
# ----------------------------

def build_query(site: str, country: str) -> str:
    site = (site or "").strip().lower()
    country = (country or "").strip()
    if not site:
        return ""
    if site in {"instagram.com", "facebook.com", "threads.net"}:
        return f'"{country} state-controlled media" site:{site}'
    if site == "tiktok.com":
        return f'"{country} state-affiliated media" site:{site}'
    if site == "youtube.com":
        return f'"is funded in whole or in part by the {country} government." site:{site}'
    # Fallback: simple country phrase
    return f'"{country} state-controlled media" site:{site}'

# ----------------------------
# CSV / input handling
# ----------------------------

def _sniff_csv_and_rows(raw: str) -> list[dict]:
    """Robust CSV sniffer that returns a list of row dicts.
    Falls back to simple comma-splitting if sniffer fails.
    """
    if not raw or not raw.strip():
        return []
    buf = io.StringIO(raw)
    sample = buf.read(4096)
    buf.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample) if sample.strip() else csv.excel
        reader = csv.DictReader(buf, dialect=dialect)
    except Exception:
        buf.seek(0)
        reader = csv.DictReader(buf)
    rows = []
    if reader.fieldnames:
        for row in reader:
            rows.append({k.strip(): (v or "").strip() for k, v in row.items()})
    return rows


def _parse_paste_to_rows(text: str) -> list[dict]:
    """Accept pasted CSV or simple lines of `country,site` or raw `query`.
    - If header contains `query`, use as-is.
    - Otherwise attempt to map first two columns to country/site.
    """
    text = text or ""
    # rows = _sniff_csv_and_rows(text)
    # if rows:
    #     return rows
    # Fallback: treat each non-empty line as either `query` or `country,site`.
    out = []
    for ln in (ln.strip() for ln in text.splitlines()):
        if not ln:
            continue
        if "," in ln:
            parts = [p.strip() for p in ln.split(",", 2)]
            if len(parts) >= 2:
                out.append({"country": parts[0], "site": parts[1]})
                continue
        out.append({"query": ln})
    return out


def _extract_input_rows(mode: str) -> list[dict]:
    if mode == "file":
        up = request.files.get("file")
        if up:
            raw = up.read()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            return _sniff_csv_and_rows(raw)
        return []
    if mode == "paste":
        text = request.form.get("pasted", "")
        return _parse_paste_to_rows(text)
    # preset
    countries = request.form.get("countries", ",".join(DEFAULT_COUNTRIES))
    sites = request.form.get("sites", ",".join(DEFAULT_SITES))
    countries = [c.strip() for c in countries.split(",") if c.strip()]
    sites = [s.strip() for s in sites.split(",") if s.strip()]
    out = []
    for c in countries:
        for s in sites:
            out.append({"country": c, "site": s})
    return out

# ----------------------------
# Profile extraction
# ----------------------------

_FB_BANNED = {"share", "photo.php", "watch", "events", "groups", "marketplace", "reel", "reels", "gaming"}
_IG_BANNED = {"p", "reel", "reels", "explore", "stories", "tv", "accounts"}
_YT_PREFIX_TYPES = {
    "@": "handle",
    "channel": "channel",
    "c": "custom",
    "user": "user",
}


def parse_profile(link: str) -> dict:
    """Return platform parsing with best-effort profile extraction.
    Schema:
    - platform: facebook | instagram | youtube | telegram | tiktok | threads | other
    - is_profile: bool
    - profile_type: free text (page, handle, channel, etc.)
    - profile_handle: extracted handle or id
    - normalized_profile_url: canonical profile URL if derivable
    - domain: netloc
    """
    try:
        u = urlparse(link)
    except Exception:
        return {
            "platform": None,
            "domain": None,
            "is_profile": False,
            "profile_type": None,
            "profile_handle": None,
            "normalized_profile_url": None,
        }

    netloc = (u.netloc or "").lower()
    path = unquote(u.path or "/").strip("/")
    segs = [s for s in path.split("/") if s]

    def out(platform=None, is_profile=False, ptype=None, handle=None, norm=None):
        return {
            "platform": platform,
            "domain": netloc,
            "is_profile": is_profile,
            "profile_type": ptype,
            "profile_handle": handle,
            "normalized_profile_url": norm,
        }

    # Instagram
    if "instagram.com" in netloc:
        if not segs:
            return out("instagram")
        first = segs[0].lower()
        if first in _IG_BANNED:
            return out("instagram")
        handle = first
        norm = f"https://www.instagram.com/{handle}/"
        return out("instagram", True, "user", handle, norm)

    # Threads
    if "threads.net" in netloc:
        if segs and segs[0].startswith("@"):
            handle = segs[0].lstrip("@")
            norm = f"https://www.threads.net/@{handle}"
            return out("threads", True, "handle", handle, norm)
        if segs:
            handle = segs[0]
            norm = f"https://www.threads.net/@{handle}"
            return out("threads", True, "handle", handle, norm)
        return out("threads")

    # Facebook
    if "facebook.com" in netloc:
        if not segs:
            return out("facebook")
        first = segs[0].lower()
        if first in _FB_BANNED:
            return out("facebook")
        if first == "profile.php":
            # profile.php?id=12345
            from urllib.parse import parse_qs
            qs = parse_qs(u.query or "")
            pid = (qs.get("id") or [None])[0]
            if pid:
                norm = f"https://www.facebook.com/profile.php?id={pid}"
                return out("facebook", True, "profile_id", pid, norm)
            return out("facebook")
        if first == "pages" and len(segs) >= 3:
            # /pages/Page-Name/123456789
            handle = segs[2]
            norm = f"https://www.facebook.com/pages/{segs[1]}/{handle}"
            return out("facebook", True, "page_id", handle, norm)
        # Default: /<pagename>
        handle = first
        norm = f"https://www.facebook.com/{handle}"
        return out("facebook", True, "page", handle, norm)

    # YouTube
    if "youtube.com" in netloc or "youtu.be" in netloc:
        if not segs:
            return out("youtube")
        head = segs[0]
        # Handle forms: /@handle, /channel/UC..., /c/custom, /user/legacy
        if head.startswith("@"):
            handle = head.lstrip("@")
            norm = f"https://www.youtube.com/@{handle}"
            return out("youtube", True, "handle", handle, norm)
        if head in _YT_PREFIX_TYPES and len(segs) >= 2:
            ptype = _YT_PREFIX_TYPES[head]
            handle = segs[1]
            base = {
                "channel": "https://www.youtube.com/channel/",
                "custom": "https://www.youtube.com/c/",
                "user": "https://www.youtube.com/user/",
            }.get(ptype, "https://www.youtube.com/")
            norm = f"{base}{handle}"
            return out("youtube", True, ptype, handle, norm)
        return out("youtube")

    # Telegram
    if netloc.endswith("t.me"):
        if not segs:
            return out("telegram")
        first = segs[0]
        if first == "s" and len(segs) >= 2:
            handle = segs[1]
        else:
            handle = first
        norm = f"https://t.me/{handle}"
        return out("telegram", True, "channel", handle, norm)

    # TikTok
    if "tiktok.com" in netloc:
        if segs and segs[0].startswith("@"):
            handle = segs[0].lstrip("@")
            norm = f"https://www.tiktok.com/@{handle}"
            return out("tiktok", True, "handle", handle, norm)
        return out("tiktok")

    return out(None, False, None, None, None)

# ----------------------------
# SerpAPI caller
# ----------------------------

def serpapi_search(api_key: str, query: str, start: int, gparams: dict) -> tuple[dict | None, str | None]:
    """Call SerpAPI. Return (json, error_text)."""
    params = {
        **gparams,
        "q": query,
        "start": str(start),
        "api_key": api_key,
        "no_cache": "true"
    }
    print(f"SerpAPI search: q={query} start={start}")
    try:
        r = requests.get(SERPAPI_URL, params=params, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        return None, f"RequestException: {e}"
    if r.status_code != 200:
        # Try to parse error body if possible
        try:
            body = r.json()
            msg = body.get("error") or json.dumps(body)[:500]
        except Exception:
            msg = (r.text or "").strip()[:500]
        return None, f"HTTP {r.status_code}: {msg}"
    try:
        data = r.json()
        if isinstance(data, dict) and data.get("error"):
            return None, str(data.get("error"))
        return data, None
    except Exception as e:
        return None, f"JSON decode error: {e}"


# ----------------------------
# Dataframe construction
# ----------------------------

def _standardize_columns(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    for c in order:
        if c not in df.columns:
            df[c] = None
    return df[order + [c for c in df.columns if c not in order]]


def _now_iso() -> str:
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()


# ----------------------------
# Flask endpoint
# ----------------------------

@serp_social_bp.route("/", methods=["GET", "POST"])
def serp_social():
    results: dict = {}
    if request.method == "POST":
        mode = request.form.get("mode", "preset")  # preset | paste | file
        output_mode = request.form.get("output_mode", "table")  # table | csv | excel
        

        # SerpAPI key resolution
        api_key = (request.form.get("api_key") or os.environ.get("SERPAPI_KEY") or "").strip()
        if not api_key:
            # Immediate feedback path
            results["error"] = "Missing SerpAPI API key. Provide it in the form or set SERPAPI_KEY."
            return render_template("serp_social.html", results=results)

        # Google params (allow overrides)
        gparams = dict(BASE_PARAMS)
        for k in ["google_domain", "gl", "hl", "location"]:
            v = (request.form.get(k) or gparams.get(k) or "").strip()
            if v:
                gparams[k] = v

        page_limit = request.form.get("page_limit")
        try:
            page_limit = max(1, min(int(page_limit), MAX_PAGE_LIMIT)) if page_limit else DEFAULT_PAGE_LIMIT
        except Exception:
            page_limit = DEFAULT_PAGE_LIMIT

        throttle_ms = request.form.get("throttle_ms")
        try:
            throttle_ms = max(0, int(throttle_ms)) if throttle_ms else DEFAULT_THROTTLE_MS
        except Exception:
            throttle_ms = DEFAULT_THROTTLE_MS

        # Build input rows
        inputs = _extract_input_rows(mode)

        # If CSV provided, carry forward arbitrary columns
        # Build queries for each row
        expanded_rows: list[dict] = []
        for row in inputs:
            row = {k: (v if v is not None else "") for k, v in row.items()}
            country = row.get("country", "")
            site = row.get("site", "")
            query = row.get("query") or build_query(site, country)
            if not query:
                # Record a skipped row with inline status
                expanded_rows.append({
                    **row,
                    "query": None,
                    "status": "Skipped",
                    "error": "Missing query and insufficient fields to build one.",
                })
                continue
            expanded_rows.append({**row, "query": query})

        # Perform searches
        out_rows: list[dict] = []
        total_queries = 0
        total_results = 0

        for src in expanded_rows:
            query = src.get("query")
            if not query:
                out_rows.append({**src})
                continue
            
            #save to file every 15 queries
            if total_queries > 0 and total_queries % 15 == 0:
                df_partial = pd.DataFrame(out_rows)
                base_cols = [
                    "query", "country", "site", "status", "error",
                    "page_start", "position", "title", "link", "snippet",
                    "domain", "platform", "is_profile", "profile_type", "profile_handle",
                    "normalized_profile_url", "fetched_at", "serpapi_search_url",
                ]
                df_partial = _standardize_columns(df_partial, base_cols)
                partial_filename = f"serp_social_partial_{total_queries}_queries.csv"
                df_partial.to_csv(partial_filename, index=False)
                print(f"Saved partial results to {partial_filename} {len(out_rows)} rows.")
            any_results = False
            for p in range(page_limit):
                start = p * 10
                data, err = serpapi_search(api_key, query, start, gparams)
                fetched_at = _now_iso()
                serpapi_link = f"{SERPAPI_URL}?"  # base for debugging (full link is long; SerpAPI returns link in metadata)

                if err or not data:
                    out_rows.append({
                        **src,
                        "status": "Error",
                        "error": err or "Unknown error",
                        "page_start": start,
                        "position": None,
                        "title": None,
                        "link": None,
                        "snippet": None,
                        "domain": None,
                        "platform": None,
                        "is_profile": False,
                        "profile_type": None,
                        "profile_handle": None,
                        "normalized_profile_url": None,
                        "fetched_at": fetched_at,
                        "serpapi_search_url": serpapi_link,
                    })
                    break  # Stop paging this query on error

                org = data.get("organic_results") or []
                if not org:
                    if p == 0:
                        # Record a No Results row for the first page
                        out_rows.append({
                            **src,
                            "status": "No Results",
                            "error": None,
                            "page_start": start,
                            "position": None,
                            "title": None,
                            "link": None,
                            "snippet": None,
                            "domain": None,
                            "platform": None,
                            "is_profile": False,
                            "profile_type": None,
                            "profile_handle": None,
                            "normalized_profile_url": None,
                            "fetched_at": fetched_at,
                            "serpapi_search_url": serpapi_link,
                        })
                    break

                    
                any_results = True
                for idx, res in enumerate(org, start=1):
                    title = res.get("title")
                    link = res.get("link")
                    snippet = res.get("snippet")

                    prof = parse_profile(link or "") if link else {}

                    out_rows.append({
                        **src,
                        "status": "OK",
                        "error": None,
                        "page_start": start,
                        "position": start + idx,  # 1-based rank across pages
                        "title": title,
                        "link": link,
                        "snippet": snippet,
                        "domain": prof.get("domain"),
                        "platform": prof.get("platform"),
                        "is_profile": prof.get("is_profile"),
                        "profile_type": prof.get("profile_type"),
                        "profile_handle": prof.get("profile_handle"),
                        "normalized_profile_url": prof.get("normalized_profile_url"),
                        "fetched_at": fetched_at,
                        "serpapi_search_url": serpapi_link,
                    })

                # Throttle between pages
                if p < page_limit - 1:
                    time.sleep(throttle_ms / 1000.0)

            total_queries += 1
            if any_results:
                total_results += 1

        # Build DataFrame and enforce base order
        df = pd.DataFrame(out_rows)
        base_cols = [
            "query", "country", "site", "status", "error",
            "page_start", "position", "title", "link", "snippet",
            "domain", "platform", "is_profile", "profile_type", "profile_handle",
            "normalized_profile_url", "fetched_at", "serpapi_search_url",
        ]
        df = _standardize_columns(df, base_cols)

        if output_mode == "excel":
            mem = io.BytesIO()
            with pd.ExcelWriter(mem, engine="xlsxwriter") as writer:
                df.to_excel(writer, sheet_name="serp_social", index=False)
            mem.seek(0)
            return send_file(
                mem,
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                as_attachment=True,
                download_name="serp_social_results.xlsx",
            )

        if output_mode == "csv":
            out = io.StringIO()
            df.to_csv(out, index=False)
            mem = io.BytesIO(out.getvalue().encode("utf-8"))
            return send_file(
                mem,
                mimetype="text/csv",
                as_attachment=True,
                download_name="serp_social_results.csv",
            )

        # Otherwise render table summary
        results["table_rows"] = df.to_dict(orient="records")
        results["total_rows"] = len(df)
        results["error_rows"] = int((df["status"] == "Error").sum())
        results["no_results_rows"] = int((df["status"] == "No Results").sum())
        results["ok_rows"] = int((df["status"] == "OK").sum())
        results["queries_submitted"] = total_queries
        results["queries_with_results"] = total_results

        return render_template("serp_social.html", results=results)

    # GET
    return render_template("serp_social.html", results={})
