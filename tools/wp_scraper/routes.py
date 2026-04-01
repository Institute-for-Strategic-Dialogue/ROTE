# File: wp_scraper.py
import json
import re
import time
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from flask import Blueprint, render_template, request, send_file
from pandas.api.types import is_datetime64tz_dtype

wp_scraper_bp = Blueprint("wp_scraper", __name__, template_folder="templates")

from urllib.parse import urlparse

# Browser-ish defaults that most WP installs (and their WAFs) accept
DEFAULT_WP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.9",
    "DNT": "1",
}

# Sometimes WAFs want more generic Accepts; this helps on 406
ALT_WP_HEADERS = {
    "Accept": "application/json, */*; q=0.8",
}

def _origin_and_referer_for(endpoint: str) -> dict:
    try:
        u = urlparse(endpoint)
        origin = f"{u.scheme}://{u.netloc}"
        return {"Origin": origin, "Referer": origin}
    except Exception:
        return {}


# -----------------------
# Helpers
# -----------------------

def _excel_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Make tz-aware datetime columns Excel-safe (naive)."""
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    out = df.copy()
    for col in out.columns:
        try:
            if is_datetime64tz_dtype(out[col]):
                out[col] = out[col].dt.tz_localize(None)
        except Exception:
            try:
                out[col] = out[col].astype(str)
            except Exception:
                pass
    return out

_TAG_RE = re.compile(r"<.*?>", flags=re.S)

def strip_html(text: Optional[str]) -> str:
    if text is None:
        return ""
    return _TAG_RE.sub("", text).strip()

def _normalize_endpoint(site_input: str) -> str:
    """Accept either a base site URL or a full API endpoint."""
    s = (site_input or "").strip().rstrip("/")
    if not s:
        return ""
    if "/wp-json/" in s:
        return s
    return f"{s}/wp-json/wp/v2/posts"

def _backoff_sleep(attempt: int, base: float = 1.0, cap: float = 8.0):
    time.sleep(min(cap, base * (2 ** max(0, attempt - 1))))

def _flatten_post(
    post: Dict,
    *,
    resolve_names: bool,
    strip_html_flag: bool
) -> Dict:
    """Flatten a WP post object into a row suitable for Excel."""
    # Core
    out = {
        "id": post.get("id"),
        "date": post.get("date"),
        "date_gmt": post.get("date_gmt"),
        "modified": post.get("modified"),
        "modified_gmt": post.get("modified_gmt"),
        "slug": post.get("slug"),
        "status": post.get("status"),
        "type": post.get("type"),
        "link": post.get("link"),
        "author_id": post.get("author"),
        "featured_media": post.get("featured_media"),
        "comment_status": post.get("comment_status"),
        "ping_status": post.get("ping_status"),
        "sticky": bool(post.get("sticky", False)),
        "template": post.get("template"),
        "format": post.get("format"),
    }

    # Title / content / excerpt
    title_html = (post.get("title") or {}).get("rendered") or ""
    content_html = (post.get("content") or {}).get("rendered") or ""
    excerpt_html = (post.get("excerpt") or {}).get("rendered") or ""

    if strip_html_flag:
        out["title"] = strip_html(title_html)
        out["content"] = strip_html(content_html)
        out["excerpt"] = strip_html(excerpt_html)
    else:
        out["title_html"] = title_html
        out["content_html"] = content_html
        out["excerpt_html"] = excerpt_html

    # Taxonomy IDs
    out["category_ids"] = ",".join(map(str, post.get("categories", []) or []))
    out["tag_ids"] = ",".join(map(str, post.get("tags", []) or []))

    # Meta: flatten shallow keys; keep a json dump as well
    meta = post.get("meta")
    if isinstance(meta, dict):
        for k, v in meta.items():
            # simple scalars only; complex goes to meta_json
            if isinstance(v, (str, int, float, bool)) or v is None:
                out[f"meta_{k}"] = v
        out["meta_json"] = json.dumps(meta, ensure_ascii=False)
    else:
        out["meta_json"] = json.dumps(meta, ensure_ascii=False)

    # Optional: resolve names via _embedded
    if resolve_names:
        emb = post.get("_embedded") or {}
        # Author name
        try:
            out["author_name"] = (emb.get("author") or [{}])[0].get("name")
        except Exception:
            pass
        # Terms (categories, tags)
        try:
            terms = emb.get("wp:term") or []
            cat_names, tag_names = [], []
            for group in terms:
                for term in group or []:
                    if term.get("taxonomy") == "category":
                        cat_names.append(term.get("name"))
                    elif term.get("taxonomy") == "post_tag":
                        tag_names.append(term.get("name"))
            if cat_names:
                out["categories"] = ", ".join(sorted(set([c for c in cat_names if c])))
            if tag_names:
                out["tags"] = ", ".join(sorted(set([t for t in tag_names if t])))
        except Exception:
            pass

    return out

def _fetch_posts(
    endpoint: str,
    *,
    search: str,
    after_iso: str,
    before_iso: str,
    per_page: int,
    max_pages: int,
    resolve_names: bool,
    sleep_s: float,
    lang: str,
) -> Tuple[List[Dict], Dict]:
    if not endpoint:
        raise ValueError("No endpoint provided.")

    params = {
        "per_page": max(1, min(100, per_page)),

    }
    if search:
        params["search"] = search
    if after_iso:
        params["after"] = after_iso
    if before_iso:
        params["before"] = before_iso
    if resolve_names:
        # Empty value -> encodes as “&_embed=” which most WP treat like “&_embed”
        params["_embed"] = ""
    if lang:
        params["lang"] = lang

    session = requests.Session()
    session.headers.update(DEFAULT_WP_HEADERS)
    session.headers.update(_origin_and_referer_for(endpoint))

    records: List[Dict] = []
    total_pages_header = None
    total_items_header = None

    page = 1
    attempts = 0
    while page <= max_pages:
        p = params.copy()
        p["page"] = page
        print(p)
        def _do_request(local_params, tweak_headers=None):
            if tweak_headers:
                session.headers.update(tweak_headers)
            r = session.get(endpoint, params=local_params, timeout=30)
            return r

        try:
            r = _do_request(p)
            if r.status_code == 406:
                # Soften headers, try once
                r = _do_request(p, ALT_WP_HEADERS)

                # If still 406 and we asked for _embed, try dropping it
                if r.status_code == 406 and "_embed" in p:
                    p2 = dict(p)
                    p2.pop("_embed", None)
                    r = _do_request(p2, ALT_WP_HEADERS)

            if r.status_code in (429, 502, 503, 504):
                attempts += 1
                _backoff_sleep(attempts)
                continue

            r.raise_for_status()
        except Exception:
            break

        total_pages_header = total_pages_header or r.headers.get("X-WP-TotalPages")
        total_items_header = total_items_header or r.headers.get("X-WP-Total")

        try:
            posts = r.json()
        except Exception:
            break
        if not posts:
            break

        for post in posts:
            records.append(
                _flatten_post(post, resolve_names=resolve_names, strip_html_flag=True)
            )

        # If header declares a hard page limit, respect it
        if total_pages_header:
            try:
                if page >= int(total_pages_header):
                    break
            except Exception:
                pass

        page += 1
        attempts = 0
        time.sleep(max(0.0, float(sleep_s)))

    meta = {
        "endpoint": endpoint,
        "params": {k: v for k, v in params.items() if k != "_embed"} | ({"_embed": "on"} if resolve_names else {}),
        "total_pages_header": total_pages_header,
        "total_items_header": total_items_header,
        "collected": len(records),
        "pages_fetched": page,
    }
    return records, meta


# -----------------------
# Route
# -----------------------

@wp_scraper_bp.route("/", methods=["GET", "POST"])
def wp_scraper():
    results = {}
    if request.method == "POST":
        # Inputs
        site_input = request.form.get("site_input") or ""
        endpoint = _normalize_endpoint(site_input)
        search = (request.form.get("search") or "").strip()
        after = (request.form.get("after") or "").strip()      # YYYY-MM-DD or empty
        before = (request.form.get("before") or "").strip()    # YYYY-MM-DD or empty
        per_page = int(request.form.get("per_page") or 100)
        max_pages = int(request.form.get("max_pages") or 50)
        resolve_names = bool(request.form.get("resolve_names"))
        sleep_s = float(request.form.get("sleep_s") or 0.5)
        output_mode = (request.form.get("output_mode") or "table").strip()
        lang = (request.form.get("lang") or "").strip()

        # Convert dates to ISO8601 if provided (append times to include whole days)
        after_iso = f"{after}T00:00:00Z" if after else ""
        before_iso = f"{before}T23:59:59Z" if before else ""

        if not endpoint:
            return render_template("wp_scraper.html", results={"error": "Please enter a valid site URL or WP REST endpoint."})

        try:
            rows, meta = _fetch_posts(
                endpoint,
                search=search,
                after_iso=after_iso,
                before_iso=before_iso,
                per_page=per_page,
                max_pages=max_pages,

                resolve_names=resolve_names,
                sleep_s=sleep_s,
                lang=lang,
            )
        except Exception as e:
            return render_template("wp_scraper.html", results={"error": f"Fetch failed: {e}"})

        df = pd.DataFrame(rows)

        # Excel download
        if output_mode == "excel":
            out = BytesIO()
            with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
                meta_df = pd.DataFrame([{
                    "endpoint": meta.get("endpoint"),
                    "search": search,
                    "after": after,
                    "before": before,

                    "resolve_names": resolve_names,
                    "sleep_s": sleep_s,
                    "per_page": per_page,
                    "max_pages": max_pages,
                    "lang": lang,
                    "total_items_header": meta.get("total_items_header"),
                    "total_pages_header": meta.get("total_pages_header"),
                    "collected": meta.get("collected"),
                    "pages_fetched": meta.get("pages_fetched"),
                }])
                _excel_safe(meta_df).to_excel(writer, sheet_name="META", index=False)
                _excel_safe(df).to_excel(writer, sheet_name="POSTS", index=False)
            out.seek(0)
            return send_file(
                out,
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                as_attachment=True,
                download_name="wp_posts_export.xlsx",
            )

        # Otherwise: on-page preview + summary
        preview = df.head(100).to_dict(orient="records") if not df.empty else []
        results = {
            "ok": True,
            "meta": meta,
            "collected": int(len(df)),
            "preview": preview,
        }

    return render_template("wp_scraper.html", results=results)
