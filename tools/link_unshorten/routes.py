# File: link_unshorten_blueprint.py
"""
Flask Blueprint: Link Unshortener (CSV-only input)

- Upload a CSV and specify which column(s) contain URLs (defaults to `url`).
- Preserves all original columns verbatim.
- Appends, for each specified URL column, these fields:
  - final_url_<col>
  - http_status_<col>
  - redirect_chain_<col> ("start -> ... -> final")
  - resolved_domain_<col>
  - error_<col>
- Aggregates a per-row status (`row_status`) and `processed_url_columns` for quick filtering.
- Exports results as an on-page table, CSV, or Excel.

Template
- Expects `templates/link_unshorten.html` (provided separately) that reads a `results` dict.

POST Form Fields
- file: CSV upload (required)
- url_columns: comma-separated column names containing URLs (default: `url`)
- method: `head` (default) or `get`
- max_redirects: int, default 10
- timeout: seconds, default 10
- output_mode: `table` | `csv` | `excel`
"""
import csv
import io
import re
from urllib.parse import urlparse, urljoin   # add urljoin
import html     
import pandas as pd
import requests
from flask import Blueprint, render_template, request, send_file

link_unshorten_bp = Blueprint("link_unshorten", __name__, template_folder="templates")

REQUEST_TIMEOUT_DEFAULT = 10
MAX_REDIRECTS_DEFAULT = 10
MAX_REDIRECTS_HARD = 20
MAX_ROWS = 200000  # guardrail

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)
MAX_HTML_BYTES = 200_000
SHORTENER_HINTS = {
    "t.co", "bit.ly", "tinyurl.com", "ow.ly", "buff.ly",
    "lnkd.in", "ift.tt", "trib.al", "goo.gl", "t.ly", "is.gd",
}

META_REFRESH_RE = re.compile(
    r'<meta[^>]+http-equiv=["\']?refresh["\']?[^>]+content=["\']?\s*\d+\s*;\s*url\s*=\s*([^"\'>\s]+)',
    re.IGNORECASE,
)
JS_ASSIGN_RE = re.compile(
    r'location\.(?:href|replace)\s*=\s*["\']([^"\']+)["\']',
    re.IGNORECASE,
)
JS_CALL_RE = re.compile(
    r'location\.(?:replace|assign)\s*\(\s*["\']([^"\']+)["\']\s*\)',
    re.IGNORECASE,
)
TITLE_URL_RE = re.compile(
    r'<title>\s*(https?://[^<\s]+)\s*</title>',
    re.IGNORECASE,
)

def _extract_redirect_from_html(html_text: str, base_url: str) -> str | None:
    """Return a URL found in meta-refresh or JS location redirects; fallback to <title> if it’s a URL."""
    if not html_text:
        return None
    txt = html.unescape(html_text).replace("\\/", "/")
    for rx in (META_REFRESH_RE, JS_ASSIGN_RE, JS_CALL_RE, TITLE_URL_RE):
        m = rx.search(txt)
        if m:
            raw = (m.group(1) or "").strip()
            try:
                return urljoin(base_url, raw)
            except Exception:
                return raw
    return None

# ----------------------------
# Unshortening core
# ----------------------------

def _is_url(x: str) -> bool:
    if not x or not isinstance(x, str):
        return False
    return x.startswith("http://") or x.startswith("https://")


def _resolve_domain(url: str) -> str | None:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return None


def _unshorten(url: str, method: str, timeout: int, max_redirects: int) -> tuple[str | None, int | None, list[str], str | None]:
    """Follow redirects and return (final_url, status_code, chain, error).
    Enhanced to parse HTML-based redirects (noscript meta-refresh, JS location.*) common behind Cloudflare/t.co.
    """
    if not _is_url(url):
        return None, None, [], "Not a URL"

    # Look like a real browser
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://t.co/",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    allow_redirects = True

    def run_head():
        try:
            r = requests.head(url, headers=headers, timeout=timeout, allow_redirects=allow_redirects)
            hist = r.history or []
            chain_local = [h.url for h in hist] + [r.url]
            return r.url, r.status_code, chain_local, None, r
        except requests.RequestException as e:
            return None, None, [], str(e), None

    def run_get():
        try:
            # stream=False so we can read a tiny HTML body if needed
            r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=allow_redirects)
            hist = r.history or []
            chain_local = [h.url for h in hist] + [r.url]
            return r.url, r.status_code, chain_local, None, r
        except requests.RequestException as e:
            return None, None, [], str(e), None

    # Try HEAD first if requested, but fall back to GET aggressively
    final_url, code, chain, err, resp = (None, None, [], None, None)
    if method.lower() == "head":
        final_url, code, chain, err, resp = run_head()
        if err or code in (400, 401, 403, 405) or (code in (301, 302, 303, 307, 308) and len(chain) <= 1):
            final_url, code, chain, err, resp = run_get()
    else:
        final_url, code, chain, err, resp = run_get()

    # If we landed on an HTML page (often t.co) without a real 3xx hop, sniff for a meta/JS redirect.
    if resp is not None and (code == 200 or code is None):
        ct = (resp.headers.get("Content-Type") or "").lower()
        netloc = _resolve_domain(url) or ""
        if ("text/html" in ct or netloc in SHORTENER_HINTS) and len(chain) <= 1:
            try:
                html_text = resp.text[:MAX_HTML_BYTES]
            except Exception:
                html_text = ""
            extracted = _extract_redirect_from_html(html_text, resp.url or url)
            if extracted and extracted != final_url:
                chain = (chain or [url]) + ([extracted] if (not chain or chain[-1] != extracted) else [])
                final_url = extracted
                err = None

    # Cap redirect chain length
    if chain and len(chain) > max_redirects + 1:
        chain = chain[: max_redirects + 1]
        err = f"Exceeded max_redirects={max_redirects}"

    return final_url, code, chain, err


# ----------------------------
# Flask endpoint
# ----------------------------

@link_unshorten_bp.route("/", methods=["GET", "POST"])
def link_unshorten():
    results: dict = {}

    if request.method == "POST":
        up = request.files.get("file")
        if not up:
            results["error"] = "Please upload a CSV file."
            return render_template("link_unshorten.html", results=results)

        raw = up.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")

        rows = pd.read_csv(io.StringIO(raw)).to_dict(orient="records")
        url_columns = [c.strip() for c in (request.form.get("url_columns", "url") or "").split(",") if c.strip()]
        method = (request.form.get("method") or "head").lower()
        if method not in ("head", "get"):
            method = "head"
        try:
            timeout = int(request.form.get("timeout", REQUEST_TIMEOUT_DEFAULT))
        except Exception:
            timeout = REQUEST_TIMEOUT_DEFAULT
        try:
            max_redirects = int(request.form.get("max_redirects", MAX_REDIRECTS_DEFAULT))
        except Exception:
            max_redirects = MAX_REDIRECTS_DEFAULT
        max_redirects = max(1, min(max_redirects, MAX_REDIRECTS_HARD))

        # Process
        out_rows: list[dict] = []
        appended_cols: set[str] = set()
        ok_rows = 0
        error_rows = 0

        for src in rows:
            # Preserve original
            base = dict(src)
            processed_cols = []
            any_ok = False
            any_err = False

            for col in url_columns:
                val = (src.get(col) or "").strip()
                if not val:
                    # Still append empty fields for consistency
                    base[f"final_url_{col}"] = None
                    base[f"http_status_{col}"] = None
                    base[f"redirect_chain_{col}"] = None
                    base[f"resolved_domain_{col}"] = None
                    base[f"error_{col}"] = None
                    appended_cols.update({
                        f"final_url_{col}", f"http_status_{col}", f"redirect_chain_{col}", f"resolved_domain_{col}", f"error_{col}"
                    })
                    continue

                final_url, code, chain, err = _unshorten(val, method, timeout, max_redirects)
                base[f"final_url_{col}"] = final_url
                base[f"http_status_{col}"] = code
                base[f"redirect_chain_{col}"] = " -> ".join(chain) if chain else None
                base[f"resolved_domain_{col}"] = _resolve_domain(final_url) if final_url else None
                base[f"error_{col}"] = err
                appended_cols.update({
                    f"final_url_{col}", f"http_status_{col}", f"redirect_chain_{col}", f"resolved_domain_{col}", f"error_{col}"
                })

                processed_cols.append(col)
                if final_url and not err:
                    any_ok = True
                if err:
                    any_err = True

            base["processed_url_columns"] = ", ".join(processed_cols)
            base["row_status"] = (
                "OK" if any_ok else ("Error" if any_err else "No URL")
            )
            if base["row_status"] == "OK":
                ok_rows += 1
            elif base["row_status"] == "Error":
                error_rows += 1

            out_rows.append(base)

        # Build DataFrame preserving original column order, then appended
        df = pd.DataFrame(out_rows)
        original_cols = list(rows[0].keys())
        ordered_cols = original_cols.copy()
        # Ensure these meta columns appear next
        meta_cols = ["processed_url_columns", "row_status"]
        for c in meta_cols:
            if c not in ordered_cols:
                ordered_cols.append(c)
        # Then all appended cols in sorted order for stability
        for c in sorted(appended_cols):
            if c not in ordered_cols:
                ordered_cols.append(c)
        # Plus any stragglers that may have been introduced
        for c in df.columns:
            if c not in ordered_cols:
                ordered_cols.append(c)
        df = df[ordered_cols]

        # Output modes
        output_mode = request.form.get("output_mode", "table")
        if output_mode == "excel":
            mem = io.BytesIO()
            with pd.ExcelWriter(mem, engine="xlsxwriter") as writer:
                df.to_excel(writer, sheet_name="unshortened", index=False)
            mem.seek(0)
            return send_file(
                mem,
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                as_attachment=True,
                download_name="link_unshorten_results.xlsx",
            )

        if output_mode == "csv":
            out = io.StringIO()
            df.to_csv(out, index=False)
            mem = io.BytesIO(out.getvalue().encode("utf-8"))
            return send_file(
                mem,
                mimetype="text/csv",
                as_attachment=True,
                download_name="link_unshorten_results.csv",
            )

        # Otherwise render table
        results["table_rows"] = df.to_dict(orient="records")
        results["total_rows"] = len(df)
        results["ok_rows"] = ok_rows
        results["error_rows"] = error_rows
        return render_template("link_unshorten.html", results=results)

    # GET
    return render_template("link_unshorten.html", results={})