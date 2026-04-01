# File: yt_oembed.py
import csv
import io
import re
import json
from collections import OrderedDict
from urllib.parse import quote

import pandas as pd
import requests
from flask import Blueprint, render_template, request, send_file

yt_oembed_bp = Blueprint("yt_oembed", __name__, template_folder="templates")

OEMBED_BASE = "https://www.youtube.com/oembed?format=json&url="
REQUEST_TIMEOUT = 5  # seconds
MAX_URLS = 200000      # simple guardrail

URL_REGEX = re.compile(
    r"""(?P<url>https?://[^\s<>"']+)""",
    re.IGNORECASE
)

def extract_urls_from_text(text: str) -> list[str]:
    """Extract URLs from free text; preserve order, drop dups."""
    if not text:
        return []
    seen = OrderedDict()
    for m in URL_REGEX.finditer(text):
        u = m.group("url").strip()
        # Strip trailing punctuation that commonly clings to URLs
        u = u.rstrip(").,;\"'“”’")
        if u and u not in seen:
            seen[u] = None
    return list(seen.keys())

def extract_urls_from_file(f) -> list[str]:
    """
    Accepts .txt (newline URLs) or .csv.
    CSV: uses 'url' column if present, otherwise picks the first column,
    plus any extra URLs found in raw text.
    """
    filename = (getattr(f, "filename", "") or "").lower()
    raw = f.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")

    urls = []
    if filename.endswith(".csv"):
        buf = io.StringIO(raw)
        sample = buf.read(4096)
        buf.seek(0)
        dialect = csv.Sniffer().sniff(sample) if sample.strip() else csv.excel
        reader = csv.DictReader(buf, dialect=dialect)
        if reader.fieldnames:
            lowered = [c.lower() for c in reader.fieldnames]
            url_col = None
            if "url" in lowered:
                url_col = reader.fieldnames[lowered.index("url")]
            else:
                url_col = reader.fieldnames[0]
            buf.seek(0)
            reader = csv.DictReader(buf, dialect=dialect)
            for row in reader:
                cell = (row.get(url_col) or "").strip()
                urls.extend(extract_urls_from_text(cell) if "http" not in cell else [cell])
        # Also sweep entire CSV for any embedded URLs
        urls.extend(extract_urls_from_text(raw))
    else:
        # Treat as plaintext list; still sweep for embedded URLs
        lines = [ln.strip() for ln in raw.splitlines()]
        for ln in lines:
            if not ln:
                continue
            if ln.startswith("http"):
                urls.append(ln)
            else:
                urls.extend(extract_urls_from_text(ln))

    # Deduplicate while preserving order, limit length
    deduped = list(OrderedDict.fromkeys(u.strip() for u in urls if u.strip()))
    return deduped[:MAX_URLS]

def call_oembed(target_url: str) -> dict:
    """
    Call YouTube oEmbed. Returns a normalized result dict with status info.
    """
    oembed_url = OEMBED_BASE + quote(target_url, safe="")
    row = {
        "input_url": target_url,
        "oembed_request": oembed_url,
        "status": None,
        "status_code": None,
        "error": None,
        "title": None,
        "author_name": None,
        "author_url": None,
        "type": None,
        "provider_name": None,
        "provider_url": None,
        "thumbnail_url": None,
        "thumbnail_width": None,
        "thumbnail_height": None,
        "width": None,
        "height": None,
        "version": None,
        "embed_src": None,  # parsed from html
    }
    try:
        r = requests.get(oembed_url, timeout=REQUEST_TIMEOUT)
        row["status_code"] = r.status_code

        if r.status_code == 200:
            data = r.json()
            # Copy known fields defensively
            for k in [
                "title", "author_name", "author_url", "type", "provider_name",
                "provider_url", "thumbnail_url", "thumbnail_width",
                "thumbnail_height", "width", "height", "version"
            ]:
                row[k] = data.get(k)

            html = data.get("html") or ""
            row["embed_src"] = _extract_embed_src(html)
            row["status"] = "Found"
            return row

        # Non-200 status mapping
        if r.status_code == 404:
            row["status"] = "Not Found"
        elif r.status_code == 400:
            row["status"] = "Bad Request"
        else:
            row["status"] = "Error"
        # Include short error text if present
        try:
            msg = r.json()
            row["error"] = msg.get("error", None) or json.dumps(msg)[:500]
        except Exception:
            row["error"] = (r.text or "").strip()[:500]
        return row

    except requests.RequestException as e:
        row["status"] = "Error"
        row["error"] = str(e)
        return row

def _extract_embed_src(html: str) -> str | None:
    if not html:
        return None
    m = re.search(r'src="([^"]+)"', html)
    return m.group(1) if m else None

def _make_dataframe(rows: list[dict]) -> pd.DataFrame:
    cols = [
        "input_url", "status", "status_code", "error",
        "title", "author_name", "author_url",
        "provider_name", "provider_url",
        "type", "version",
        "thumbnail_url", "thumbnail_width", "thumbnail_height",
        "width", "height",
        "embed_src", "oembed_request",
    ]
    df = pd.DataFrame(rows)
    # Ensure consistent column order
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]

@yt_oembed_bp.route("/", methods=["GET", "POST"])
def yt_oembed():
    results = {}
    if request.method == "POST":
        mode = request.form.get("mode")  # "paste" or "file"
        output_mode = request.form.get("output_mode")  # "table" | "excel" | "csv"

        urls: list[str] = []
        if mode == "paste":
            text = request.form.get("urls", "")
            urls = extract_urls_from_text(text)
        elif mode == "file":
            up = request.files.get("url_file")
            if up:
                urls = extract_urls_from_file(up)
        urls = [u.strip() for u in urls if u.strip()]
        urls = list(OrderedDict.fromkeys(urls))[:MAX_URLS]

        rows = [call_oembed(u) for u in urls]
        df = _make_dataframe(rows)

        if output_mode == "excel":
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df.to_excel(writer, sheet_name="oembed", index=False)
            output.seek(0)
            return send_file(
                output,
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                as_attachment=True,
                download_name="youtube_oembed_results.xlsx",
            )

        if output_mode == "csv":
            output = io.StringIO()
            df.to_csv(output, index=False)
            mem = io.BytesIO(output.getvalue().encode("utf-8"))
            return send_file(
                mem,
                mimetype="text/csv",
                as_attachment=True,
                download_name="youtube_oembed_results.csv",
            )

        # Otherwise render as table
        results["table_rows"] = df.to_dict(orient="records")
        results["total"] = len(df)
        results["found"] = int((df["status"] == "Found").sum())
        results["not_found"] = int((df["status"] == "Not Found").sum())
        results["bad_request"] = int((df["status"] == "Bad Request").sum())
        results["errors"] = int((df["status"] == "Error").sum())

    return render_template("yt_oembed.html", results=results)