# File: site_downloader.py
"""
Blueprint: Download site pages listed in RSS feeds or XML sitemaps.

POST /site_downloader/download
- feed_url: URL to RSS or sitemap (required)
- source_type: rss | sitemap (default: rss)
- max_urls: optional int cap (default 50, hard cap 200)

Returns a ZIP containing HTML (or error text) for each discovered URL plus a
CSV manifest. Designed to be fast and self-contained for offline analysis.
"""
import io
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, List, Tuple

import requests
from flask import Blueprint, jsonify, render_template, request, send_file
from lxml import etree

site_downloader_bp = Blueprint("site_downloader", __name__, template_folder="templates")

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
    )
}

MAX_URLS = 200
DEFAULT_MAX_URLS = 50
REQUEST_TIMEOUT = 12
MAX_WORKERS = 8


# ----------------------------
# Helpers
# ----------------------------
def _uniq_preserve(seq: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in seq:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _parse_feed(feed_bytes: bytes, source_type: str) -> List[str]:
    parser = etree.XMLParser(recover=True, resolve_entities=False)
    root = etree.fromstring(feed_bytes, parser=parser)
    urls: List[str] = []

    if source_type == "sitemap":
        urls = [loc.strip() for loc in root.xpath("//url/loc/text()") if loc and loc.strip()]
    else:
        urls.extend([link.text.strip() for link in root.xpath("//item/link") if link is not None and link.text])
        urls.extend([href.strip() for href in root.xpath("//entry/link/@href") if href and href.strip()])
    return _uniq_preserve(urls)


def _fetch_url(url: str) -> Tuple[str, int | None, str]:
    try:
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return url, resp.status_code, resp.text
    except Exception as exc:
        return url, None, f"ERROR: {exc}"


def _slugify(url: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "-", url).strip("-")
    return (safe[:80] if safe else "page")


def _download_urls(urls: List[str]) -> List[Tuple[str, int | None, str]]:
    results: List[Tuple[str, int | None, str]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {pool.submit(_fetch_url, url): url for url in urls}
        for future in as_completed(future_map):
            results.append(future.result())
    return results


def _zip_results(results: List[Tuple[str, int | None, str]]) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        manifest_lines = ["url,status,filename,bytes"]
        for idx, (url, status, body) in enumerate(results):
            is_error = status is None
            ext = "txt" if is_error else "html"
            fname = f"{idx:03d}_{_slugify(url)}.{ext}"
            payload = body or ""
            zf.writestr(fname, payload)
            manifest_lines.append(f'"{url}",{status if status is not None else ""},{fname},{len(payload.encode("utf-8"))}')
        zf.writestr("manifest.csv", "\n".join(manifest_lines))
    buf.seek(0)
    return buf


# ----------------------------
# Routes
# ----------------------------
@site_downloader_bp.route("/", methods=["GET"])
def form():
    return render_template("site_downloader.html")


@site_downloader_bp.route("/download", methods=["POST"])
def download():
    data = request.get_json(silent=True) or {}
    feed_url = request.form.get("feed_url") or data.get("feed_url")
    source_type = (request.form.get("source_type") or data.get("source_type") or "rss").lower().strip()
    max_urls = request.form.get("max_urls") or data.get("max_urls")

    if not feed_url:
        return jsonify({"error": "feed_url is required"}), 400
    if source_type not in {"rss", "sitemap"}:
        return jsonify({"error": "source_type must be 'rss' or 'sitemap'"}), 400

    try:
        limit = int(max_urls) if max_urls is not None else DEFAULT_MAX_URLS
    except ValueError:
        return jsonify({"error": "max_urls must be an integer"}), 400
    limit = max(1, min(limit, MAX_URLS))

    try:
        feed_resp = requests.get(feed_url, headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT)
        feed_resp.raise_for_status()
    except Exception as exc:
        return jsonify({"error": f"Failed to fetch feed: {exc}"}), 502

    urls = _parse_feed(feed_resp.content, source_type)
    if not urls:
        return jsonify({"error": "No URLs found in feed"}), 400
    urls = urls[:limit]

    results = _download_urls(urls)
    zip_buf = _zip_results(results)
    return send_file(
        zip_buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name="site_downloads.zip",
    )
