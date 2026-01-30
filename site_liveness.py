# File: site_liveness_blueprint.py
"""
Flask Blueprint: Site Liveness Checker (CSV-only input)

Goal
- Determine whether a site "is live" using a layered approach:
  - Normalize input into candidate URLs (HTTPS-first by default).
  - DNS resolve the hostname (captures NXDOMAIN, timeouts, etc.).
  - Attempt TCP connect to common ports (443 and 80).
  - Perform HTTP request (HEAD default, with smart fallback to GET).
  - Optionally follow redirects and record redirect chain.
- "Live" is defined as: we can reach the site over HTTP(S) and receive a response.
  - By default, any HTTP status code counts as "live" if a response is received.
  - You can optionally require 2xx/3xx via `require_http_ok`.

Input
- Upload a CSV and specify which column(s) contain sites/URLs (defaults to `site`).
- Accepts:
  - Full URLs (https://example.com/path)
  - Hostnames/domains (example.com)
  - Bare host+path (example.com/foo)

Output Columns (appended per specified input column <col>)
- normalized_input_<col>        (string)
- checked_url_<col>             (the URL we actually attempted first)
- final_url_<col>               (after redirects, if followed)
- http_status_<col>             (int)
- redirect_chain_<col>          ("start -> ... -> final")
- response_time_ms_<col>        (int)
- resolved_ips_<col>            ("ip1, ip2, ...")
- dns_ok_<col>                  (bool)
- tcp_443_ok_<col>              (bool)
- tcp_80_ok_<col>               (bool)
- is_live_<col>                 (bool)
- error_<col>                   (string, if any)

Row Aggregates
- processed_site_columns
- row_status                    ("OK" | "Error" | "No Site")
- live_columns                  (comma-separated cols that were live)

Template
- Expects `templates/site_liveness.html` that reads a `results` dict.

POST Form Fields
- file: CSV upload (required)
- site_columns: comma-separated columns (default: `site`)
- path: request path to check (default: `/`)
- method: `head` (default) or `get`
- follow_redirects: `1` (default) or `0`
- timeout: seconds, default 10
- tcp_timeout: seconds, default 3
- max_redirects: int, default 10 (hard cap 20)
- prefer_https: `1` (default) or `0`
- require_http_ok: `0` (default) or `1`  (if 1, require status 200–399 to mark live)
- output_mode: `table` | `csv` | `excel`
"""
import io
import socket
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse, urlunparse

import pandas as pd
import requests
from flask import Blueprint, render_template, request, send_file

site_liveness_bp = Blueprint("site_liveness", __name__, template_folder="templates")

REQUEST_TIMEOUT_DEFAULT = 10
TCP_TIMEOUT_DEFAULT = 3
MAX_REDIRECTS_DEFAULT = 10
MAX_REDIRECTS_HARD = 20
MAX_ROWS = 200_000  # guardrail

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)

# ----------------------------
# Helpers
# ----------------------------

def _safe_str(x) -> str:
    return ("" if x is None else str(x)).strip()

def _looks_like_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")

def _split_host_path(s: str) -> tuple[str, str]:
    """
    For inputs like:
      - example.com
      - example.com/foo
      - example.com:8080/bar   (we preserve port if present)
    Return (netloc-ish, path-ish).
    """
    s = _safe_str(s)
    if not s:
        return "", ""
    # If it's already a URL, parse it properly elsewhere.
    if _looks_like_url(s):
        return s, ""
    # If it contains a slash, treat everything before first slash as netloc.
    if "/" in s:
        host, path = s.split("/", 1)
        return host.strip(), "/" + path.strip()
    return s, ""

def _normalize_to_candidate_urls(raw: str, prefer_https: bool, path: str) -> list[str]:
    """
    Convert raw input into a list of URLs to try, in order.
    - If raw is a full URL, keep its scheme and path unless missing path.
    - If raw is a hostname (or hostname/path), create https:// and http:// candidates.
    """
    raw = _safe_str(raw)
    path = path if path.startswith("/") else ("/" + path if path else "/")
    if not raw:
        return []

    if _looks_like_url(raw):
        p = urlparse(raw)
        # Ensure we have a path
        final_path = p.path if p.path else path
        rebuilt = urlunparse((p.scheme, p.netloc, final_path, p.params, p.query, p.fragment))
        return [rebuilt]

    host, maybe_path = _split_host_path(raw)
    if not host:
        return []

    use_path = maybe_path if maybe_path else path
    https_url = f"https://{host}{use_path}"
    http_url = f"http://{host}{use_path}"
    return [https_url, http_url] if prefer_https else [http_url, https_url]

def _hostname_from_url(u: str) -> str:
    try:
        return (urlparse(u).hostname or "").strip().lower()
    except Exception:
        return ""

def _resolve_dns(hostname: str) -> tuple[bool, list[str], Optional[str]]:
    """
    Return (dns_ok, ips, error).
    Uses getaddrinfo so it works across IPv4/IPv6 environments.
    """
    hostname = _safe_str(hostname)
    if not hostname:
        return False, [], "Missing hostname"
    try:
        infos = socket.getaddrinfo(hostname, None)
        ips = []
        for fam, _, _, _, sockaddr in infos:
            if fam == socket.AF_INET:
                ips.append(sockaddr[0])
            elif fam == socket.AF_INET6:
                ips.append(sockaddr[0])
        # De-dupe, stable order
        seen = set()
        uniq = []
        for ip in ips:
            if ip not in seen:
                seen.add(ip)
                uniq.append(ip)
        return (len(uniq) > 0), uniq, None
    except Exception as e:
        return False, [], f"DNS error: {e}"

def _tcp_probe(hostname: str, port: int, timeout_s: int) -> tuple[bool, Optional[str]]:
    """
    Attempt a TCP connect. This is not ICMP ping, so it works in more environments.
    """
    hostname = _safe_str(hostname)
    if not hostname:
        return False, "Missing hostname"
    try:
        with socket.create_connection((hostname, port), timeout=timeout_s):
            return True, None
    except Exception as e:
        return False, f"TCP {port} error: {e}"

@dataclass
class CheckResult:
    normalized_input: str
    checked_url: Optional[str]
    final_url: Optional[str]
    http_status: Optional[int]
    redirect_chain: list[str]
    response_time_ms: Optional[int]
    resolved_ips: list[str]
    dns_ok: bool
    tcp_443_ok: bool
    tcp_80_ok: bool
    is_live: bool
    error: Optional[str]

def _http_check(
    url: str,
    method: str,
    timeout_s: int,
    follow_redirects: bool,
    max_redirects: int,
) -> tuple[Optional[str], Optional[int], list[str], Optional[int], Optional[str]]:
    """
    Return (final_url, status_code, chain, response_time_ms, error)
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    sess = requests.Session()
    sess.max_redirects = max_redirects

    def do_head():
        try:
            t0 = time.perf_counter()
            r = sess.head(url, headers=headers, timeout=timeout_s, allow_redirects=follow_redirects)
            dt = int(round((time.perf_counter() - t0) * 1000))
            hist = r.history or []
            chain = [h.url for h in hist] + [r.url]
            return r.url, r.status_code, chain, dt, None
        except requests.RequestException as e:
            return None, None, [], None, str(e)

    def do_get():
        try:
            t0 = time.perf_counter()
            r = sess.get(url, headers=headers, timeout=timeout_s, allow_redirects=follow_redirects, stream=True)
            dt = int(round((time.perf_counter() - t0) * 1000))
            hist = r.history or []
            chain = [h.url for h in hist] + [r.url]
            return r.url, r.status_code, chain, dt, None
        except requests.RequestException as e:
            return None, None, [], None, str(e)

    method = (method or "head").lower()
    if method not in ("head", "get"):
        method = "head"

    # HEAD-first with smart fallback
    final_url, code, chain, ms, err = (None, None, [], None, None)
    if method == "head":
        final_url, code, chain, ms, err = do_head()
        if err or code in (400, 401, 403, 405):
            final_url, code, chain, ms, err = do_get()
    else:
        final_url, code, chain, ms, err = do_get()

    # Hard-cap redirect chain, if present
    if chain and len(chain) > max_redirects + 1:
        chain = chain[: max_redirects + 1]
        err = f"Exceeded max_redirects={max_redirects}"

    return final_url, code, chain, ms, err

def _is_live_from_http(code: Optional[int], got_response: bool, require_http_ok: bool) -> bool:
    if not got_response:
        return False
    if code is None:
        return True
    if require_http_ok:
        return 200 <= code <= 399
    return True

def _check_one(raw_value: str, path: str, prefer_https: bool, method: str, follow_redirects: bool,
               timeout_s: int, tcp_timeout_s: int, max_redirects: int, require_http_ok: bool) -> CheckResult:
    normalized = _safe_str(raw_value)
    candidates = _normalize_to_candidate_urls(normalized, prefer_https=prefer_https, path=path)

    if not candidates:
        return CheckResult(
            normalized_input=normalized,
            checked_url=None,
            final_url=None,
            http_status=None,
            redirect_chain=[],
            response_time_ms=None,
            resolved_ips=[],
            dns_ok=False,
            tcp_443_ok=False,
            tcp_80_ok=False,
            is_live=False,
            error="Missing site/URL",
        )

    # DNS + TCP probes based on first candidate's hostname
    hostname = _hostname_from_url(candidates[0])
    dns_ok, ips, dns_err = _resolve_dns(hostname)
    tcp_443_ok, tcp_443_err = _tcp_probe(hostname, 443, tcp_timeout_s) if hostname else (False, "Missing hostname")
    tcp_80_ok, tcp_80_err = _tcp_probe(hostname, 80, tcp_timeout_s) if hostname else (False, "Missing hostname")

    # HTTP attempts: try candidates in order (scheme fallback)
    last_err = None
    for attempt_url in candidates:
        final_url, code, chain, ms, err = _http_check(
            attempt_url,
            method=method,
            timeout_s=timeout_s,
            follow_redirects=follow_redirects,
            max_redirects=max_redirects,
        )
        got_response = (err is None) and (code is not None or final_url is not None)
        is_live = _is_live_from_http(code, got_response=got_response, require_http_ok=require_http_ok)

        if is_live:
            return CheckResult(
                normalized_input=normalized,
                checked_url=attempt_url,
                final_url=final_url,
                http_status=code,
                redirect_chain=chain,
                response_time_ms=ms,
                resolved_ips=ips,
                dns_ok=dns_ok,
                tcp_443_ok=tcp_443_ok,
                tcp_80_ok=tcp_80_ok,
                is_live=True,
                error=None,
            )

        last_err = err or last_err

    # If HTTP failed, return diagnostics combining DNS/TCP notes
    diag_parts = []
    if dns_err:
        diag_parts.append(dns_err)
    if tcp_443_err:
        diag_parts.append(tcp_443_err)
    if tcp_80_err:
        diag_parts.append(tcp_80_err)
    if last_err:
        diag_parts.append(f"HTTP error: {last_err}")
    diag = " | ".join([p for p in diag_parts if p]) or "Unreachable"

    return CheckResult(
        normalized_input=normalized,
        checked_url=candidates[0],
        final_url=None,
        http_status=None,
        redirect_chain=[],
        response_time_ms=None,
        resolved_ips=ips,
        dns_ok=dns_ok,
        tcp_443_ok=tcp_443_ok,
        tcp_80_ok=tcp_80_ok,
        is_live=False,
        error=diag,
    )

# ----------------------------
# Flask endpoint
# ----------------------------

@site_liveness_bp.route("/", methods=["GET", "POST"])
def site_liveness():
    results: dict = {}

    if request.method == "POST":
        up = request.files.get("file")
        if not up:
            results["error"] = "Please upload a CSV file."
            return render_template("site_liveness.html", results=results)

        raw = up.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")

        df_in = pd.read_csv(io.StringIO(raw))
        if len(df_in) > MAX_ROWS:
            results["error"] = f"Too many rows: {len(df_in)}. Max allowed is {MAX_ROWS}."
            return render_template("site_liveness.html", results=results)

        rows = df_in.to_dict(orient="records")
        if not rows:
            results["error"] = "CSV contained no rows."
            return render_template("site_liveness.html", results=results)

        site_columns = [
            c.strip() for c in (_safe_str(request.form.get("site_columns")) or "site").split(",") if c.strip()
        ]
        path = _safe_str(request.form.get("path")) or "/"
        method = (_safe_str(request.form.get("method")) or "head").lower()
        follow_redirects = (_safe_str(request.form.get("follow_redirects")) or "1") != "0"
        prefer_https = (_safe_str(request.form.get("prefer_https")) or "1") != "0"
        require_http_ok = (_safe_str(request.form.get("require_http_ok")) or "0") == "1"

        try:
            timeout_s = int(request.form.get("timeout", REQUEST_TIMEOUT_DEFAULT))
        except Exception:
            timeout_s = REQUEST_TIMEOUT_DEFAULT

        try:
            tcp_timeout_s = int(request.form.get("tcp_timeout", TCP_TIMEOUT_DEFAULT))
        except Exception:
            tcp_timeout_s = TCP_TIMEOUT_DEFAULT

        try:
            max_redirects = int(request.form.get("max_redirects", MAX_REDIRECTS_DEFAULT))
        except Exception:
            max_redirects = MAX_REDIRECTS_DEFAULT
        max_redirects = max(1, min(max_redirects, MAX_REDIRECTS_HARD))

        out_rows: list[dict] = []
        appended_cols: set[str] = set()
        ok_rows = 0
        error_rows = 0
        live_rows = 0

        for src in rows:
            base = dict(src)
            processed_cols = []
            live_cols = []
            any_live = False
            any_err = False

            for col in site_columns:
                val = _safe_str(src.get(col))
                cr = _check_one(
                    val,
                    path=path,
                    prefer_https=prefer_https,
                    method=method,
                    follow_redirects=follow_redirects,
                    timeout_s=timeout_s,
                    tcp_timeout_s=tcp_timeout_s,
                    max_redirects=max_redirects,
                    require_http_ok=require_http_ok,
                )

                base[f"normalized_input_{col}"] = cr.normalized_input
                base[f"checked_url_{col}"] = cr.checked_url
                base[f"final_url_{col}"] = cr.final_url
                base[f"http_status_{col}"] = cr.http_status
                base[f"redirect_chain_{col}"] = " -> ".join(cr.redirect_chain) if cr.redirect_chain else None
                base[f"response_time_ms_{col}"] = cr.response_time_ms
                base[f"resolved_ips_{col}"] = ", ".join(cr.resolved_ips) if cr.resolved_ips else None
                base[f"dns_ok_{col}"] = cr.dns_ok
                base[f"tcp_443_ok_{col}"] = cr.tcp_443_ok
                base[f"tcp_80_ok_{col}"] = cr.tcp_80_ok
                base[f"is_live_{col}"] = cr.is_live
                base[f"error_{col}"] = cr.error

                appended_cols.update({
                    f"normalized_input_{col}",
                    f"checked_url_{col}",
                    f"final_url_{col}",
                    f"http_status_{col}",
                    f"redirect_chain_{col}",
                    f"response_time_ms_{col}",
                    f"resolved_ips_{col}",
                    f"dns_ok_{col}",
                    f"tcp_443_ok_{col}",
                    f"tcp_80_ok_{col}",
                    f"is_live_{col}",
                    f"error_{col}",
                })

                processed_cols.append(col)
                if cr.is_live:
                    any_live = True
                    live_cols.append(col)
                if cr.error and not cr.is_live:
                    any_err = True

            base["processed_site_columns"] = ", ".join(processed_cols)
            base["live_columns"] = ", ".join(live_cols)

            base["row_status"] = (
                "OK" if any_live else ("Error" if any_err else "No Site")
            )

            if base["row_status"] == "OK":
                ok_rows += 1
            elif base["row_status"] == "Error":
                error_rows += 1
            if any_live:
                live_rows += 1

            out_rows.append(base)

        df = pd.DataFrame(out_rows)
        original_cols = list(rows[0].keys())
        ordered_cols = original_cols.copy()

        meta_cols = ["processed_site_columns", "live_columns", "row_status"]
        for c in meta_cols:
            if c not in ordered_cols:
                ordered_cols.append(c)

        for c in sorted(appended_cols):
            if c not in ordered_cols:
                ordered_cols.append(c)

        for c in df.columns:
            if c not in ordered_cols:
                ordered_cols.append(c)

        df = df[ordered_cols]

        output_mode = _safe_str(request.form.get("output_mode")) or "table"
        if output_mode == "excel":
            mem = io.BytesIO()
            with pd.ExcelWriter(mem, engine="xlsxwriter") as writer:
                df.to_excel(writer, sheet_name="site_liveness", index=False)
            mem.seek(0)
            return send_file(
                mem,
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                as_attachment=True,
                download_name="site_liveness_results.xlsx",
            )

        if output_mode == "csv":
            out = io.StringIO()
            df.to_csv(out, index=False)
            mem = io.BytesIO(out.getvalue().encode("utf-8"))
            return send_file(
                mem,
                mimetype="text/csv",
                as_attachment=True,
                download_name="site_liveness_results.csv",
            )

        results["table_rows"] = df.to_dict(orient="records")
        results["total_rows"] = len(df)
        results["ok_rows"] = ok_rows
        results["error_rows"] = error_rows
        results["live_rows"] = live_rows
        return render_template("site_liveness.html", results=results)

    return render_template("site_liveness.html", results={})