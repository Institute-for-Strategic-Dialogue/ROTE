# File: tiktok_ads.py
import time
import json
import math
import re
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from flask import Blueprint, render_template, request, send_file

tiktok_ads_bp = Blueprint("tiktok_ads", __name__, template_folder="templates")

# ---------- Helpers: parsing & dates ----------

def _parse_terms_text(text: str) -> List[str]:
    """
    Split on commas or newlines, trim, dedupe while preserving order.
    """
    if not text:
        return []
    parts = re.split(r"[,\n]+", text)
    seen, out = set(), []
    for p in (s.strip() for s in parts):
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out

def _pick_csv_column(df: pd.DataFrame, hint: Optional[str]) -> str:
    """
    Choose a CSV column for terms: explicit hint if valid; else first object-like column; else first column.
    """
    if hint and hint in df.columns:
        return hint
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    return obj_cols[0] if obj_cols else df.columns[0]

def _terms_from_csv(file_storage, column_hint: Optional[str]) -> List[str]:
    """
    Load a CSV, pick a column, coerce to str, dedupe.
    """
    df = pd.read_csv(file_storage)
    col = _pick_csv_column(df, (column_hint or "").strip() or None)
    vals = df[col].astype(str).fillna("").tolist()
    return _parse_terms_text("\n".join(vals))

def _compute_time_window(mode: str, start_date: str, end_date: str, months_back: int) -> Tuple[int, int]:
    """
    Return (start_ts, end_ts) as Unix seconds, in UTC.
    Modes:
      - 'relative': last N months (months_back)
      - 'custom'  : with YYYY-MM-DD inputs (inclusive of end date’s day)
    """
    now = datetime.now(timezone.utc)
    if mode == "custom" and start_date and end_date:
        # Interpret as UTC midnights; include the full end day
        start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end   = datetime.strptime(end_date,   "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1) - timedelta(seconds=1)
        return int(start.timestamp()), int(end.timestamp())
    # Relative by months
    start = now - relativedelta(months=months_back)
    return int(start.timestamp()), int(now.timestamp())

# ---------- Networking ----------

DEFAULT_HEADERS = {
    "authority":           "library.tiktok.com",
    "accept":              "application/json, text/plain, */*",
    "accept-language":     "en-US,en;q=0.9",
    "content-type":        "application/json",
    "origin":              "https://library.tiktok.com",
    "sec-ch-ua":           '"Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114"',
    "sec-ch-ua-mobile":    "?0",
    "sec-ch-ua-platform":  '"Linux"',
    "sec-fetch-dest":      "empty",
    "sec-fetch-mode":      "cors",
    "sec-fetch-site":      "same-origin",
    "user-agent":          (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36"
    ),
}

def _build_referer(region: str, start_ms: int, end_ms: int) -> str:
    return (
        "https://library.tiktok.com/ads"
        f"?region={region}"
        f"&start_time={start_ms}"
        f"&end_time={end_ms}"
        "&adv_name=&adv_biz_ids=&query_type=1"
        "&sort_type=last_shown_date,desc"
    )

def _backoff_sleep(attempt: int, base: float = 1.0, cap: float = 8.0):
    time.sleep(min(cap, base * (2 ** max(0, attempt - 1))))

def _fetch_page(
    session: requests.Session,
    *,
    term: str,
    region: str,
    start_ts: int,
    end_ts: int,
    query_type: str,
    adv_biz_ids: str,
    order: str,
    offset: int,
    limit: int,
    max_retries: int = 5,
    timeout: int = 20,
) -> dict:
    url = (
        "https://library.tiktok.com/api/v1/search"
        f"?region={region}"
        f"&type=1"
        f"&start_time={start_ts}"
        f"&end_time={end_ts}"
    )
    payload = {
        "query":       term,
        "query_type":  query_type,
        "adv_biz_ids": adv_biz_ids,
        "order":       order,
        "offset":      offset,
        "search_id":   "",
        "limit":       limit,
    }

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            r = session.post(url, json=payload, timeout=timeout)
            if r.status_code == 429 or 500 <= r.status_code < 600:
                _backoff_sleep(attempt)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            _backoff_sleep(attempt)
    raise RuntimeError(f"Failed after {max_retries} attempts: {last_exc}")

def _flatten_item(item: dict, term: str) -> dict:
    """
    Keep core fields, and JSON-encode lists for Excel-friendly storage.
    """
    core_keys = (
        "id", "name", "audit_status", "type",
        "first_shown_date", "last_shown_date",
        "estimated_audience", "spent", "impression",
        "show_mode", "rejection_info", "sor_audit_status"
    )
    out = {k: item.get(k) for k in core_keys}
    out.update({
        "videos":     json.dumps(item.get("videos", []), ensure_ascii=False),
        "image_urls": json.dumps(item.get("image_urls", []), ensure_ascii=False),
        "search_term": term,
    })
    return out

# ---------- Excel: make tz-naive if needed ----------

from pandas.api.types import is_datetime64tz_dtype

def _excel_safe(df: pd.DataFrame) -> pd.DataFrame:
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

# ---------- Route ----------

@tiktok_ads_bp.route("/", methods=["GET", "POST"])
def tiktok_ads():
    results = {}
    if request.method == "POST":
        mode = (request.form.get("mode") or "raw").strip()
        output_mode = request.form.get("output_mode") or "table"

        # Query params
        region      = (request.form.get("region") or "PL").strip()
        query_type  = (request.form.get("query_type") or "1").strip()
        adv_biz_ids = (request.form.get("adv_biz_ids") or "").strip()
        order       = (request.form.get("order") or "last_shown_date,desc").strip()

        # Date range
        range_mode   = request.form.get("range_mode") or "relative"
        months_back  = int(request.form.get("months_back") or 1)
        start_date   = request.form.get("start_date") or ""
        end_date     = request.form.get("end_date") or ""
        start_ts, end_ts = _compute_time_window(range_mode, start_date, end_date, months_back)

        # Pagination controls
        page_size          = max(1, min(50, int(request.form.get("page_size") or 20)))  # TikTok accepts small pages; 50 cap
        per_term_max_items = int(request.form.get("per_term_max_items") or 500)
        polite_sleep       = float(request.form.get("polite_sleep") or 1.0)

        # Collect terms
        terms: List[str] = []
        try:
            if mode == "csv":
                file = request.files.get("csv_file")
                if not file:
                    return render_template("tiktok_ads.html", results={"error": "Please upload a CSV file."})
                column_hint = request.form.get("csv_column") or ""
                terms = _terms_from_csv(file, column_hint)
            else:
                raw = request.form.get("terms_input") or ""
                terms = _parse_terms_text(raw)
        except Exception as e:
            return render_template("tiktok_ads.html", results={"error": f"Could not read terms: {e}"})

        if not terms:
            return render_template("tiktok_ads.html", results={"error": "No search terms provided."})

        # Session with headers (attach referer)
        start_ms, end_ms = start_ts * 1000, end_ts * 1000
        headers = DEFAULT_HEADERS.copy()
        headers["referer"] = _build_referer(region, start_ms, end_ms)
        session = requests.Session()
        session.headers.update(headers)

        # Crawl
        records = []
        summary_rows = []
        for term in terms:
            term_count = 0
            offset = 0
            try:
                while True:
                    resp = _fetch_page(
                        session,
                        term=term,
                        region=region,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        query_type=query_type,
                        adv_biz_ids=adv_biz_ids,
                        order=order,
                        offset=offset,
                        limit=page_size,
                    )
                    items = (resp.get("data", {}) or {}).get("items", []) or []
                    if not items:
                        break

                    for item in items:
                        records.append(_flatten_item(item, term))
                    term_count += len(items)

                    has_more = bool(resp.get("has_more", False))
                    if not has_more:
                        break
                    if per_term_max_items and term_count >= per_term_max_items:
                        break

                    offset += len(items)
                    time.sleep(polite_sleep)
            except Exception as e:
                # Keep going for other terms; note error in summary
                summary_rows.append({
                    "search_term": term,
                    "items_collected": term_count,
                    "error": str(e)
                })
                continue

            summary_rows.append({
                "search_term": term,
                "items_collected": term_count,
                "error": ""
            })

        # Build DataFrames
        df_ads = pd.DataFrame(records)
        df_summary = pd.DataFrame(summary_rows).sort_values(["items_collected", "search_term"], ascending=[False, True])

        # Output
        if output_mode == "excel":
            out = BytesIO()
            with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
                meta = pd.DataFrame([{
                    "region": region,
                    "query_type": query_type,
                    "adv_biz_ids": adv_biz_ids,
                    "order": order,
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "months_back": months_back if range_mode == "relative" else "",
                    "range_mode": range_mode,
                    "page_size": page_size,
                    "per_term_max_items": per_term_max_items,
                    "polite_sleep_s": polite_sleep,
                    "unique_terms": len(terms),
                    "total_records": int(len(df_ads)),
                }])

                _excel_safe(meta).to_excel(writer, sheet_name="META", index=False)
                _excel_safe(df_summary).to_excel(writer, sheet_name="SUMMARY", index=False)
                _excel_safe(df_ads).to_excel(writer, sheet_name="ADS", index=False)
            out.seek(0)
            return send_file(
                out,
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                as_attachment=True,
                download_name="tiktok_search_results.xlsx",
            )

        # Otherwise, render on page (preview)
        preview_rows = df_ads.head(100).to_dict(orient="records") if not df_ads.empty else []
        results = {
            "ok": True,
            "params": {
                "region": region,
                "range_mode": range_mode,
                "months_back": months_back,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "page_size": page_size,
                "per_term_max_items": per_term_max_items,
                "polite_sleep": polite_sleep,
            },
            "summary": df_summary.to_dict(orient="records"),
            "preview": preview_rows,
            "total_records": int(len(df_ads)),
        }

    return render_template("tiktok_ads.html", results=results)
