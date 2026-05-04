"""
Flask blueprint: Social Post & Account Status checker.

Two distinct modes (the user picks one per submission):

  - Posts: oEmbed-style probe per platform. Cleanly distinguishes live vs gone,
    but cannot say *why* a post is gone -- delete / suspend / private collapse
    into a single "gone" verdict.

  - Accounts: platform-specific multi-probe logic. For X this is syndication
    plus a best-effort GraphQL probe; can split live / protected /
    suspended-or-deactivated / never-existed.

Inputs: paste textarea (one per line) OR CSV upload (with optional platform
column). All original CSV columns are preserved on output.
"""
from __future__ import annotations

import csv
import io
import time
from collections import OrderedDict
from dataclasses import asdict
from typing import Optional

import pandas as pd
from flask import Blueprint, render_template, request, send_file

from .platforms import (
    PLATFORMS,
    detect_platform_from_url,
    get_platform,
    list_platforms,
)
from .platforms.base import (
    NotSupported,
    StatusResult,
    STATUS_ERROR,
    STATUS_INVALID,
)

post_status_bp = Blueprint("post_status", __name__, template_folder="templates")

MAX_INPUTS = 5_000
DEFAULT_INTER_REQUEST_PAUSE_SEC = 0.35  # gentle on Cloudflare
MAX_PAUSE_SEC = 5.0


def _safe(x) -> str:
    return ("" if x is None else str(x)).strip()


def _parse_paste(text: str) -> list[str]:
    if not text:
        return []
    seen: "OrderedDict[str, None]" = OrderedDict()
    for line in text.splitlines():
        s = line.strip()
        if s and s not in seen:
            seen[s] = None
    return list(seen.keys())[:MAX_INPUTS]


def _parse_csv(file_storage, identifier_col: str, platform_col: str) -> list[dict]:
    raw = file_storage.read()
    if isinstance(raw, bytes):
        # utf-8-sig strips a leading BOM so column names match user-typed strings.
        raw = raw.decode("utf-8-sig", errors="ignore")

    buf = io.StringIO(raw)
    sample = buf.read(4096)
    buf.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample) if sample.strip() else csv.excel
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(buf, dialect=dialect)
    fields = reader.fieldnames or []
    if not fields:
        return []
    lowered = {f.lower(): f for f in fields}

    id_col: Optional[str] = None
    if identifier_col:
        if identifier_col in fields:
            id_col = identifier_col
        elif identifier_col.lower() in lowered:
            id_col = lowered[identifier_col.lower()]
    if not id_col:
        for cand in (
            "url", "link", "post_url", "post", "tweet_url", "tweet",
            "account_url", "account", "username", "handle", "screen_name",
            "identifier",
        ):
            if cand in lowered:
                id_col = lowered[cand]
                break
    if not id_col:
        id_col = fields[0]

    plat_col: Optional[str] = None
    if platform_col:
        if platform_col in fields:
            plat_col = platform_col
        elif platform_col.lower() in lowered:
            plat_col = lowered[platform_col.lower()]
    if not plat_col and "platform" in lowered:
        plat_col = lowered["platform"]

    items: list[dict] = []
    for row in reader:
        ident = _safe(row.get(id_col))
        if not ident:
            continue
        items.append(
            {
                "input": ident,
                "platform_hint": _safe(row.get(plat_col)) if plat_col else "",
                "extra": dict(row),
            }
        )
        if len(items) >= MAX_INPUTS:
            break
    return items


def _resolve_platform(input_str: str, hint: str, default: str) -> Optional[str]:
    if hint and hint.strip().lower() in PLATFORMS:
        return hint.strip().lower()
    detected = detect_platform_from_url(input_str)
    if detected:
        return detected
    return default if default in PLATFORMS else None


def _check_one(item: dict, mode: str, default_platform: str) -> StatusResult:
    raw = item["input"]
    hint = item.get("platform_hint", "")
    platform_name = _resolve_platform(raw, hint, default_platform)
    if not platform_name:
        return StatusResult(
            input=raw, mode=mode, platform="",
            status=STATUS_INVALID, status_label="Unknown platform",
            error=f"Could not determine platform (hint='{hint}', no URL match).",
        )
    checker = get_platform(platform_name)
    if checker is None:
        return StatusResult(
            input=raw, mode=mode, platform=platform_name,
            status=STATUS_INVALID, status_label="Unsupported platform",
            error=f"No checker registered for '{platform_name}'.",
        )
    try:
        if mode == "post":
            return checker.check_post(raw)
        return checker.check_account(raw)
    except NotSupported as e:
        return StatusResult(
            input=raw, mode=mode, platform=platform_name,
            status=STATUS_INVALID, status_label="Mode not supported on this platform",
            error=str(e),
        )
    except Exception as e:  # pragma: no cover - defensive
        return StatusResult(
            input=raw, mode=mode, platform=platform_name,
            status=STATUS_ERROR, status_label="Unexpected error",
            error=f"{type(e).__name__}: {e}",
        )


def _result_to_row(result: StatusResult, extra: Optional[dict]) -> dict:
    row = asdict(result)
    if extra:
        merged = dict(extra)
        for k, v in row.items():
            merged[k] = v
        return merged
    return row


@post_status_bp.route("/", methods=["GET", "POST"])
def post_status_view():
    ctx = {
        "platforms": list_platforms(),
        "mode": "post",
        "default_platform": "x",
    }
    if request.method != "POST":
        return render_template("post_status.html", **ctx)

    mode = (_safe(request.form.get("mode")) or "post").lower()
    if mode not in ("post", "account"):
        mode = "post"
    ctx["mode"] = mode

    default_platform = (_safe(request.form.get("default_platform")) or "x").lower()
    if default_platform not in PLATFORMS:
        default_platform = "x"
    ctx["default_platform"] = default_platform

    paste = _safe(request.form.get("inputs"))
    csv_file = request.files.get("csv_file")
    identifier_col = _safe(request.form.get("identifier_col"))
    platform_col = _safe(request.form.get("platform_col"))
    output_mode = _safe(request.form.get("output_mode")) or "table"

    try:
        pause = float(request.form.get("inter_request_pause", DEFAULT_INTER_REQUEST_PAUSE_SEC))
    except (TypeError, ValueError):
        pause = DEFAULT_INTER_REQUEST_PAUSE_SEC
    pause = max(0.0, min(pause, MAX_PAUSE_SEC))

    items: list[dict] = []
    if csv_file and getattr(csv_file, "filename", ""):
        items = _parse_csv(csv_file, identifier_col, platform_col)
    elif paste:
        items = [{"input": s, "platform_hint": "", "extra": {}} for s in _parse_paste(paste)]
    else:
        ctx["error"] = "Paste at least one input or upload a CSV."
        return render_template("post_status.html", **ctx)

    if not items:
        ctx["error"] = "No usable inputs found."
        return render_template("post_status.html", **ctx)

    if len(items) > MAX_INPUTS:
        items = items[:MAX_INPUTS]

    counts = {
        "live": 0, "gone": 0, "protected": 0,
        "suspended_or_deactivated": 0, "suspended": 0, "deactivated": 0,
        "not_found": 0, "unknown": 0, "rate_limited": 0,
        "error": 0, "invalid_url": 0, "other": 0,
    }
    rows: list[dict] = []
    for i, it in enumerate(items):
        res = _check_one(it, mode=mode, default_platform=default_platform)
        rows.append(_result_to_row(res, it.get("extra")))
        counts[res.status if res.status in counts else "other"] += 1
        if i + 1 < len(items) and pause > 0:
            time.sleep(pause)

    df = pd.DataFrame(rows)

    if output_mode == "csv":
        out = io.StringIO()
        df.to_csv(out, index=False)
        mem = io.BytesIO(out.getvalue().encode("utf-8"))
        return send_file(
            mem,
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"post_status_{mode}.csv",
        )

    if output_mode == "excel":
        mem = io.BytesIO()
        with pd.ExcelWriter(mem, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name=f"{mode}_status", index=False)
        mem.seek(0)
        return send_file(
            mem,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name=f"post_status_{mode}.xlsx",
        )

    ctx["results"] = rows
    ctx["total"] = len(rows)
    ctx["counts"] = counts
    return render_template("post_status.html", **ctx)
