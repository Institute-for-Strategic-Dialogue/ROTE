# File: crypto_transfers_bp.py
"""
Flask Blueprint: Arkham Intelligence – Transfers to CSV or Gephi (GEXF)

Purpose
- Given a set of crypto wallet addresses, pull transfers from Arkham Intelligence's `/transfers` endpoint.
- Preserve all input columns from the source CSV/Excel into results (for easy sorting/joining).
- Export flattened transactions as CSV, or an interaction graph as a Gephi-friendly GEXF.

Input modes
- Paste CSV-like text (addresses, or a table with headers), or upload CSV/XLSX.
- You may choose the column that holds wallet addresses; defaults to `Address` (case-insensitive fallback to `wallet` or `address`).

Outputs
- CSV: one row per transfer *leg* (fan-out for multi-party from/to fields), with source columns preserved.
- Gephi (GEXF): directed edges from `from_address` → `to_address`, with per-edge attributes (token, amount, tx_hash, timestamp, chain).

Config
- API key is taken from the form field `api_key` or the `ARKHAM_API_KEY` environment variable.
- API base from `ARKHAM_API_BASE` or defaults to `https://api.arkhamintelligence.com`.

Notes
- Arkham's schema for transfers can vary by chain/token; this code defensively flattens present keys.
- Pagination: if the response includes a `next` cursor/link or returns `limit` items repeatedly, we paginate up to `max_pages`.
- Flows: `in`, `out`, or `all` (both directions).
- Rate limits/errors are recorded inline as `status`/`error` fields on rows.
"""
from __future__ import annotations

import csv
import io
import os
import re
import time
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import requests
import networkx as nx
from flask import Blueprint, render_template, request, send_file

crypto_transfers_bp = Blueprint("crypto_transfers", __name__, template_folder="templates")

# ----------------------------
# Defaults & helpers
# ----------------------------

DEFAULT_API_BASE = os.environ.get("ARKHAM_API_BASE", "https://api.arkhamintelligence.com")
TRANSFERS_PATH = "/transfers"
REQUEST_TIMEOUT = 20
DEFAULT_LIMIT = 1000
MAX_PAGES_HARD = 20
DEFAULT_MAX_PAGES = 3

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)

# Rows cap (safety)
MAX_INPUT_ROWS = 200_000


# ----------------------------
# Input parsing (paste/file)
# ----------------------------

def _sniff_csv_rows(raw: str) -> List[Dict[str, Any]]:
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
    rows: List[Dict[str, Any]] = []
    if reader.fieldnames:
        for i, row in enumerate(reader):
            if i >= MAX_INPUT_ROWS:
                break
            rows.append({k.strip(): (v or "").strip() for k, v in row.items()})
    return rows


def _read_upload_to_rows(up) -> List[Dict[str, Any]]:
    name = (getattr(up, "filename", "") or "").lower()
    raw = up.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    if name.endswith(".xlsx") or name.endswith(".xls"):
        # Use pandas to read Excel robustly
        up.stream.seek(0)
        df = pd.read_excel(up, sheet_name=0)
        return df.astype(str).fillna("").to_dict(orient="records")
    return _sniff_csv_rows(raw)


def _pick_address_column(fieldnames: Iterable[str], preferred: str | None) -> str | None:
    if preferred and preferred in fieldnames:
        return preferred
    lowers = {c.lower(): c for c in fieldnames}
    for cand in ("address", "wallet"):
        if cand in lowers:
            return lowers[cand]
    # Fallback to first column
    return next(iter(fieldnames), None)


# ----------------------------
# Arkham API client
# ----------------------------

def _arkham_headers(api_key: str) -> Dict[str, str]:
    return {
        "API-Key": api_key,
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }


def _fetch_transfers(api_base: str, api_key: str, base_wallet: str, flow: str, limit: int, max_pages: int) -> Tuple[List[Dict[str, Any]], str | None]:
    """Return (list_of_pages, error). Handles simple pagination heuristics.
    Arkham example params used here: base (wallet), flow (in|out|all), limit.
    """
    url = api_base.rstrip("/") + TRANSFERS_PATH
    params = {"base": base_wallet, "flow": flow, "limit": limit}
    headers = _arkham_headers(api_key)

    pages: List[Dict[str, Any]] = []
    error: str | None = None

    for p in range(max_pages):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            error = f"RequestException: {e}"
            break
        if r.status_code != 200:
            # Try to capture API error message
            try:
                msg = r.json()
                error = msg.get("message") or str(msg)[:500]
            except Exception:
                error = (r.text or "").strip()[:500]
            break
        try:
            data = r.json() or {}
        except Exception as e:
            error = f"JSON decode error: {e}"
            break

        pages.append(data)

        # Heuristic pagination: stop if fewer than limit transfers are returned or no obvious cursor
        transfers = (data.get("transfers") or [])
        if len(transfers) < limit:
            break
        # If the API exposes a cursor/next, support it
        next_cursor = data.get("next") or data.get("cursor") or data.get("nextCursor")
        if next_cursor:
            params["cursor"] = next_cursor
        else:
            # Without cursor, bail after first full page unless user asked for more
            # Continue loop up to max_pages as a best effort
            pass

        time.sleep(0.3)  # gentle pause

    return pages, error


# ----------------------------
# Flattening helpers
# ----------------------------

def _safe(d: Dict[str, Any], key: str, default: Any = None):
    v = d.get(key, default)
    return v if v is not None else default


def _flat_addr(prefix: str, addr: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(addr, dict):
        return out
    # Common fields
    out[f"{prefix}_address"] = _safe(addr, "address")
    # Optional nested entity/label
    ent = addr.get("arkhamEntity") or {}
    lab = addr.get("arkhamLabel") or {}
    if isinstance(ent, dict):
        for k, v in ent.items():
            out[f"{prefix}_entity_{k}"] = v
    if isinstance(lab, dict):
        for k, v in lab.items():
            out[f"{prefix}_label_{k}"] = v
    return out


def _flatten_transfer(base_wallet: str, src_cols: Dict[str, Any], t: Dict[str, Any], direction_hint: str) -> List[Dict[str, Any]]:
    """Return one row per counterparty leg for a single transfer object.
    - direction_hint: "in" means many from → one to; "out" means one from → many to; "all" try to infer.
    """
    rows: List[Dict[str, Any]] = []

    # Pull top-level transfer fields safely
    core = {
        "tx_hash": t.get("hash") or t.get("txHash") or t.get("transactionHash"),
        "token_symbol": (t.get("token") or {}).get("symbol") if isinstance(t.get("token"), dict) else t.get("tokenSymbol"),
        "token_address": (t.get("token") or {}).get("address") if isinstance(t.get("token"), dict) else t.get("tokenAddress"),
        "amount": t.get("value") or t.get("amount"),
        "amount_usd": t.get("valueUsd") or t.get("amountUsd"),
        "chain": t.get("chain") or t.get("chainName"),
        "timestamp": t.get("timestamp") or t.get("time") or t.get("blockTimestamp"),
        "direction_reported": t.get("direction") or direction_hint,
    }

    # Normalize parties
    to_addr = t.get("toAddress") or t.get("to") or {}
    to_list = t.get("toAddresses") or []
    from_addr = t.get("fromAddress") or t.get("from") or {}
    from_list = t.get("fromAddresses") or []

    # If lists are empty and singletons exist, adapt
    if to_list == [] and isinstance(to_addr, dict) and to_addr:
        to_list = [to_addr]
    if from_list == [] and isinstance(from_addr, dict) and from_addr:
        from_list = [from_addr]

    # Build rows
    def build_row(f_addr: Dict[str, Any], t_addr: Dict[str, Any]) -> Dict[str, Any]:
        r = {}
        r.update(core)
        r.update({"base_wallet": base_wallet})
        r.update(_flat_addr("from", f_addr))
        r.update(_flat_addr("to", t_addr))
        # Preserve source columns verbatim; avoid overwriting by only adding missing keys
        for k, v in src_cols.items():
            if k not in r:
                r[k] = v
            else:
                r[f"src_{k}"] = v
        r["status"] = "OK"
        r["error"] = None
        return r

    if direction_hint == "in":
        for f in from_list or [{}]:
            rows.append(build_row(f, to_list[0] if to_list else {}))
    elif direction_hint == "out":
        for taddr in to_list or [{}]:
            rows.append(build_row(from_list[0] if from_list else {}, taddr))
    else:  # all or unknown: generate cartesian pairs conservatively
        if from_list and to_list:
            for f in from_list:
                for taddr in to_list:
                    rows.append(build_row(f, taddr))
        elif from_list:
            for f in from_list:
                rows.append(build_row(f, to_addr or {}))
        elif to_list:
            for taddr in to_list:
                rows.append(build_row(from_addr or {}, taddr))
        else:
            rows.append(build_row(from_addr or {}, to_addr or {}))

    return rows


# ----------------------------
# Gephi (GEXF) builder
# ----------------------------

def _df_to_gexf(df: pd.DataFrame) -> bytes:
    """Build a directed graph from a flattened transfers DataFrame and return bytes of a .gexf file."""
    G = nx.DiGraph()

    # Helper: node attrs
    def add_node(addr: str, label: str | None = None, entity: str | None = None):
        if not addr:
            return
        if addr not in G:
            G.add_node(addr, label=label or addr, entity=entity)

    # Create nodes/edges
    for _, row in df.iterrows():
        f = row.get("from_address") or row.get("from_address") or row.get("from_address")
        # Our flattener stored exact keys: from_address, to_address, plus optional labels
        f = row.get("from_address")
        t = row.get("to_address")
        if not f and isinstance(row.get("from_address"), str):
            f = row.get("from_address")
        if not t and isinstance(row.get("to_address"), str):
            t = row.get("to_address")

        if not f or not t:
            # Try fallback field names from _flat_addr:
            f = f or row.get("from_address")
            t = t or row.get("to_address")
        if not f or not t:
            continue

        add_node(f, label=row.get("from_label_name") or row.get("from_entity_name"), entity=row.get("from_entity_name"))
        add_node(t, label=row.get("to_label_name") or row.get("to_entity_name"), entity=row.get("to_entity_name"))

        attrs = {
            "tx_hash": row.get("tx_hash"),
            "token_symbol": row.get("token_symbol"),
            "token_address": row.get("token_address"),
            "amount": row.get("amount"),
            "amount_usd": row.get("amount_usd"),
            "chain": row.get("chain"),
            "timestamp": row.get("timestamp"),
            "base_wallet": row.get("base_wallet"),
        }
        # Use a composite key to aggregate multiple edges between the same pair
        if G.has_edge(f, t):
            # Increment a simple count and last-seen attrs
            G[f][t]["count"] = G[f][t].get("count", 1) + 1
        else:
            G.add_edge(f, t, **attrs, count=1)

    mem = io.BytesIO()
    nx.write_gexf(G, mem)
    mem.seek(0)
    return mem.read()


# ----------------------------
# Flask route
# ----------------------------

@crypto_transfers_bp.route("/", methods=["GET", "POST"])
def crypto_transfers():
    results: dict = {}

    if request.method == "POST":
        mode = request.form.get("mode", "file")  # paste|file
        output_mode = request.form.get("output_mode", "csv")  # csv|gexf|table

        api_key = (request.form.get("api_key") or os.environ.get("ARKHAM_API_KEY") or "").strip()
        if not api_key:
            results["error"] = "Missing Arkham API key. Provide it in the form or set ARKHAM_API_KEY."
            return render_template("crypto_transfers.html", results=results)

        api_base = (request.form.get("api_base") or DEFAULT_API_BASE).strip()
        flow = (request.form.get("flow") or "all").lower()  # in|out|all
        try:
            limit = max(1, min(int(request.form.get("limit", DEFAULT_LIMIT)), DEFAULT_LIMIT))
        except Exception:
            limit = DEFAULT_LIMIT
        try:
            max_pages = max(1, min(int(request.form.get("max_pages", DEFAULT_MAX_PAGES)), MAX_PAGES_HARD))
        except Exception:
            max_pages = DEFAULT_MAX_PAGES

        # Ingest inputs
        rows: List[Dict[str, Any]] = []
        if mode == "paste":
            text = request.form.get("pasted", "")
            rows = _sniff_csv_rows(text)
        else:
            up = request.files.get("file")
            if up:
                rows = _read_upload_to_rows(up)

        if not rows:
            results["error"] = "Could not parse any rows. Upload a CSV/XLSX or paste CSV text with a header row."
            return render_template("crypto_transfers.html", results=results)

        addr_col = request.form.get("address_column") or _pick_address_column(rows[0].keys(), None)
        if not addr_col:
            results["error"] = "Could not determine the address column. Provide it explicitly."
            return render_template("crypto_transfers.html", results=results)

        # Query Arkham per wallet
        flat_rows: List[Dict[str, Any]] = []
        error_rows = 0
        ok_rows = 0

        for i, src in enumerate(rows):
            base_wallet = (src.get(addr_col) or "").strip()
            if not base_wallet:
                # Record a no-address row with preserved columns
                r = dict(src)
                r.update({"status": "Skipped", "error": "Missing address", "base_wallet": None})
                flat_rows.append(r)
                error_rows += 1
                continue

            pages, err = _fetch_transfers(api_base, api_key, base_wallet, flow, limit, max_pages)
            if err:
                rr = dict(src)
                rr.update({"status": "Error", "error": err, "base_wallet": base_wallet})
                flat_rows.append(rr)
                error_rows += 1
                continue

            any_rows = False
            for page in pages:
                transfers = page.get("transfers") or []
                if not transfers:
                    continue
                for t in transfers:
                    legs = _flatten_transfer(base_wallet, src, t, flow)
                    if legs:
                        flat_rows.extend(legs)
                        any_rows = True

            if not any_rows:
                rr = dict(src)
                rr.update({"status": "No Results", "error": None, "base_wallet": base_wallet})
                flat_rows.append(rr)
            else:
                ok_rows += 1

        # Build dataframe
        df = pd.DataFrame(flat_rows)
        # Normalize some common columns to stable names for Gephi
        if "from_address" not in df.columns:
            # Our _flat_addr uses keys `from_address` & `to_address`
            pass

        if output_mode == "gexf":
            try:
                gexf_bytes = _df_to_gexf(df)
            except Exception as e:
                results["error"] = f"Failed to build GEXF: {e}"
                return render_template("crypto_transfers.html", results=results)
            mem = io.BytesIO(gexf_bytes)
            return send_file(
                mem,
                mimetype="application/gexf+xml",
                as_attachment=True,
                download_name="crypto_transfers.gexf",
            )

        if output_mode == "csv":
            out = io.StringIO()
            df.to_csv(out, index=False)
            mem = io.BytesIO(out.getvalue().encode("utf-8"))
            return send_file(
                mem,
                mimetype="text/csv",
                as_attachment=True,
                download_name="crypto_transfers.csv",
            )

        # Otherwise render a small table preview
        results["table_rows"] = df.head(1000).to_dict(orient="records")
        results["total_rows"] = len(df)
        results["ok_rows"] = ok_rows
        results["error_rows"] = error_rows
        results["address_column"] = addr_col
        return render_template("crypto_transfers.html", results=results)

    # GET
    return render_template("crypto_transfers.html", results={})
