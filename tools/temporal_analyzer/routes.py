# File: temporal_analyzer.py
import io
import json
import math
import re
from io import BytesIO
from typing import Optional, Tuple
from itertools import combinations
from collections import Counter as PyCounter

import numpy as np
import pandas as pd
# top of file (imports)
from pandas.api.types import is_datetime64tz_dtype

from flask import Blueprint, render_template, request, send_file

temporal_bp = Blueprint("temporal_analyzer", __name__, template_folder="templates")

# ----------------------------
# Parsing helpers
# ----------------------------
def _maybe_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    return pd.to_numeric(s, errors="coerce")

def _parse_unix(series: pd.Series) -> Optional[pd.Series]:
    s = _maybe_numeric_series(series)
    if s.isna().all():
        return None
    sample = s.dropna()
    if sample.empty:
        return None

    med = float(sample.median())
    candidates = []
    if 1e8 <= med < 1e11: candidates.append(("s","s"))
    if 1e11 <= med < 1e14: candidates.append(("ms","ms"))
    if 1e14 <= med < 1e17: candidates.append(("us","us"))
    if 1e17 <= med < 1e20: candidates.append(("ns","ns"))
    if not candidates:
        med_len = int(np.median([len(str(int(abs(x)))) for x in sample[:100].tolist() if not pd.isna(x)]))
        if med_len <= 10: candidates.append(("s","s"))
        elif med_len <= 13: candidates.append(("ms","ms"))
        elif med_len <= 16: candidates.append(("us","us"))
        else: candidates.append(("ns","ns"))

    for _, unit in candidates:
        try:
            dt = pd.to_datetime(s, unit=unit, utc=True, errors="coerce")
            if dt.notna().mean() > 0.8:
                return dt
        except Exception:
            pass
    return None

def _parse_datetimes(raw: pd.Series, dayfirst: bool) -> Optional[pd.Series]:
    """
    Try several parsing strategies and return whichever yields the most parsed rows.
    Only rejects columns where *nothing* parses — sparse columns (e.g. a reply-date
    field populated on only 30% of rows) are still accepted.
    """
    non_null = raw.notna().sum()
    if non_null == 0:
        return None

    best: Optional[pd.Series] = None
    best_count = 0

    strategies = [
        dict(utc=True, errors="coerce"),
        dict(utc=True, errors="coerce", dayfirst=dayfirst),
        dict(utc=True, errors="coerce", format="mixed"),
        dict(utc=True, errors="coerce", format="ISO8601"),
    ]
    for kw in strategies:
        try:
            dt = pd.to_datetime(raw, **kw)
        except Exception:
            continue
        parsed = dt.notna().sum()
        if parsed > best_count:
            best, best_count = dt, parsed

    dt_unix = _parse_unix(raw)
    if dt_unix is not None:
        parsed = dt_unix.notna().sum()
        if parsed > best_count:
            best, best_count = dt_unix, parsed

    # Accept if we parsed at least 20% of non-null values (or 10 rows, whichever is lower).
    threshold = max(1, min(10, int(non_null * 0.2)))
    if best is not None and best_count >= threshold:
        return best
    return None


# Word-boundary regex so "Sentiment" no longer matches via its substring "time".
_DATETIME_COL_RE = re.compile(
    r"(?:^|[_\W])(time|date|timestamp|created|posted|published|updated|added)(?:$|[_\W])",
    re.I,
)

def _auto_pick_datetime_column(df: pd.DataFrame) -> Optional[str]:
    # Prefer columns whose name matches a datetime keyword at a word boundary,
    # ranked by how many values we can actually parse.
    candidates = [c for c in df.columns if _DATETIME_COL_RE.search(str(c))]
    best_col, best_count = None, 0
    for c in candidates:
        parsed = _parse_datetimes(df[c], dayfirst=False)
        count = int(parsed.notna().sum()) if parsed is not None else 0
        if count > best_count:
            best_col, best_count = c, count
    if best_col:
        return best_col
    # Fallback: scan every column and take whichever parses the most values.
    for c in df.columns:
        parsed = _parse_datetimes(df[c], dayfirst=False)
        count = int(parsed.notna().sum()) if parsed is not None else 0
        if count > best_count:
            best_col, best_count = c, count
    return best_col or (df.columns[0] if len(df.columns) else None)


# ----------------------------
# Flexible column detection & tokenization (URLs, hashtags, text)
# ----------------------------
_URL_COL_RE = re.compile(
    r"(?:^|[_\W])(url|urls|link|links|href|expanded[_\s]?urls?|short[_\s]?urls?)(?:$|[_\W])",
    re.I,
)
_HASHTAG_COL_RE = re.compile(
    r"(?:^|[_\W])(hashtag|hashtags|tag|tags)(?:$|[_\W])", re.I,
)
_TEXT_COL_RE = re.compile(
    r"(?:^|[_\W])(text|body|message|content|caption|full[_\s]?text|post|title)(?:$|[_\W])",
    re.I,
)
_ID_COL_RE = re.compile(
    r"(?:^|[_\W])(post[_\s]?id|resource[_\s]?id|mention[_\s]?id|thread[_\s]?id|tweet[_\s]?id|id)(?:$|[_\W])",
    re.I,
)

_URL_EXTRACT_RE = re.compile(r"https?://[^\s,;\"'<>\]\[|]+", re.I)
_HASHTAG_EXTRACT_RE = re.compile(r"#[\w\u00C0-\uFFFF]+")


def _pick_column(
    df: pd.DataFrame,
    requested: Optional[str],
    pattern: Optional[re.Pattern],
    preferred_names: Optional[list[str]] = None,
) -> Optional[str]:
    """
    Return *requested* if present; else try exact matches against *preferred_names*
    (case/space/underscore insensitive); else first non-empty column matching *pattern*.
    """
    if requested and requested in df.columns:
        return requested

    def _norm(s: str) -> str:
        return re.sub(r"[\s_\-]+", "", s.strip().lower())

    if preferred_names:
        normed = {_norm(c): c for c in df.columns}
        for name in preferred_names:
            hit = normed.get(_norm(name))
            if hit and df[hit].notna().any():
                return hit

    if pattern is not None:
        for c in df.columns:
            if pattern.search(str(c)) and df[c].notna().any():
                return c
    return None


def _tokenize_urls(cell) -> list[str]:
    """Extract URL tokens from arbitrary cell content — single URL, CSV list, JSON, Brandwatch style, etc."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    s = str(cell).strip()
    if not s or s.lower() in ("nan", "none", "null", "[]", "{}"):
        return []
    out: list[str] = []
    for u in _URL_EXTRACT_RE.findall(s):
        u = u.rstrip('.,;)"\'>]}')
        if u:
            out.append(u)
    return out


def _tokenize_hashtags(cell, strict: bool = True) -> list[str]:
    """
    Extract hashtag tokens from arbitrary cell content.

    - Any literal `#tag` is always collected.
    - When *strict* is True (text columns), anything lacking a `#` is ignored —
      we never invent hashtags from free prose.
    - When *strict* is False (explicit hashtag columns like Brandwatch's "Hashtags"),
      a short comma/semicolon-separated list of bare tag tokens is also accepted,
      but only if the whole cell looks tag-like (no multi-word phrases, short tokens,
      small count).
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    s = str(cell).strip()
    if not s or s.lower() in ("nan", "none", "null", "[]"):
        return []
    tags = _HASHTAG_EXTRACT_RE.findall(s)
    if tags:
        return [t.lower() for t in tags]
    if strict:
        return []
    # Permissive fallback for dedicated hashtag columns.
    parts = [p.strip().lstrip("#") for p in re.split(r"[,;\n]+", s)]
    parts = [p for p in parts if p]
    if not parts or len(parts) > 30:
        return []
    if not all(re.match(r"^\w{2,30}$", p) for p in parts):
        return []
    return ["#" + p.lower() for p in parts]


def _explode_tokens(df_work: pd.DataFrame, source: Optional[pd.Series], tokenizer, col_name: str) -> pd.DataFrame:
    """
    Expand each event into (ts, entity, token, row_index) rows by applying
    *tokenizer* to the aligned value in *source*. Aligns on df_work's original index.
    The *row_index* column preserves the source-CSV row number for later lookups.
    """
    cols_out = list(df_work.columns) + [col_name, "row_index"]
    if source is None or df_work.empty:
        return pd.DataFrame(columns=cols_out)
    aligned = source.reindex(df_work.index)
    tokens = aligned.apply(tokenizer)
    base = df_work.copy()
    base["row_index"] = base.index
    base[col_name] = tokens
    base = base[base[col_name].map(lambda x: isinstance(x, list) and len(x) > 0)]
    if base.empty:
        return pd.DataFrame(columns=cols_out)
    return base.explode(col_name, ignore_index=True).dropna(subset=[col_name, "ts"])


def _ensure_timezone(dt: pd.Series, tz_name: Optional[str]) -> pd.Series:
    if tz_name:
        try:
            return dt.dt.tz_convert(tz_name)
        except Exception:
            try:
                return dt.dt.tz_localize("UTC").dt.tz_convert(tz_name)
            except Exception:
                return dt
    return dt
def _excel_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with timezone-aware datetime columns made naive (Excel-safe)."""
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    out = df.copy()
    for col in out.columns:
        try:
            if is_datetime64tz_dtype(out[col]):
                out[col] = out[col].dt.tz_localize(None)
        except Exception:
            # If anything odd happens, stringify that column as a fallback
            out[col] = out[col].astype(str)
    return out
# ----------------------------
# Stats helpers
# ----------------------------
def _entropy_bits(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def _gini(arr: np.ndarray) -> float:
    x = np.array(arr, dtype=float)
    if x.size == 0 or np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    g = (2.0 * np.sum((np.arange(1, n + 1)) * x) / (n * x.sum())) - ((n + 1.0) / n)
    return float(g)

def _autocorr_top(series: pd.Series, max_lag: int = 120) -> pd.DataFrame:
    out = []
    s = series.astype(float)
    for lag in range(1, max_lag + 1):
        try:
            ac = s.autocorr(lag=lag)
            if not pd.isna(ac):
                out.append({"lag_minutes": lag, "autocorr": float(ac)})
        except Exception:
            continue
    df = pd.DataFrame(out).sort_values("autocorr", ascending=False)
    return df.head(10)

# ----------------------------
# Core analysis
# ----------------------------
# ------- NEW: small, focused helpers -------

def _prep_timeseries(
    df: pd.DataFrame,
    ts_col: str,
    tz_name: Optional[str],
    dayfirst: bool,
    entity_col: Optional[str] = None
) -> tuple[pd.Series, pd.DataFrame, bool]:
    """
    Parse timestamps, apply timezone, and build a working frame.

    Returns:
      ts         : pd.Series of parsed, tz-adjusted timestamps (sorted, NA dropped)
      df_work    : DataFrame with columns ["ts"] and optional ["entity"], sorted by ts
      entity_mode: True if entity_col provided and present in df
    """
    parsed = _parse_datetimes(df[ts_col], dayfirst=dayfirst)
    if parsed is None or parsed.notna().sum() == 0:
        raise ValueError("Could not parse the chosen datetime column with known formats or Unix epochs.")

    ts = parsed.dropna().sort_values()
    ts = _ensure_timezone(ts, tz_name)

    entity_mode = bool(entity_col) and (entity_col in df.columns)
    df_work = pd.DataFrame({"ts": parsed})
    if entity_mode:
        df_work["entity"] = df[entity_col].astype(str)
    df_work = df_work.dropna(subset=["ts"]).sort_values("ts")
    df_work["ts"] = _ensure_timezone(df_work["ts"], tz_name)

    return ts, df_work, entity_mode


def _compute_volume_tables(ts: pd.Series) -> dict[str, pd.DataFrame]:
    """
    Standard volume buckets for quick profiling and dashboards.
    """
    by_second = ts.dt.second.value_counts().rename_axis("second").sort_index().reset_index(name="count")
    by_minute = ts.dt.minute.value_counts().rename_axis("minute").sort_index().reset_index(name="count")
    by_hour   = ts.dt.hour.value_counts().rename_axis("hour").sort_index().reset_index(name="count")
    by_dow    = ts.dt.dayofweek.value_counts().rename_axis("dow").sort_index().reset_index(name="count")
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_dow["dow_name"] = by_dow["dow"].apply(lambda i: dow_names[int(i)] if pd.notna(i) else None)
    by_dom    = ts.dt.day.value_counts().rename_axis("day_of_month").sort_index().reset_index(name="count")
    by_date   = ts.dt.floor("D").value_counts().rename_axis("date").sort_index().reset_index(name="count")

    return {
        "BY_SECOND": by_second,
        "BY_MINUTE": by_minute,
        "BY_HOUR": by_hour,
        "BY_DOW": by_dow[["dow", "dow_name", "count"]],
        "BY_DOM": by_dom,
        "BY_DATE": by_date,
    }


def _compute_intervals(ts: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Inter-arrival analysis (seconds): raw intervals, summary stats, and rounded counts.
    """
    diffs = ts.diff().dropna().dt.total_seconds()
    intervals = pd.DataFrame({"interval_seconds": diffs})

    if len(diffs):
        stats = {
            "n_intervals": int(len(diffs)),
            "min": float(np.nanmin(diffs)),
            "p05": float(np.nanpercentile(diffs, 5)),
            "median": float(np.nanmedian(diffs)),
            "mean": float(np.nanmean(diffs)),
            "p95": float(np.nanpercentile(diffs, 95)),
            "max": float(np.nanmax(diffs)),
            "std": float(np.nanstd(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
            "cv": float((np.nanstd(diffs, ddof=1) / np.nanmean(diffs))) if np.nanmean(diffs) not in (0, np.nan) and len(diffs) > 1 else np.nan,
        }
        rounded = np.round(diffs).astype(int)
        interval_counts = (
            pd.Series(rounded).value_counts()
            .rename_axis("interval_seconds_rounded")
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
    else:
        stats = {"n_intervals": 0, "min": np.nan, "p05": np.nan, "median": np.nan, "mean": np.nan,
                 "p95": np.nan, "max": np.nan, "std": np.nan, "cv": np.nan}
        interval_counts = pd.DataFrame(columns=["interval_seconds_rounded", "count"])

    return intervals, pd.DataFrame([stats]), interval_counts


def _compute_coordination_indicators(ts: pd.Series, by_minute: pd.DataFrame) -> dict:
    """
    Global, entity-agnostic indicators that often surface automation or scheduling.
    """
    total = len(ts)
    sec = ts.dt.second
    minute = ts.dt.minute

    per_minute_counts = by_minute["count"].to_numpy() if not by_minute.empty else np.array([])

    pct_sec_00 = float((sec == 0).sum()) / total if total else 0.0
    pct_sec_30 = float((sec == 30).sum()) / total if total else 0.0
    pct_sec_mult_5 = float(((sec % 5) == 0).sum()) / total if total else 0.0

    pct_min_00 = float((minute == 0).sum()) / total if total else 0.0
    pct_min_30 = float((minute == 30).sum()) / total if total else 0.0
    pct_min_mult_5 = float(((minute % 5) == 0).sum()) / total if total else 0.0
    pct_min_quarter = float(minute.isin([0, 15, 30, 45]).sum()) / total if total else 0.0

    dup_counts = ts.dt.floor("s").value_counts()
    duplicate_share = float(dup_counts[dup_counts >= 2].sum()) / total if total else 0.0

    per_min_series = ts.dt.floor("min").value_counts().sort_index()
    if len(per_min_series) > 0 and per_min_series.median() > 0:
        burst_ratio = float(per_min_series.max()) / float(per_min_series.median())
    else:
        burst_ratio = np.nan

    minute_entropy_bits = _entropy_bits(per_minute_counts) if per_minute_counts.size else 0.0
    minute_entropy_norm = minute_entropy_bits / math.log2(60) if per_minute_counts.sum() > 0 else np.nan
    minute_gini = _gini(per_minute_counts)

    top_minute_idx = int(by_minute.sort_values("count", ascending=False)["minute"].iloc[0]) if not by_minute.empty else None
    top_minute_share = float(by_minute["count"].max()) / total if total and not by_minute.empty else np.nan

    # Autocorrelation (minute-level), computed once, reused everywhere
    ac_top = _autocorr_top(per_min_series, max_lag=120)

    def _ac_at(lag: int) -> float:
        r = ac_top[ac_top["lag_minutes"] == lag]
        return float(r["autocorr"].iloc[0]) if not r.empty else np.nan

    ac_5, ac_10, ac_15, ac_30, ac_60 = _ac_at(5), _ac_at(10), _ac_at(15), _ac_at(30), _ac_at(60)

    # Top duplicate timestamps (most-collided seconds)
    top_dupes = (
        dup_counts[dup_counts >= 2]
        .sort_values(ascending=False)
        .rename_axis("timestamp_floor_s")
        .reset_index(name="count")
        .head(50)
    )

    return {
        "total": total,
        "pct_sec_00": pct_sec_00,
        "pct_sec_30": pct_sec_30,
        "pct_sec_mult_5": pct_sec_mult_5,
        "pct_min_00": pct_min_00,
        "pct_min_30": pct_min_30,
        "pct_min_mult_5": pct_min_mult_5,
        "pct_min_quarter": pct_min_quarter,
        "duplicate_share": duplicate_share,
        "burst_ratio": burst_ratio,
        "minute_entropy_bits": minute_entropy_bits,
        "minute_entropy_norm": minute_entropy_norm,
        "minute_gini": minute_gini,
        "top_minute_idx": top_minute_idx,
        "top_minute_share": top_minute_share,
        "ac_top": ac_top,
        "ac_5": ac_5, "ac_10": ac_10, "ac_15": ac_15, "ac_30": ac_30, "ac_60": ac_60,
        "top_dupes": top_dupes,
    }


def _entity_near_simultaneity_and_jitter(
    df_work: pd.DataFrame,
    co_window_seconds: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int, float, float, int, int]:
    """
    Entity-aware features:
      - Near-simultaneity windows (>=2 unique entities within co_window_seconds)
      - Pairwise co-post counts across windows
      - Per-entity synchrony (share of events in multi-entity windows)
      - Per-entity jitter (uniform-gap signatures)

    Returns:
      near_sim_windows, pairwise_coposts, entity_synchrony, entity_jitter,
      windows_copost_n, share_events_in_copost_windows, top_pair_coposts,
      entities_with_repetition, entities_total
    """
    # Defaults (for empty or no-entity scenarios)
    empty_df_ents = pd.DataFrame()
    if df_work.empty or "entity" not in df_work.columns:
        return (empty_df_ents, empty_df_ents, empty_df_ents, empty_df_ents,
                0, np.nan, np.nan, 0, 0)

    # Bin to near-simultaneity window
    bin_freq = f"{int(co_window_seconds)}s"
    df_work = df_work.copy()
    df_work["bin"] = df_work["ts"].dt.floor(bin_freq)

    g = df_work.groupby("bin")
    win = g.agg(event_count=("ts", "size"),
                unique_entities=("entity", pd.Series.nunique),
                entities=("entity", lambda x: list(pd.unique(x)))).reset_index()

    # Windows with >=2 entities
    win_multi = win[win["unique_entities"] >= 2].copy()
    if not win_multi.empty:
        win_multi["window_start"] = win_multi["bin"]
        win_multi["window_end"] = win_multi["bin"] + pd.Timedelta(seconds=co_window_seconds - 1)
        win_multi["entities"] = win_multi["entities"].apply(
            lambda lst: ", ".join(map(str, lst[:20])) + (" …" if len(lst) > 20 else "")
        )
        near_sim_windows = win_multi[["window_start", "window_end", "event_count", "unique_entities", "entities"]] \
            .sort_values(["unique_entities", "event_count"], ascending=[False, False]) \
            .reset_index(drop=True)
        near_sim_windows.insert(0, "window_id", range(1, len(near_sim_windows) + 1))
    else:
        near_sim_windows = pd.DataFrame(columns=["window_id", "window_start", "window_end", "event_count", "unique_entities", "entities"])
    windows_copost_n = int(len(near_sim_windows))

    # Per-event copost flag
    df_work = df_work.merge(win[["bin", "unique_entities"]], on="bin", how="left")
    df_work["is_copost"] = df_work["unique_entities"].fillna(1) >= 2

    # Entity synchrony
    s = df_work.groupby("entity").agg(
        events=("ts", "size"),
        copost_events=("is_copost", lambda x: int(x.sum()))
    ).reset_index()
    s["copost_share"] = s["copost_events"] / s["events"].replace(0, np.nan)
    entity_synchrony = s.sort_values(["copost_share", "events"], ascending=[False, False])
    entities_total = int(len(entity_synchrony))

    # Share of all events that fall inside copost windows
    share_events_in_copost_windows = float(df_work["is_copost"].sum()) / float(len(df_work)) if len(df_work) else np.nan

    # Pairwise co-posts across windows
    pairs = PyCounter()
    MAX_ENTITIES_PER_WINDOW_FOR_PAIRS = 200
    if not win_multi.empty:
        for ents in win_multi["entities"].str.split(", ").dropna():
            ent_set = list(dict.fromkeys([e.strip() for e in ents if e.strip()]))
            if len(ent_set) <= MAX_ENTITIES_PER_WINDOW_FOR_PAIRS:
                for a, b in combinations(sorted(ent_set), 2):
                    pairs[(a, b)] += 1
    if pairs:
        pairwise_coposts = pd.DataFrame(
            [(a, b, c) for (a, b), c in pairs.items()],
            columns=["entity_a", "entity_b", "copost_windows"]
        ).sort_values("copost_windows", ascending=False)
        top_pair_coposts = int(pairwise_coposts["copost_windows"].iloc[0])
    else:
        pairwise_coposts = pd.DataFrame(columns=["entity_a", "entity_b", "copost_windows"])
        top_pair_coposts = np.nan

    # Jitter per entity (uniform gap signatures)
    ej_rows = []
    for ent, grp in df_work.sort_values("ts").groupby("entity"):
        if len(grp) <= 1:
            ej_rows.append({
                "entity": ent, "n_events": len(grp), "n_intervals": 0,
                "mean_interval_s": np.nan, "std_interval_s": np.nan, "cv_interval": np.nan,
                "modal_interval_s": np.nan, "modal_interval_share": np.nan, "near_modal_share": np.nan,
                "repetition_flag": False
            })
            continue

        secs = pd.Series(pd.to_datetime(grp["ts"].values)).diff().dropna().dt.total_seconds().values
        if secs.size == 0:
            ej_rows.append({
                "entity": ent, "n_events": len(grp), "n_intervals": 0,
                "mean_interval_s": np.nan, "std_interval_s": np.nan, "cv_interval": np.nan,
                "modal_interval_s": np.nan, "modal_interval_share": np.nan, "near_modal_share": np.nan,
                "repetition_flag": False
            })
            continue

        mean_iv = float(np.mean(secs))
        std_iv = float(np.std(secs, ddof=1)) if len(secs) > 1 else 0.0
        cv_iv = float(std_iv / mean_iv) if mean_iv > 0 and len(secs) > 1 else np.nan

        rounded = np.round(secs).astype(int)
        counts = PyCounter(rounded)
        if counts:
            modal_interval, modal_count = max(counts.items(), key=lambda kv: kv[1])
            modal_share = float(modal_count) / float(len(rounded))
            near_modal = np.sum(np.abs(rounded - modal_interval) <= 1)
            near_modal_share = float(near_modal) / float(len(rounded))
            repetition_flag = bool((len(rounded) >= 5) and (modal_share >= 0.50))
        else:
            modal_interval, modal_share, near_modal_share, repetition_flag = np.nan, np.nan, np.nan, False

        ej_rows.append({
            "entity": ent,
            "n_events": int(len(grp)),
            "n_intervals": int(len(rounded)),
            "mean_interval_s": mean_iv,
            "std_interval_s": std_iv,
            "cv_interval": cv_iv,
            "modal_interval_s": int(modal_interval) if not pd.isna(modal_interval) else np.nan,
            "modal_interval_share": modal_share,
            "near_modal_share": near_modal_share,
            "repetition_flag": repetition_flag
        })

    entity_jitter = pd.DataFrame(ej_rows).sort_values(
        ["repetition_flag", "modal_interval_share", "n_intervals"], ascending=[False, False, False]
    )
    entities_with_repetition = int(entity_jitter["repetition_flag"].sum())

    return (near_sim_windows, pairwise_coposts, entity_synchrony, entity_jitter,
            windows_copost_n, share_events_in_copost_windows, top_pair_coposts,
            entities_with_repetition, entities_total)


# ----------------------------
# CIB features
# ----------------------------

def _analyze_amplification(
    exploded: pd.DataFrame,
    token_col: str,
    window_seconds: int,
    min_entities: int,
) -> pd.DataFrame:
    """
    Flag tokens (URLs, hashtags, …) shared by ≥min_entities distinct entities inside one
    time bin of window_seconds.
    """
    if exploded is None or exploded.empty or "entity" not in exploded.columns:
        return pd.DataFrame()
    df = exploded.copy()
    df["bin"] = df["ts"].dt.floor(f"{int(window_seconds)}s")
    grp = df.groupby([token_col, "bin"]).agg(
        distinct_entities=("entity", "nunique"),
        total_shares=("entity", "size"),
        first_ts=("ts", "min"),
        last_ts=("ts", "max"),
        entities_list=("entity", lambda x: list(pd.unique(x))),
    ).reset_index()
    grp = grp[grp["distinct_entities"] >= int(min_entities)].copy()
    cols = ["group_id", token_col, "window_start", "first_ts", "last_ts", "distinct_entities", "total_shares", "entities"]
    if grp.empty:
        return pd.DataFrame(columns=cols)
    grp = grp.rename(columns={"bin": "window_start"})
    grp["entities"] = grp["entities_list"].apply(
        lambda lst: ", ".join(map(str, lst[:20])) + (" …" if len(lst) > 20 else "")
    )
    grp = grp.drop(columns=["entities_list"])
    result = grp.sort_values(
        ["distinct_entities", "total_shares"], ascending=[False, False]
    ).reset_index(drop=True)
    result.insert(0, "group_id", range(1, len(result) + 1))
    return result[cols]


def _analyze_entity_hour_profile(df_work: pd.DataFrame) -> pd.DataFrame:
    """
    Per-entity hour-of-day distribution with Shannon entropy. Flags "always-on" accounts
    (broad, high-entropy 24/7 activity) and narrow accounts (≤4 hours of activity).
    """
    if df_work.empty or "entity" not in df_work.columns:
        return pd.DataFrame()
    df = df_work.copy()
    df["hour"] = df["ts"].dt.hour.astype(int)
    log2_24 = math.log2(24)
    rows = []
    for ent, grp in df.groupby("entity"):
        counts = np.bincount(grp["hour"].to_numpy(), minlength=24)[:24]
        n = int(counts.sum())
        if n == 0:
            continue
        e_bits = _entropy_bits(counts.astype(float))
        e_norm = e_bits / log2_24
        active_hours = int((counts > 0).sum())
        top_hour = int(counts.argmax())
        top_hour_share = float(counts.max() / n)
        rows.append({
            "entity": ent,
            "n_events": n,
            "active_hours_of_24": active_hours,
            "top_hour": top_hour,
            "top_hour_share": top_hour_share,
            "hour_entropy_bits": e_bits,
            "hour_entropy_normalized": e_norm,
            "always_on_flag": bool(active_hours >= 22 and e_norm >= 0.90 and n >= 20),
            "narrow_hours_flag": bool(active_hours <= 4 and n >= 10),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["always_on_flag", "hour_entropy_normalized", "n_events"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _analyze_burst_correlation(
    df_work: pd.DataFrame,
    top_n: int = 50,
    min_correlation: float = 0.5,
    min_events_per_entity: int = 10,
) -> pd.DataFrame:
    """
    Pearson correlation of per-minute posting counts across the top-N most active entities.
    Gated on activity to keep the O(N²) comparison tractable.
    """
    if df_work.empty or "entity" not in df_work.columns:
        return pd.DataFrame()
    df = df_work.copy()
    df["minute"] = df["ts"].dt.floor("min")
    ent_counts = df["entity"].value_counts()
    ent_counts = ent_counts[ent_counts >= int(min_events_per_entity)].head(int(top_n))
    if len(ent_counts) < 2:
        return pd.DataFrame(columns=["entity_a", "entity_b", "pearson_r", "events_a", "events_b"])
    top_entities = ent_counts.index.tolist()
    df = df[df["entity"].isin(top_entities)]
    pivot = (
        df.groupby(["minute", "entity"]).size()
        .unstack("entity", fill_value=0)
        .sort_index()
    )
    if pivot.shape[0] < 3 or pivot.shape[1] < 2:
        return pd.DataFrame(columns=["entity_a", "entity_b", "pearson_r", "events_a", "events_b"])
    corr = pivot.corr(method="pearson")
    rows = []
    entities = list(corr.columns)
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            a, b = entities[i], entities[j]
            c = corr.iat[i, j]
            if pd.isna(c) or c < float(min_correlation):
                continue
            rows.append({
                "entity_a": a,
                "entity_b": b,
                "pearson_r": float(c),
                "events_a": int(ent_counts[a]),
                "events_b": int(ent_counts[b]),
            })
    if not rows:
        return pd.DataFrame(columns=["entity_a", "entity_b", "pearson_r", "events_a", "events_b"])
    return pd.DataFrame(rows).sort_values("pearson_r", ascending=False).reset_index(drop=True)


def _analyze_dormancy_bursts(
    df_work: pd.DataFrame,
    dormancy_days: float = 30,
    burst_hours: float = 24,
    min_burst_events: int = 5,
) -> pd.DataFrame:
    """
    Entities that went silent for ≥dormancy_days then posted ≥min_burst_events within burst_hours
    of reactivation. Classic "dormant sleeper cell" reactivation pattern.
    """
    if df_work.empty or "entity" not in df_work.columns:
        return pd.DataFrame()
    dormancy_td = pd.Timedelta(days=float(dormancy_days))
    burst_td = pd.Timedelta(hours=float(burst_hours))
    min_events = int(min_burst_events)
    rows = []
    for ent, grp in df_work.sort_values("ts").groupby("entity"):
        ts_series = grp["ts"].reset_index(drop=True)
        if len(ts_series) < min_events + 1:
            continue
        gaps = ts_series.diff()
        for i in range(1, len(ts_series)):
            gap = gaps.iloc[i]
            if pd.notna(gap) and gap >= dormancy_td:
                reactivation = ts_series.iloc[i]
                burst_end = reactivation + burst_td
                n_in_burst = int(((ts_series >= reactivation) & (ts_series <= burst_end)).sum())
                if n_in_burst >= min_events:
                    rows.append({
                        "entity": ent,
                        "dormancy_days": float(gap.total_seconds() / 86400),
                        "reactivation_ts": reactivation,
                        "burst_end_ts": burst_end,
                        "events_in_burst": n_in_burst,
                        "total_events": int(len(ts_series)),
                    })
                    break  # report first qualifying reactivation per entity
    cols = ["entity", "dormancy_days", "reactivation_ts", "burst_end_ts", "events_in_burst", "total_events"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols].sort_values(
        ["events_in_burst", "dormancy_days"], ascending=[False, False]
    ).reset_index(drop=True)


# ------- REFACTORED: main orchestrator -------

def analyze_temporal(
    df: pd.DataFrame,
    ts_col: str,
    tz_name: Optional[str],
    dayfirst: bool,
    entity_col: Optional[str] = None,
    co_window_seconds: int = 10,
    url_col: Optional[str] = None,
    hashtag_col: Optional[str] = None,
    text_col: Optional[str] = None,
    id_col: Optional[str] = None,
    amp_window_seconds: int = 300,
    min_amp_entities: int = 3,
    dormancy_days: float = 30,
    burst_hours: float = 24,
    min_burst_events: int = 5,
    corr_top_n: int = 50,
    corr_min: float = 0.5,
    corr_min_events_per_entity: int = 10,
):
    """
    End-to-end temporal analysis.

    Parameters:
      df                : Input DataFrame.
      ts_col            : Column name containing timestamps (string, Unix seconds/ms/µs/ns, or parseable datetime).
      tz_name           : Optional timezone name; timestamps are converted from UTC to this timezone for bucketing.
      dayfirst          : If True, parse day-first date strings.
      entity_col        : Optional column with entity/account labels for near-simultaneity and jitter features.
      co_window_seconds : Window size, in seconds, for near-simultaneity detection (default: 10).

    Returns (dict of DataFrames and preview payload):
      BY_SECOND, BY_MINUTE, BY_HOUR, BY_DOW, BY_DOM, BY_DATE,
      INTERVALS, INTERVAL_STATS, INTERVAL_COUNTS,
      INDICATORS, AUTOCORR_MIN, TOP_DUPLICATE_TIMESTAMPS,
      NEAR_SIMULTANEITY_WINDOWS, PAIRWISE_COPOSTS, ENTITY_SYNCHRONY, ENTITY_JITTER,
      _preview
    """
    # 1) Parse/prep
    ts, df_work, entity_mode = _prep_timeseries(df, ts_col, tz_name, dayfirst, entity_col)

    # 2) Volumes
    vols = _compute_volume_tables(ts)
    by_minute = vols["BY_MINUTE"]

    # 3) Intervals
    intervals, interval_stats, interval_counts = _compute_intervals(ts)

    # 4) Global indicators (automation/scheduling fingerprints)
    ci = _compute_coordination_indicators(ts, by_minute)
    ac_top = ci["ac_top"]                          # precomputed autocorr table
    top_dupes = ci["top_dupes"]

    # 5) Entity-aware near-simultaneity & jitter (optional)
    (near_sim_windows, pairwise_coposts, entity_synchrony, entity_jitter,
     windows_copost_n, share_events_in_copost_windows, top_pair_coposts,
     entities_with_repetition, entities_total) = _entity_near_simultaneity_and_jitter(df_work, co_window_seconds)

    # 5b) CIB features (all entity-gated)
    url_amplification = pd.DataFrame()
    hashtag_storms = pd.DataFrame()
    entity_hour_profile = pd.DataFrame()
    burst_correlation = pd.DataFrame()
    dormancy_bursts = pd.DataFrame()
    url_exploded: pd.DataFrame = pd.DataFrame()
    tag_exploded: pd.DataFrame = pd.DataFrame()

    if entity_mode:
        # URL amplification
        if url_col and url_col in df.columns:
            url_exploded = _explode_tokens(df_work, df[url_col], _tokenize_urls, "url")
            url_amplification = _analyze_amplification(
                url_exploded, "url", amp_window_seconds, min_amp_entities,
            )

        # Hashtag storms: prefer hashtag column (permissive tokenizer for bare tag lists);
        # fall back to text column using the strict tokenizer (only literal #tags).
        hashtag_source = None
        hashtag_tokenizer = lambda c: _tokenize_hashtags(c, strict=True)
        if hashtag_col and hashtag_col in df.columns:
            hashtag_source = df[hashtag_col]
            hashtag_tokenizer = lambda c: _tokenize_hashtags(c, strict=False)
        elif text_col and text_col in df.columns:
            hashtag_source = df[text_col]
        if hashtag_source is not None:
            tag_exploded = _explode_tokens(df_work, hashtag_source, hashtag_tokenizer, "hashtag")
            hashtag_storms = _analyze_amplification(
                tag_exploded, "hashtag", amp_window_seconds, min_amp_entities,
            )

        entity_hour_profile = _analyze_entity_hour_profile(df_work)
        burst_correlation = _analyze_burst_correlation(
            df_work,
            top_n=corr_top_n,
            min_correlation=corr_min,
            min_events_per_entity=corr_min_events_per_entity,
        )
        dormancy_bursts = _analyze_dormancy_bursts(
            df_work,
            dormancy_days=dormancy_days,
            burst_hours=burst_hours,
            min_burst_events=min_burst_events,
        )

    # 5c) Per-event detail sheets for flagged groups (joins back to source CSV).
    near_sim_events = pd.DataFrame()
    url_amp_events = pd.DataFrame()
    hashtag_storm_events = pd.DataFrame()

    def _attach_source_cols(events: pd.DataFrame, index_col: str = "row_index") -> pd.DataFrame:
        """Enrich an events frame with id/text/url columns looked up from the source df."""
        if events.empty or index_col not in events.columns:
            return events
        out = events.copy()
        idx = pd.Index(out[index_col].tolist())
        if id_col and id_col in df.columns:
            out["id"] = df[id_col].reindex(idx).to_numpy()
        if text_col and text_col in df.columns:
            out["text"] = df[text_col].reindex(idx).to_numpy()
        # For near-sim events only, add the raw URL cell (context); URL/hashtag amp events
        # already carry the token, so skip there.
        return out

    if entity_mode and not near_sim_windows.empty:
        bin_freq = f"{int(co_window_seconds)}s"
        dfw = df_work.copy()
        dfw["bin"] = dfw["ts"].dt.floor(bin_freq)
        dfw["row_index"] = dfw.index
        flagged = near_sim_windows[["window_id", "window_start", "window_end", "unique_entities"]].rename(
            columns={"window_start": "bin"}
        )
        events = dfw.merge(flagged, on="bin", how="inner")
        # Include the raw URL cell for near-sim (it's the most useful context column here)
        if url_col and url_col in df.columns:
            events["url"] = df[url_col].reindex(pd.Index(events["row_index"])).to_numpy()
        events = events.rename(columns={"bin": "window_start"})
        events = _attach_source_cols(events)
        cols = ["window_id", "window_start", "window_end", "unique_entities", "ts", "entity", "row_index"]
        for extra in ("id", "text", "url"):
            if extra in events.columns:
                cols.append(extra)
        near_sim_events = events[cols].sort_values(["window_id", "ts"]).reset_index(drop=True)

    if entity_mode and not url_amplification.empty and not url_exploded.empty:
        ue = url_exploded.copy()
        ue["bin"] = ue["ts"].dt.floor(f"{int(amp_window_seconds)}s")
        keys = url_amplification[["group_id", "url", "window_start", "distinct_entities"]].rename(
            columns={"window_start": "bin"}
        )
        events = ue.merge(keys, on=["url", "bin"], how="inner")
        events = events.rename(columns={"bin": "window_start"})
        events = _attach_source_cols(events)
        cols = ["group_id", "url", "window_start", "distinct_entities", "ts", "entity", "row_index"]
        for extra in ("id", "text"):
            if extra in events.columns:
                cols.append(extra)
        url_amp_events = events[cols].sort_values(["group_id", "ts"]).reset_index(drop=True)

    if entity_mode and not hashtag_storms.empty and not tag_exploded.empty:
        te = tag_exploded.copy()
        te["bin"] = te["ts"].dt.floor(f"{int(amp_window_seconds)}s")
        keys = hashtag_storms[["group_id", "hashtag", "window_start", "distinct_entities"]].rename(
            columns={"window_start": "bin"}
        )
        events = te.merge(keys, on=["hashtag", "bin"], how="inner")
        events = events.rename(columns={"bin": "window_start"})
        events = _attach_source_cols(events)
        cols = ["group_id", "hashtag", "window_start", "distinct_entities", "ts", "entity", "row_index"]
        for extra in ("id", "text"):
            if extra in events.columns:
                cols.append(extra)
        hashtag_storm_events = events[cols].sort_values(["group_id", "ts"]).reset_index(drop=True)

    # 6) Indicator table (single-row summary)
    indicators = pd.DataFrame([{
        "total_events": ci["total"],
        "start": ts.iloc[0],
        "end": ts.iloc[-1],
        "pct_second_00": ci["pct_sec_00"],
        "pct_second_30": ci["pct_sec_30"],
        "pct_second_mult_5": ci["pct_sec_mult_5"],
        "pct_minute_00": ci["pct_min_00"],
        "pct_minute_30": ci["pct_min_30"],
        "pct_minute_mult_5": ci["pct_min_mult_5"],
        "pct_minute_quarter": ci["pct_min_quarter"],
        "duplicate_timestamp_share": ci["duplicate_share"],
        "burst_ratio_max_over_median_minute": ci["burst_ratio"],
        "minute_entropy_bits": ci["minute_entropy_bits"],
        "minute_entropy_normalized": ci["minute_entropy_norm"],
        "minute_gini": ci["minute_gini"],
        "top_minute": ci["top_minute_idx"],
        "top_minute_share": ci["top_minute_share"],
        "autocorr_top_lag_minutes": int(ac_top.iloc[0]["lag_minutes"]) if not ac_top.empty else np.nan,
        "autocorr_top_value": float(ac_top.iloc[0]["autocorr"]) if not ac_top.empty else np.nan,
        "autocorr_5m": ci["ac_5"], "autocorr_10m": ci["ac_10"], "autocorr_15m": ci["ac_15"], "autocorr_30m": ci["ac_30"], "autocorr_60m": ci["ac_60"],
        # Entity-aware summary
        "entity_mode": bool(entity_mode),
        "co_window_seconds": int(co_window_seconds),
        "near_simultaneity_windows": int(windows_copost_n),
        "share_events_in_copost_windows": share_events_in_copost_windows,
        "top_pair_copost_windows": top_pair_coposts,
        "entities_with_repetition_flag": entities_with_repetition,
        "entities_total": entities_total,
        "entities_repetition_share": (entities_with_repetition / entities_total) if entities_total else np.nan,
        # CIB feature summaries
        "amp_window_seconds": int(amp_window_seconds),
        "min_amp_entities": int(min_amp_entities),
        "url_amplified_rows": int(len(url_amplification)),
        "hashtag_storm_rows": int(len(hashtag_storms)),
        "entities_always_on": int(entity_hour_profile["always_on_flag"].sum()) if not entity_hour_profile.empty else 0,
        "entities_narrow_hours": int(entity_hour_profile["narrow_hours_flag"].sum()) if not entity_hour_profile.empty else 0,
        "burst_correlated_pairs": int(len(burst_correlation)),
        "dormancy_burst_entities": int(len(dormancy_bursts)),
        "near_sim_events_n": int(len(near_sim_events)),
        "url_amp_events_n": int(len(url_amp_events)),
        "hashtag_storm_events_n": int(len(hashtag_storm_events)),
    }])

    # 7) Compile return payload
    return {
        **vols,
        "INTERVALS": intervals,
        "INTERVAL_STATS": interval_stats,
        "INTERVAL_COUNTS": interval_counts,
        "INDICATORS": indicators,
        "AUTOCORR_MIN": ac_top,  # use the single computed table
        "TOP_DUPLICATE_TIMESTAMPS": top_dupes,
        "NEAR_SIMULTANEITY_WINDOWS": near_sim_windows,
        "PAIRWISE_COPOSTS": pairwise_coposts,
        "ENTITY_SYNCHRONY": entity_synchrony,
        "ENTITY_JITTER": entity_jitter,
        "URL_AMPLIFICATION": url_amplification,
        "HASHTAG_STORMS": hashtag_storms,
        "ENTITY_HOUR_PROFILE": entity_hour_profile,
        "BURST_CORRELATION": burst_correlation,
        "DORMANCY_BURSTS": dormancy_bursts,
        "NEAR_SIM_EVENTS": near_sim_events,
        "URL_AMP_EVENTS": url_amp_events,
        "HASHTAG_STORM_EVENTS": hashtag_storm_events,
        "_preview": {
            "total": int(indicators.loc[0, "total_events"]),
            "start": str(indicators.loc[0, "start"]),
            "end": str(indicators.loc[0, "end"]),
            "top_minute": int(indicators.loc[0, "top_minute"]) if not pd.isna(indicators.loc[0, "top_minute"]) else None,
            "top_minute_share": float(indicators.loc[0, "top_minute_share"]) if not pd.isna(indicators.loc[0, "top_minute_share"]) else None,
            "burst_ratio": float(indicators.loc[0, "burst_ratio_max_over_median_minute"]) if not pd.isna(indicators.loc[0, "burst_ratio_max_over_median_minute"]) else None,
        }
    }


# ----------------------------
# Route
# ----------------------------
@temporal_bp.route("/", methods=["GET", "POST"])
def temporal():
    results = {}
    if request.method == "POST":
        output_mode = request.form.get("output_mode")  # "table" | "excel"
        tz_name = request.form.get("timezone") or None
        dayfirst = bool(request.form.get("dayfirst"))
        entity_col = (request.form.get("entity_col") or "").strip() or None
        url_col_req = (request.form.get("url_col") or "").strip() or None
        hashtag_col_req = (request.form.get("hashtag_col") or "").strip() or None
        text_col_req = (request.form.get("text_col") or "").strip() or None
        id_col_req = (request.form.get("id_col") or "").strip() or None

        def _int_form(name: str, default: int) -> int:
            try:
                return int(request.form.get(name) or default)
            except Exception:
                return default

        def _float_form(name: str, default: float) -> float:
            try:
                return float(request.form.get(name) or default)
            except Exception:
                return default

        co_window_seconds = _int_form("co_window_seconds", 10)
        amp_window_seconds = _int_form("amp_window_seconds", 300)
        min_amp_entities = _int_form("min_amp_entities", 3)
        dormancy_days = _float_form("dormancy_days", 30)
        burst_hours = _float_form("burst_hours", 24)
        min_burst_events = _int_form("min_burst_events", 5)
        corr_top_n = _int_form("corr_top_n", 50)
        corr_min = _float_form("corr_min", 0.5)

        files = [f for f in request.files.getlist("csv_file") if f and f.filename]
        if not files:
            return render_template("temporal_analyzer.html", results={"error": "Please upload a CSV file."})

        frames = []
        for f in files:
            try:
                part = pd.read_csv(f)
            except Exception as e:
                return render_template(
                    "temporal_analyzer.html",
                    results={"error": f"CSV '{f.filename}' could not be read: {e}"},
                )
            if len(files) > 1:
                part["source_file"] = f.filename
            frames.append(part)
        try:
            df = pd.concat(frames, ignore_index=True, sort=False)
        except Exception as e:
            return render_template(
                "temporal_analyzer.html",
                results={"error": f"Files could not be combined: {e}"},
            )

        dt_col = (request.form.get("datetime_col") or "").strip()
        if not dt_col or dt_col not in df.columns:
            guessed = _auto_pick_datetime_column(df)
            dt_col = guessed if guessed in df.columns else None
        if not dt_col:
            return render_template("temporal_analyzer.html", results={"error": "Could not determine a datetime column. Please specify one."})

        # Auto-detect URL / hashtag / text columns when the user didn't specify them.
        url_col = _pick_column(
            df, url_col_req, _URL_COL_RE,
            preferred_names=["expanded urls", "urls", "url", "link", "links"],
        )
        hashtag_col = _pick_column(
            df, hashtag_col_req, _HASHTAG_COL_RE,
            preferred_names=["hashtags", "hashtag", "tags"],
        )
        text_col = _pick_column(
            df, text_col_req, _TEXT_COL_RE,
            preferred_names=["full text", "text", "body", "message", "content", "caption"],
        )
        id_col = _pick_column(
            df, id_col_req, _ID_COL_RE,
            preferred_names=[
                "post id", "postid", "resource id", "resourceid",
                "mention id", "mentionid", "thread id", "threadid",
                "tweet id", "tweetid", "id",
            ],
        )

        try:
            out = analyze_temporal(
                df, dt_col, tz_name, dayfirst,
                entity_col=entity_col,
                co_window_seconds=co_window_seconds,
                url_col=url_col,
                hashtag_col=hashtag_col,
                text_col=text_col,
                id_col=id_col,
                amp_window_seconds=amp_window_seconds,
                min_amp_entities=min_amp_entities,
                dormancy_days=dormancy_days,
                burst_hours=burst_hours,
                min_burst_events=min_burst_events,
                corr_top_n=corr_top_n,
                corr_min=corr_min,
            )
        except Exception as e:
            return render_template("temporal_analyzer.html", results={"error": f"Analysis failed: {e}"})

        if output_mode == "excel":
            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                # Build summary first (keeps timezone as a separate string)
                summary = pd.DataFrame([{
                    "datetime_column": dt_col,
                    "entity_column": entity_col or "(none)",
                    "url_column": url_col or "(none)",
                    "hashtag_column": hashtag_col or "(none)",
                    "text_column": text_col or "(none)",
                    "id_column": id_col or "(none)",
                    "co_window_seconds": co_window_seconds,
                    "timezone": tz_name or "UTC (as-parsed)",
                    **out["INDICATORS"].iloc[0].to_dict()
                }])

                _excel_safe(summary).to_excel(writer, sheet_name="SUMMARY", index=False)

                # Volumes
                _excel_safe(out["BY_SECOND"]).to_excel(writer, sheet_name="BY_SECOND", index=False)
                _excel_safe(out["BY_MINUTE"]).to_excel(writer, sheet_name="BY_MINUTE", index=False)
                _excel_safe(out["BY_HOUR"]).to_excel(writer, sheet_name="BY_HOUR", index=False)
                _excel_safe(out["BY_DOW"]).to_excel(writer, sheet_name="BY_DOW", index=False)
                _excel_safe(out["BY_DOM"]).to_excel(writer, sheet_name="BY_DOM", index=False)
                _excel_safe(out["BY_DATE"]).to_excel(writer, sheet_name="BY_DATE", index=False)

                # Intervals and indicators
                _excel_safe(out["INTERVALS"]).to_excel(writer, sheet_name="INTERVALS", index=False)
                _excel_safe(out["INTERVAL_STATS"]).to_excel(writer, sheet_name="INTERVAL_STATS", index=False)
                _excel_safe(out["INTERVAL_COUNTS"]).to_excel(writer, sheet_name="INTERVAL_COUNTS", index=False)
                _excel_safe(out["INDICATORS"]).to_excel(writer, sheet_name="INDICATORS", index=False)
                _excel_safe(out["AUTOCORR_MIN"]).to_excel(writer, sheet_name="AUTOCORR_MIN", index=False)
                _excel_safe(out["TOP_DUPLICATE_TIMESTAMPS"]).to_excel(writer, sheet_name="TOP_DUPLICATE_TS", index=False)

                # Entity-aware (only populated if entity_col provided)
                _excel_safe(out["NEAR_SIMULTANEITY_WINDOWS"]).to_excel(writer, sheet_name="NEAR_SIMULTANEITY_WINDOWS", index=False)
                _excel_safe(out["PAIRWISE_COPOSTS"]).to_excel(writer, sheet_name="PAIRWISE_COPOSTS", index=False)
                _excel_safe(out["ENTITY_SYNCHRONY"]).to_excel(writer, sheet_name="ENTITY_SYNCHRONY", index=False)
                _excel_safe(out["ENTITY_JITTER"]).to_excel(writer, sheet_name="ENTITY_JITTER", index=False)

                # CIB features
                _excel_safe(out["URL_AMPLIFICATION"]).to_excel(writer, sheet_name="URL_AMPLIFICATION", index=False)
                _excel_safe(out["HASHTAG_STORMS"]).to_excel(writer, sheet_name="HASHTAG_STORMS", index=False)
                _excel_safe(out["ENTITY_HOUR_PROFILE"]).to_excel(writer, sheet_name="ENTITY_HOUR_PROFILE", index=False)
                _excel_safe(out["BURST_CORRELATION"]).to_excel(writer, sheet_name="BURST_CORRELATION", index=False)
                _excel_safe(out["DORMANCY_BURSTS"]).to_excel(writer, sheet_name="DORMANCY_BURSTS", index=False)

                # Per-event detail for flagged groups (joinable back to source CSV)
                _excel_safe(out["NEAR_SIM_EVENTS"]).to_excel(writer, sheet_name="NEAR_SIM_EVENTS", index=False)
                _excel_safe(out["URL_AMP_EVENTS"]).to_excel(writer, sheet_name="URL_AMP_EVENTS", index=False)
                _excel_safe(out["HASHTAG_STORM_EVENTS"]).to_excel(writer, sheet_name="HASHTAG_STORM_EVENTS", index=False)

            output.seek(0)
            return send_file(
                output,
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                as_attachment=True,
                download_name="temporal_analysis.xlsx",
            )


        # On-page preview (unchanged, keep it light)
        preview = out["_preview"]
        table_top_minute = out["BY_MINUTE"].sort_values("count", ascending=False).head(10).to_dict(orient="records")
        table_top_hour   = out["BY_HOUR"].sort_values("count", ascending=False).head(10).to_dict(orient="records")
        table_intervals  = out["INTERVAL_COUNTS"].head(10).to_dict(orient="records")

        # NEW small previews if entity mode
        synchrony_preview = out["ENTITY_SYNCHRONY"].head(10).to_dict(orient="records") if not out["ENTITY_SYNCHRONY"].empty else []
        jitter_preview = out["ENTITY_JITTER"][out["ENTITY_JITTER"]["repetition_flag"]].head(10).to_dict(orient="records") if not out["ENTITY_JITTER"].empty else []
        copost_pair_preview = out["PAIRWISE_COPOSTS"].head(10).to_dict(orient="records") if not out["PAIRWISE_COPOSTS"].empty else []

        url_amp_preview = out["URL_AMPLIFICATION"].head(10).to_dict(orient="records") if not out["URL_AMPLIFICATION"].empty else []
        hashtag_storm_preview = out["HASHTAG_STORMS"].head(10).to_dict(orient="records") if not out["HASHTAG_STORMS"].empty else []
        burst_corr_preview = out["BURST_CORRELATION"].head(10).to_dict(orient="records") if not out["BURST_CORRELATION"].empty else []
        dormancy_preview = out["DORMANCY_BURSTS"].head(10).to_dict(orient="records") if not out["DORMANCY_BURSTS"].empty else []
        hour_profile_preview = (
            out["ENTITY_HOUR_PROFILE"][out["ENTITY_HOUR_PROFILE"]["always_on_flag"] | out["ENTITY_HOUR_PROFILE"]["narrow_hours_flag"]]
            .head(10).to_dict(orient="records")
            if not out["ENTITY_HOUR_PROFILE"].empty else []
        )

        ind = out["INDICATORS"].iloc[0]

        results = {
            "ok": True,
            "datetime_col": dt_col,
            "timezone": tz_name or "UTC (as-parsed)",
            "entity_col": entity_col or "",
            "url_col": url_col or "",
            "hashtag_col": hashtag_col or "",
            "text_col": text_col or "",
            "id_col": id_col or "",
            "co_window_seconds": co_window_seconds,
            "amp_window_seconds": amp_window_seconds,
            "summary": preview,
            "top_minutes": table_top_minute,
            "top_hours": table_top_hour,
            "top_intervals": table_intervals,
            "synchrony_preview": synchrony_preview,
            "jitter_preview": jitter_preview,
            "copost_pair_preview": copost_pair_preview,
            "url_amp_preview": url_amp_preview,
            "hashtag_storm_preview": hashtag_storm_preview,
            "burst_corr_preview": burst_corr_preview,
            "dormancy_preview": dormancy_preview,
            "hour_profile_preview": hour_profile_preview,
            "cib_counts": {
                "url_amplified_rows": int(ind["url_amplified_rows"]),
                "hashtag_storm_rows": int(ind["hashtag_storm_rows"]),
                "entities_always_on": int(ind["entities_always_on"]),
                "entities_narrow_hours": int(ind["entities_narrow_hours"]),
                "burst_correlated_pairs": int(ind["burst_correlated_pairs"]),
                "dormancy_burst_entities": int(ind["dormancy_burst_entities"]),
                "near_sim_events_n": int(ind["near_sim_events_n"]),
                "url_amp_events_n": int(ind["url_amp_events_n"]),
                "hashtag_storm_events_n": int(ind["hashtag_storm_events_n"]),
            },
        }

    return render_template("temporal_analyzer.html", results=results)
