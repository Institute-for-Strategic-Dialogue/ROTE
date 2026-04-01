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
    try:
        dt = pd.to_datetime(raw, infer_datetime_format=True, utc=True, errors="coerce")
        if dt.notna().mean() > 0.8:
            return dt
    except Exception:
        pass
    try:
        dt = pd.to_datetime(raw, infer_datetime_format=True, utc=True, errors="coerce", dayfirst=dayfirst)
        if dt.notna().mean() > 0.8:
            return dt
    except Exception:
        pass
    dt_unix = _parse_unix(raw)
    if dt_unix is not None and dt_unix.notna().mean() > 0.8:
        return dt_unix
    return None

def _auto_pick_datetime_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if re.search(r"(time|date|timestamp|created|posted|published)", str(c), re.I)]
    for c in candidates:
        return c
    return df.columns[0] if len(df.columns) else None

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

    dup_counts = ts.dt.floor("S").value_counts()
    duplicate_share = float(dup_counts[dup_counts >= 2].sum()) / total if total else 0.0

    per_min_series = ts.dt.floor("T").value_counts().sort_index()
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
    bin_freq = f"{int(co_window_seconds)}S"
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
        win_multi["window_end"] = win_multi["bin"] + pd.to_datetime(pd.Timedelta(seconds=co_window_seconds - 1))
        win_multi["entities"] = win_multi["entities"].apply(
            lambda lst: ", ".join(map(str, lst[:20])) + (" …" if len(lst) > 20 else "")
        )
        near_sim_windows = win_multi[["window_start", "window_end", "event_count", "unique_entities", "entities"]] \
            .sort_values(["unique_entities", "event_count"], ascending=[False, False])
    else:
        near_sim_windows = pd.DataFrame(columns=["window_start", "window_end", "event_count", "unique_entities", "entities"])
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


# ------- REFACTORED: main orchestrator -------

def analyze_temporal(
    df: pd.DataFrame,
    ts_col: str,
    tz_name: Optional[str],
    dayfirst: bool,
    entity_col: Optional[str] = None,
    co_window_seconds: int = 10,
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
        try:
            co_window_seconds = int(request.form.get("co_window_seconds") or 10)  # NEW
        except Exception:
            co_window_seconds = 10

        file = request.files.get("csv_file")
        if not file:
            return render_template("temporal_analyzer.html", results={"error": "Please upload a CSV file."})

        try:
            df = pd.read_csv(file)
        except Exception as e:
            return render_template("temporal_analyzer.html", results={"error": f"CSV could not be read: {e}"})

        dt_col = (request.form.get("datetime_col") or "").strip()
        if not dt_col or dt_col not in df.columns:
            guessed = _auto_pick_datetime_column(df)
            dt_col = guessed if guessed in df.columns else None
        if not dt_col:
            return render_template("temporal_analyzer.html", results={"error": "Could not determine a datetime column. Please specify one."})

        try:
            out = analyze_temporal(
                df, dt_col, tz_name, dayfirst,
                entity_col=entity_col,
                co_window_seconds=co_window_seconds,
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

        results = {
            "ok": True,
            "datetime_col": dt_col,
            "timezone": tz_name or "UTC (as-parsed)",
            "entity_col": entity_col or "",
            "co_window_seconds": co_window_seconds,
            "summary": preview,
            "top_minutes": table_top_minute,
            "top_hours": table_top_hour,
            "top_intervals": table_intervals,
            "synchrony_preview": synchrony_preview,
            "jitter_preview": jitter_preview,
            "copost_pair_preview": copost_pair_preview,
        }

    return render_template("temporal_analyzer.html", results=results)
