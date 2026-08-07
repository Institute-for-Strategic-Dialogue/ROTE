"""Nomic Atlas access for the relabeller.

Deliberately free of Flask imports so the same module drives the tool's routes
and can be exercised from a script. Progress is reported through an optional
`progress(fraction, message)` callback.

Memory is the binding constraint: ROTE runs as a single process, so a spike
takes down every tool on the site, not just this one.

Measured on a 275 MB / 175,489-row Brandwatch export, uploading to Atlas:

    build whole frame, then upload    peak 1,619 MB
    chunked at 10k rows               peak   731 MB
    chunked at 2.5k rows + gc         peak   491 MB   <- current

Chunking bounds the *transient* cost, but peak is not flat in input size: RSS
climbed 388 -> 491 MB between 15k and 150k rows uploaded, i.e. roughly
0.75 MB per 1,000 rows, on top of a ~330 MB floor. Extrapolating, ~300k rows
would peak near 600 MB and ~500k near 750 MB. The residual growth is inside the
upload path, not this module's frames, which are released each chunk.

Practical consequence: on a 512 MB instance this comfortably handles a few
hundred thousand rows and no more. Size the host to the corpus, and re-measure
before assuming a much larger one will fit.
"""

from __future__ import annotations

import gc
import re
from typing import Callable, Iterator, Optional

import pandas as pd
import pyarrow as pa

import nomic.dataset
from nomic.data_inference import convert_pyarrow_schema_for_atlas
from nomic.dataset import AtlasDataset

DEPTH_COLS = ["topic_depth_1", "topic_depth_2", "topic_depth_3"]
UPLOAD_BATCH = 10_000
# Measured on a 44-column Brandwatch export: a 10k-row chunk costs ~155 MB
# transient through coerce+preflight, which pushed peak RSS to 731 MB. At 2.5k
# that quarters, and network time dominates anyway so the extra round trips are
# free in wall-clock terms.
CSV_CHUNK = 2_500
GET_DATA_CHUNK = 1_000

NOMIC_LABEL_COL = "nomic_label"
SPECIFIC_COL = "Topic (Specific)"
BROAD_COL = "Topic (Broad)"

# Atlas stores an empty string as null, which shows up as a blank entry in the
# map's filter panel and can't be selected. Topics the analyst left without a
# broad label get this instead, so they remain filterable.
UNASSIGNED = "(unassigned)"

Progress = Optional[Callable[[float, str], None]]


def _report(progress: Progress, frac: float, msg: str) -> None:
    if progress:
        progress(max(0.0, min(1.0, frac)), msg)


# ── Auth ──────────────────────────────────────────────────────────────


def authenticate(api_key: str) -> None:
    """Authenticate in memory, without touching disk.

    nomic.login() writes the token to ~/.nomic/credentials. On a shared host
    that would persist one user's key where every later request can read it, so
    it is never called. Keys beginning `nk-` are valid bearer tokens as-is.

    dataset.py does `from .cli import refresh_bearer_token` at import time and
    holds its own reference, so patching nomic.cli alone would not take.
    """
    key = (api_key or "").strip()
    if not key:
        raise ValueError("No API key provided.")
    creds = {"token": key, "tenant": "production", "expires": None, "refresh_token": None}
    nomic.dataset.refresh_bearer_token = lambda: creds


def open_dataset(identifier: str) -> AtlasDataset:
    ident = (identifier or "").strip().rstrip("/")
    if not ident:
        raise ValueError("No dataset identifier provided.")
    m = re.search(r"atlas\.nomic\.ai/data/([^/]+/[^/]+)", ident)
    if m:
        ident = m.group(1)
    return AtlasDataset(ident)


# ── Topics ────────────────────────────────────────────────────────────


def load_topics(dataset: AtlasDataset):
    """-> (topics_df, metadata_df, id_field)."""
    if not dataset.maps:
        raise ValueError(
            f"Dataset `{dataset.identifier}` has no maps yet — its index may "
            "still be building. Wait for the map to finish, then retry."
        )
    topics = dataset.maps[0].topics
    return topics.df, topics.metadata, dataset.id_field


def depth_summary(topics_df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for depth, col in enumerate(DEPTH_COLS, start=1):
        if col not in topics_df.columns:
            continue
        labels = topics_df[col].dropna()
        examples = metadata.loc[metadata["depth"] == depth, col].dropna().head(4).tolist()
        rows.append({
            "depth": depth,
            "topics": int(labels.nunique()),
            "posts_assigned": int((labels != "").sum()),
            "examples": ", ".join(str(e) for e in examples),
        })
    return pd.DataFrame(rows)


def label_table(topics_df: pd.DataFrame, metadata: pd.DataFrame, depth: int) -> pd.DataFrame:
    """One row per topic, largest first, with blank columns for the analyst."""
    col = DEPTH_COLS[depth - 1]
    counts = topics_df[col].value_counts()
    md = metadata[metadata["depth"] == depth]

    # Nomic ships `topic_description` as a slash-joined keyword list, not prose.
    # Name it for what it is and comma-join so it reads in a spreadsheet cell.
    keywords = md.get("topic_description", pd.Series([""] * len(md))).fillna("")
    keywords = keywords.astype(str).str.replace("/", ", ", regex=False)

    out = pd.DataFrame({
        "depth": depth,
        "topic_id": md["topic_id"].values,
        "nomic_label": md[col].values,
        "common_keywords": keywords.values,
        "n_posts": [int(counts.get(lbl, 0)) for lbl in md[col].values],
        "specific_label": "",
        "broad_label": "",
    })
    return out.sort_values("n_posts", ascending=False).reset_index(drop=True)


def topic_lookup(topics_df: pd.DataFrame, id_field: str, depth: int) -> pd.Series:
    """id -> topic label, as a Series indexed by id.

    Kept instead of the full topics DataFrame so the chunked join holds only
    what it needs.
    """
    col = DEPTH_COLS[depth - 1]
    s = topics_df[[id_field, col]].dropna(subset=[col])
    return pd.Series(s[col].values, index=s[id_field].astype(str), name="topic")


# ── Sampling ──────────────────────────────────────────────────────────


def sample_ids(topics_df: pd.DataFrame, id_field: str, depth: int,
               n_per_topic: int, seed: int = 0) -> pd.DataFrame:
    """-> [id_field, topic], up to n_per_topic random ids per topic."""
    col = DEPTH_COLS[depth - 1]
    sub = topics_df[[id_field, col]].dropna(subset=[col])
    # Explicit loop: groupby.apply's handling of the grouping column is
    # deprecated in pandas 2.2+, and include_groups= is absent on older ones.
    picks = [g.sample(min(len(g), n_per_topic), random_state=seed)
             for _, g in sub.groupby(col, sort=False)]
    sampled = pd.concat(picks, ignore_index=True) if picks else sub.head(0)
    return sampled.rename(columns={col: "topic"})


def fetch_rows(dataset: AtlasDataset, ids, progress: Progress = None) -> pd.DataFrame:
    """Fetch specific rows by id, rather than pulling the whole dataset."""
    ids = [str(i) for i in ids]
    out = []
    for start in range(0, len(ids), GET_DATA_CHUNK):
        chunk = ids[start:start + GET_DATA_CHUNK]
        out.extend(dataset.get_data(ids=chunk))
        done = min(start + len(chunk), len(ids))
        _report(progress, done / max(1, len(ids)), f"fetched {done:,}/{len(ids):,} posts")
    return pd.DataFrame(out)


# ── Type coercion ─────────────────────────────────────────────────────

INT32_MAX, INT32_MIN = 2_147_483_647, -2_147_483_648


def coerce_for_atlas(df: pd.DataFrame) -> pd.DataFrame:
    """Force every column into a type Atlas accepts.

    Atlas permits only string / int32 / float32 / timestamp — booleans, lists
    and structs raise in convert_pyarrow_schema_for_atlas. Integers are coerced
    to int32, so out-of-range values must become floats.
    """
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            out[col] = s.map({True: "true", False: "false"}).fillna("").astype(str)
        elif pd.api.types.is_datetime64_any_dtype(s):
            ser = pd.to_datetime(s, errors="coerce")
            if getattr(ser.dtype, "tz", None) is not None:
                ser = ser.dt.tz_localize(None)
            out[col] = ser.astype("datetime64[ms]")
        elif pd.api.types.is_integer_dtype(s):
            out[col] = (s.astype("float32")
                        if (s.max() > INT32_MAX or s.min() < INT32_MIN)
                        else s.astype("Int32"))
        elif pd.api.types.is_float_dtype(s):
            out[col] = s.astype("float32")
        else:
            out[col] = s.fillna("").astype(str)
    return out


def preflight(df: pd.DataFrame) -> pa.Schema:
    """Run Atlas's own schema converter locally, so a rejected type fails in a
    second rather than partway through a multi-minute upload."""
    return convert_pyarrow_schema_for_atlas(pa.Table.from_pandas(df).schema)


# ── Chunked sources ───────────────────────────────────────────────────


def iter_csv_labelled(csv_path_or_buf, lookup: pd.Series, id_field: str,
                      specific_map: dict, broad_map: dict,
                      chunksize: int = CSV_CHUNK) -> Iterator[pd.DataFrame]:
    """Yield label-joined chunks, never holding more than one in memory.

    This is what keeps peak RSS flat: a 275 MB CSV read whole peaks ~1.6 GB,
    which would OOM a 512 MB instance and take the whole site with it.
    """
    reader = pd.read_csv(csv_path_or_buf, dtype=str, keep_default_na=False,
                         chunksize=chunksize)
    for chunk in reader:
        if id_field not in chunk.columns:
            raise ValueError(
                f"The CSV has no `{id_field}` column, so it cannot be joined to "
                f"the dataset. Columns found: {', '.join(list(chunk.columns)[:8])}…"
            )
        chunk[id_field] = chunk[id_field].astype(str)
        topic = chunk[id_field].map(lookup)
        keep = topic.notna()
        chunk = chunk[keep]
        topic = topic[keep]
        if chunk.empty:
            continue
        chunk[NOMIC_LABEL_COL] = topic.values
        chunk[SPECIFIC_COL] = topic.map(specific_map).values
        chunk[BROAD_COL] = topic.map(broad_map).replace("", UNASSIGNED).fillna(UNASSIGNED).values
        yield chunk


def iter_atlas_labelled(dataset: AtlasDataset, lookup: pd.Series, id_field: str,
                        specific_map: dict, broad_map: dict,
                        chunksize: int = CSV_CHUNK) -> Iterator[pd.DataFrame]:
    """Same, sourced from Atlas instead of a CSV.

    Caveat: AtlasMapData.df materialises the whole dataset in one go — there is
    no chunked read in the client — so this path's peak is proportional to
    dataset size. Callers should gate it on row count.
    """
    full = dataset.maps[0].data.df
    for start in range(0, len(full), chunksize):
        chunk = full.iloc[start:start + chunksize].copy()
        chunk[id_field] = chunk[id_field].astype(str)
        topic = chunk[id_field].map(lookup)
        keep = topic.notna()
        chunk = chunk[keep]
        topic = topic[keep]
        if chunk.empty:
            continue
        chunk[NOMIC_LABEL_COL] = topic.values
        chunk[SPECIFIC_COL] = topic.map(specific_map).values
        chunk[BROAD_COL] = topic.map(broad_map).replace("", UNASSIGNED).fillna(UNASSIGNED).values
        yield chunk


# ── Chunked upload ────────────────────────────────────────────────────


def upload_chunks(chunks: Iterator[pd.DataFrame], name: str, id_field: str,
                  indexed_field: str, description: str = "",
                  is_public: bool = False, total_hint: int = 0,
                  progress: Progress = None) -> tuple[AtlasDataset, int]:
    """Create a private dataset and upload chunk by chunk.

    Returns (dataset, rows_uploaded). The dataset is only created once the first
    chunk has passed preflight, so a type error doesn't leave an empty dataset
    stranded in the account.
    """
    dataset = None
    sent = 0

    for chunk in chunks:
        coerced = coerce_for_atlas(chunk)
        preflight(coerced)
        del chunk
        chunk = coerced

        if dataset is None:
            dataset = AtlasDataset(
                name,
                description=description or "Relabelled from Atlas topics.",
                unique_id_field=id_field,
                is_public=is_public,
            )
            if indexed_field not in chunk.columns:
                raise ValueError(
                    f"`{indexed_field}` is not a column in the joined data. "
                    f"Available: {', '.join(list(chunk.columns)[:10])}…"
                )

        for start in range(0, len(chunk), UPLOAD_BATCH):
            dataset.add_data(chunk.iloc[start:start + UPLOAD_BATCH])
        sent += len(chunk)

        # Hand the chunk's arenas back before pulling the next one in. RSS is a
        # high-water mark, so without this the peak is set by the worst overlap
        # of two chunks rather than by one.
        del chunk
        gc.collect()

        frac = (sent / total_hint) if total_hint else 0.5
        _report(progress, 0.1 + 0.8 * min(1.0, frac), f"uploaded {sent:,} rows")

    if dataset is None:
        raise ValueError(
            "No rows matched the dataset. Check that the CSV is the one this "
            "Atlas dataset was built from."
        )

    _report(progress, 0.92, "building index (embedding + topic model)…")
    dataset.create_index(indexed_field=indexed_field, topic_model=True,
                         duplicate_detection=True)
    return dataset, sent


def map_link(dataset: AtlasDataset) -> str:
    return (dataset.maps[0].map_link if dataset.maps
            else f"https://atlas.nomic.ai/data/{dataset.identifier}")
