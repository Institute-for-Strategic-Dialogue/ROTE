# File: text_cluster.py
import csv
import io
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
from flask import Blueprint, render_template, request, send_file
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans

text_cluster_bp = Blueprint("text_cluster", __name__, template_folder="templates")

# Load once to keep responses fast; the MiniLM model is lightweight yet strong.
_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Hard guardrails
MAX_DOCS = 500_000
DEFAULT_THRESHOLD = 0.8
DEFAULT_MIN_CLUSTER_SIZE = 2


def _normalize_threshold(raw: str | None) -> float:
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_THRESHOLD
    return min(1.00, max(0.3, val))


def _normalize_min_cluster_size(raw: str | None) -> int:
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MIN_CLUSTER_SIZE
    return max(1, min(MAX_DOCS, val))


def _read_rows_from_upload(upload, text_column: str | None = None) -> tuple[list[str], list[dict], list[str], list[int]]:
    """
    Accept .txt (one per line) or .csv.
    *text_column* lets the caller pick which CSV column to cluster on.
    Falls back to a column named 'text', then the first column.
    """
    if not upload:
        return [], [], [], []

    filename = (getattr(upload, "filename", "") or "").lower()
    raw = upload.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")

    if filename.endswith(".csv"):
        buf = io.StringIO(raw, newline="")  # newline='' avoids csv new-line errors
        try:
            sample = buf.read(4096)
            buf.seek(0)
            dialect = csv.Sniffer().sniff(sample) if sample.strip() else csv.excel
        except csv.Error:
            buf.seek(0)
            dialect = csv.excel
        try:
            reader = csv.DictReader(buf, dialect=dialect)
            rows = []
            if reader.fieldnames:
                # Resolve which column holds the text to cluster
                if text_column and text_column in reader.fieldnames:
                    text_col = text_column
                else:
                    lowered = [c.lower() for c in reader.fieldnames]
                    text_col = reader.fieldnames[lowered.index("text")] if "text" in lowered else reader.fieldnames[0]
                buf.seek(0)
                reader = csv.DictReader(buf, dialect=dialect)
                for row in reader:
                    rows.append(row)
                texts = []
                row_indices = []
                for idx, row in enumerate(rows):
                    cell = (row.get(text_col) or "").strip()
                    if cell:
                        texts.append(cell)
                        row_indices.append(idx)
                return texts, rows, reader.fieldnames, row_indices
            return [], [], [], []
        except csv.Error:
            # Fallback: treat as plain text if CSV is malformed
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            rows = [{"text": ln} for ln in lines]
            return lines, rows, ["text"], list(range(len(rows)))

    # Treat everything else as newline-delimited plaintext
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    rows = [{"text": ln} for ln in lines]
    return lines, rows, ["text"], list(range(len(rows)))


def _read_rows_from_form(text_blob: str | None) -> tuple[list[str], list[dict], list[str], list[int]]:
    if not text_blob:
        return [], [], [], []
    lines = [ln.strip() for ln in text_blob.splitlines() if ln.strip()]
    rows = [{"text": ln} for ln in lines]
    return lines, rows, ["text"], list(range(len(rows)))


def _encode(texts: list[str]) -> np.ndarray:
    return _MODEL.encode(
        texts,
        batch_size=256,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


# ---------------------------------------------------------------------------
# Mode 1 – Near-duplicate detection  (greedy cosine-threshold clustering)
# ---------------------------------------------------------------------------

def _distinct_count(differentiators: list[str] | None, indices) -> int | None:
    """Return the number of distinct non-empty differentiator values across *indices*, or None."""
    if differentiators is None:
        return None
    return len({differentiators[i] for i in indices if differentiators[i]})


def _cluster_neardup(
    texts: list[str],
    embeddings: np.ndarray,
    threshold: float,
    min_cluster_size: int,
    differentiators: list[str] | None = None,
    min_distinct_differentiators: int = 1,
) -> tuple[list[dict], list[dict]]:
    n = len(texts)

    # Greedy clustering without an n×n matrix.
    # For each unvisited seed, compute its cosine similarity to every text
    # via a single vector-matrix multiply (O(n·d) per seed, O(n·d) memory).
    visited = np.zeros(n, dtype=bool)
    clusters = []
    for i in range(n):
        if visited[i]:
            continue
        sims = embeddings @ embeddings[i]          # (n,) cosine similarities
        mask = (sims >= threshold) & ~visited
        indices = np.where(mask)[0]
        visited[indices] = True
        members = []
        for idx in indices:
            m = {"text": texts[idx], "similarity": float(sims[idx])}
            if differentiators is not None:
                m["differentiator"] = differentiators[idx]
            members.append(m)
        clusters.append(
            {
                "id": None,
                "size": len(members),
                "members": members,
                "indices": indices.tolist(),
                "distinct_differentiators": _distinct_count(differentiators, indices),
            }
        )

    # Order clusters by size descending for readability
    clusters.sort(key=lambda c: c["size"], reverse=True)
    assignments = [
        {"cluster_id": "", "cluster_size": 1, "cluster_distinct_differentiators": ""}
        for _ in range(n)
    ]

    next_id = 1
    for cluster in clusters:
        meets_size = cluster["size"] >= min_cluster_size
        distinct = cluster["distinct_differentiators"]
        meets_diffs = differentiators is None or (distinct is not None and distinct >= int(min_distinct_differentiators))
        if meets_size and meets_diffs:
            cluster["id"] = next_id
            next_id += 1
        for text_idx in cluster["indices"]:
            assignments[text_idx] = {
                "cluster_id": cluster["id"] or "",
                "cluster_size": cluster["size"],
                "cluster_distinct_differentiators": distinct if distinct is not None else "",
            }

    display_clusters = [c for c in clusters if c["id"] is not None]
    return display_clusters, assignments


# ---------------------------------------------------------------------------
# Mode 2 – Semantic grouping  (MiniBatchKMeans on SBERT embeddings)
# ---------------------------------------------------------------------------

def _cluster_semantic(
    texts: list[str],
    embeddings: np.ndarray,
    n_clusters: int,
    min_cluster_size: int,
    differentiators: list[str] | None = None,
    min_distinct_differentiators: int = 1,
) -> tuple[list[dict], list[dict]]:
    n = len(texts)
    n_clusters = max(2, min(n_clusters, n))

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, batch_size=max(256, n_clusters), n_init=3, random_state=42
    )
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms == 0] = 1
    centroids = centroids / norms

    # Group texts by label
    groups: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for i, label in enumerate(labels):
        sim = float(embeddings[i] @ centroids[label])
        groups[int(label)].append((i, sim))

    # Build output sorted by cluster size
    sorted_labels = sorted(groups, key=lambda k: len(groups[k]), reverse=True)
    clusters = []
    assignments = [
        {"cluster_id": "", "cluster_size": 1, "cluster_distinct_differentiators": ""}
        for _ in range(n)
    ]

    next_id = 1
    for label in sorted_labels:
        members_data = sorted(groups[label], key=lambda t: t[1], reverse=True)
        size = len(members_data)
        indices = [idx for idx, _ in members_data]
        members = []
        for idx, sim in members_data:
            m = {"text": texts[idx], "similarity": sim}
            if differentiators is not None:
                m["differentiator"] = differentiators[idx]
            members.append(m)

        distinct = _distinct_count(differentiators, indices)
        meets_size = size >= min_cluster_size
        meets_diffs = differentiators is None or (distinct is not None and distinct >= int(min_distinct_differentiators))
        cluster_id = None
        if meets_size and meets_diffs:
            cluster_id = next_id
            next_id += 1
        clusters.append(
            {
                "id": cluster_id,
                "size": size,
                "members": members,
                "indices": indices,
                "distinct_differentiators": distinct,
            }
        )
        for idx, _ in members_data:
            assignments[idx] = {
                "cluster_id": cluster_id or "",
                "cluster_size": size,
                "cluster_distinct_differentiators": distinct if distinct is not None else "",
            }

    display_clusters = [c for c in clusters if c["id"] is not None]
    return display_clusters, assignments


def _cluster_within_partitions(
    cluster_fn,
    texts: list[str],
    embeddings: np.ndarray,
    partitions: list[str],
    differentiators: list[str] | None,
    min_distinct_differentiators: int,
) -> tuple[list[dict], list[dict]]:
    """
    Run *cluster_fn* independently inside each bucket of *partitions*.
    *cluster_fn* must accept (sub_texts, sub_embeddings, differentiators, min_distinct_differentiators)
    and return (display_clusters, assignments) in the same shape as _cluster_neardup / _cluster_semantic.
    Results are merged with globally-unique cluster IDs sorted by size.
    """
    n = len(texts)
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(partitions):
        buckets[p if p else "(empty)"].append(i)

    all_clusters: list[dict] = []
    assignments = [
        {"cluster_id": "", "cluster_size": 1, "cluster_distinct_differentiators": "", "cluster_partition": ""}
        for _ in range(n)
    ]
    tmp_next = 1
    for part_value, idx_list in buckets.items():
        if len(idx_list) < 2:
            # Still record the partition so the row shows up with its scope in the download.
            for global_i in idx_list:
                assignments[global_i]["cluster_partition"] = part_value
            continue
        sub_texts = [texts[i] for i in idx_list]
        sub_emb = embeddings[idx_list]
        sub_diffs = None
        if differentiators is not None:
            sub_diffs = [differentiators[i] for i in idx_list]
        sub_clusters, sub_assignments = cluster_fn(
            sub_texts, sub_emb,
            differentiators=sub_diffs,
            min_distinct_differentiators=min_distinct_differentiators,
        )

        # Temporary unique ids so we can remap at the end after a global sort.
        local_to_tmp: dict = {}
        for c in sub_clusters:
            local_to_tmp[c["id"]] = tmp_next
            c["id"] = tmp_next
            c["partition"] = part_value
            c["indices"] = [idx_list[i] for i in c["indices"]]
            all_clusters.append(c)
            tmp_next += 1

        for local_i, global_i in enumerate(idx_list):
            a = sub_assignments[local_i]
            mapped = local_to_tmp.get(a["cluster_id"], "") if a["cluster_id"] else ""
            assignments[global_i] = {
                "cluster_id": mapped,
                "cluster_size": a["cluster_size"],
                "cluster_distinct_differentiators": a.get("cluster_distinct_differentiators", ""),
                "cluster_partition": part_value,
            }

    # Final global sort-by-size and ID remap so the display order is stable.
    all_clusters.sort(key=lambda c: c["size"], reverse=True)
    tmp_to_final = {}
    for final_id, c in enumerate(all_clusters, 1):
        tmp_to_final[c["id"]] = final_id
        c["id"] = final_id
    for a in assignments:
        if a["cluster_id"] and a["cluster_id"] in tmp_to_final:
            a["cluster_id"] = tmp_to_final[a["cluster_id"]]

    return all_clusters, assignments


def _build_download_payload(
    rows: list[dict],
    fieldnames: list[str],
    assignments: list[dict],
    row_indices: list[int],
    min_cluster_size: int,
    include_distinct_differentiators: bool = False,
    include_partition: bool = False,
) -> tuple[list[dict], list[str]]:
    if not rows:
        return [], []
    cluster_ids = [""] * len(rows)
    cluster_sizes = [""] * len(rows)
    cluster_distinct = [""] * len(rows)
    cluster_partition = [""] * len(rows)

    for text_idx, row_idx in enumerate(row_indices):
        cluster_id = assignments[text_idx]["cluster_id"]
        cluster_size = assignments[text_idx]["cluster_size"]
        if cluster_size < min_cluster_size:
            cluster_id = ""
        cluster_ids[row_idx] = cluster_id
        cluster_sizes[row_idx] = cluster_size
        cluster_distinct[row_idx] = assignments[text_idx].get("cluster_distinct_differentiators", "")
        cluster_partition[row_idx] = assignments[text_idx].get("cluster_partition", "")

    output_rows = []
    for idx, row in enumerate(rows):
        row_out = {**row}
        row_out["cluster_id"] = cluster_ids[idx]
        row_out["cluster_size"] = cluster_sizes[idx]
        if include_distinct_differentiators:
            row_out["cluster_distinct_differentiators"] = cluster_distinct[idx]
        if include_partition:
            row_out["cluster_partition"] = cluster_partition[idx]
        output_rows.append(row_out)

    new_fields = list(fieldnames) + ["cluster_id", "cluster_size"]
    if include_distinct_differentiators:
        new_fields.append("cluster_distinct_differentiators")
    if include_partition:
        new_fields.append("cluster_partition")
    return output_rows, new_fields


@text_cluster_bp.route("/", methods=["GET", "POST"])
def text_cluster():
    clusters: list[dict] = []
    threshold = DEFAULT_THRESHOLD
    min_cluster_size = DEFAULT_MIN_CLUSTER_SIZE
    total = 0

    cluster_method = "neardup"
    n_clusters_target = 0
    differentiator_col_used = ""
    min_distinct_diffs = 1
    partition_col_used = ""

    if request.method == "POST":
        cluster_method = request.form.get("cluster_method", "neardup")
        threshold = _normalize_threshold(request.form.get("threshold"))
        min_cluster_size = _normalize_min_cluster_size(request.form.get("min_cluster_size"))
        output_mode = request.form.get("output_mode", "table")

        text_column = (request.form.get("text_column") or "").strip() or None
        differentiator_col = (request.form.get("differentiator_col") or "").strip() or None
        partition_col = (request.form.get("partition_col") or "").strip() or None
        try:
            min_distinct_diffs = max(1, int(request.form.get("min_distinct_differentiators") or 1))
        except (TypeError, ValueError):
            min_distinct_diffs = 1

        pasted = _read_rows_from_form(request.form.get("texts"))
        uploaded = _read_rows_from_upload(request.files.get("text_file"), text_column=text_column)

        texts, rows, fieldnames, row_indices = uploaded if uploaded[0] else pasted
        if len(texts) > MAX_DOCS:
            texts = texts[:MAX_DOCS]
            row_indices = row_indices[:MAX_DOCS]
        total = len(texts)
        assignments = []

        # Align differentiator values to the filtered text list (row_indices map back to rows).
        differentiators: list[str] | None = None
        if differentiator_col and rows and differentiator_col in (fieldnames or []):
            differentiators = [
                (rows[row_idx].get(differentiator_col) or "").strip()
                for row_idx in row_indices
            ]
            differentiator_col_used = differentiator_col

        # Same alignment for partition values.
        partitions: list[str] | None = None
        if partition_col and rows and partition_col in (fieldnames or []):
            partitions = [
                (rows[row_idx].get(partition_col) or "").strip()
                for row_idx in row_indices
            ]
            partition_col_used = partition_col

        if total > 0:
            embeddings = _encode(texts)

            raw_k = request.form.get("n_clusters", "")
            try:
                n_clusters_target = max(2, int(raw_k))
            except (TypeError, ValueError):
                n_clusters_target = max(2, int(total ** 0.5))

            def _run_neardup(t, e, differentiators=None, min_distinct_differentiators=1):
                return _cluster_neardup(
                    t, e, threshold, min_cluster_size,
                    differentiators=differentiators,
                    min_distinct_differentiators=min_distinct_differentiators,
                )

            def _run_semantic(t, e, differentiators=None, min_distinct_differentiators=1):
                # In partitioned mode each bucket auto-picks its own k by sqrt(bucket size).
                k = max(2, int(len(t) ** 0.5)) if partitions is not None else n_clusters_target
                return _cluster_semantic(
                    t, e, k, min_cluster_size,
                    differentiators=differentiators,
                    min_distinct_differentiators=min_distinct_differentiators,
                )

            cluster_fn = _run_semantic if cluster_method == "semantic" else _run_neardup

            if partitions is not None:
                clusters, assignments = _cluster_within_partitions(
                    cluster_fn, texts, embeddings, partitions,
                    differentiators=differentiators,
                    min_distinct_differentiators=min_distinct_diffs,
                )
            else:
                clusters, assignments = cluster_fn(
                    texts, embeddings,
                    differentiators=differentiators,
                    min_distinct_differentiators=min_distinct_diffs,
                )

        if output_mode in {"csv", "excel"} and rows:
            output_rows, output_fields = _build_download_payload(
                rows, fieldnames, assignments, row_indices, min_cluster_size,
                include_distinct_differentiators=(differentiators is not None),
                include_partition=(partitions is not None),
            )
            if output_mode == "csv":
                output = io.StringIO(newline="")
                writer = csv.DictWriter(output, fieldnames=output_fields, extrasaction="ignore")
                writer.writeheader()
                for row in output_rows:
                    writer.writerow(row)
                mem = io.BytesIO(output.getvalue().encode("utf-8"))
                return send_file(
                    mem,
                    mimetype="text/csv",
                    as_attachment=True,
                    download_name="text_clusters.csv",
                )

            output = io.BytesIO()
            df = pd.DataFrame(output_rows, columns=output_fields)
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df.to_excel(writer, sheet_name="clusters", index=False)
            output.seek(0)
            return send_file(
                output,
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                as_attachment=True,
                download_name="text_clusters.xlsx",
            )

    return render_template(
        "text_cluster.html",
        clusters=clusters,
        threshold=threshold,
        min_cluster_size=min_cluster_size,
        total=total,
        max_docs=MAX_DOCS,
        cluster_method=cluster_method,
        n_clusters_target=n_clusters_target,
        differentiator_col_used=differentiator_col_used,
        min_distinct_differentiators=min_distinct_diffs,
        partition_col_used=partition_col_used,
    )
