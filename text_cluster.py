# File: text_cluster.py
import csv
import io
from typing import Iterable

import numpy as np
from flask import Blueprint, render_template, request
from sentence_transformers import SentenceTransformer

text_cluster_bp = Blueprint("text_cluster", __name__, template_folder="templates")

# Load once to keep responses fast; the MiniLM model is lightweight yet strong.
_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Hard guardrails to keep latency reasonable
MAX_DOCS = 5000
DEFAULT_THRESHOLD = 0.8


def _normalize_threshold(raw: str | None) -> float:
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_THRESHOLD
    return min(0.95, max(0.3, val))


def _read_texts_from_upload(upload) -> list[str]:
    """
    Accept .txt (one per line) or .csv (uses 'text' column if present, otherwise first column).
    """
    if not upload:
        return []

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
            texts = []
            if reader.fieldnames:
                lowered = [c.lower() for c in reader.fieldnames]
                text_col = reader.fieldnames[lowered.index("text")] if "text" in lowered else reader.fieldnames[0]
                buf.seek(0)
                reader = csv.DictReader(buf, dialect=dialect)
                for row in reader:
                    cell = (row.get(text_col) or "").strip()
                    if cell:
                        texts.append(cell)
            return texts
        except csv.Error:
            # Fallback: treat as plain text if CSV is malformed
            return [ln.strip() for ln in raw.splitlines() if ln.strip()]

    # Treat everything else as newline-delimited plaintext
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]


def _read_texts_from_form(text_blob: str | None) -> list[str]:
    if not text_blob:
        return []
    return [ln.strip() for ln in text_blob.splitlines() if ln.strip()]


def _cluster_texts(texts: Iterable[str], threshold: float) -> list[dict]:
    texts = [t.strip() for t in texts if t and t.strip()]
    if not texts:
        return []
    texts = texts[:MAX_DOCS]

    embeddings = _MODEL.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # Cosine similarity via dot product of normalized embeddings
    sim_matrix = np.matmul(embeddings, embeddings.T)

    visited = set()
    clusters = []
    for i in range(len(texts)):
        if i in visited:
            continue
        mask = sim_matrix[i] >= threshold
        indices = [idx for idx, ok in enumerate(mask) if ok and idx not in visited]
        for idx in indices:
            visited.add(idx)
        members = [
            {"text": texts[idx], "similarity": float(sim_matrix[i, idx])}
            for idx in indices
        ]
        clusters.append(
            {
                "id": len(clusters) + 1,
                "size": len(members),
                "members": members,
            }
        )

    # Order clusters by size descending for readability
    clusters.sort(key=lambda c: c["size"], reverse=True)
    return clusters


@text_cluster_bp.route("/", methods=["GET", "POST"])
def text_cluster():
    clusters: list[dict] = []
    threshold = DEFAULT_THRESHOLD
    total = 0

    if request.method == "POST":
        threshold = _normalize_threshold(request.form.get("threshold"))

        pasted = _read_texts_from_form(request.form.get("texts"))
        uploaded = _read_texts_from_upload(request.files.get("text_file"))

        texts = uploaded or pasted
        total = len(texts)
        clusters = _cluster_texts(texts, threshold)

    return render_template(
        "text_cluster.html",
        clusters=clusters,
        threshold=threshold,
        total=total,
        max_docs=MAX_DOCS,
    )
