"""Nomic topic relabeller.

Two steps:
  1. Point at an Atlas dataset, inspect its topic depths, export a workbook of
     sampled posts plus the labels Nomic generated.
  2. Feed the edited workbook back to build a new dataset carrying your labels.

API keys are taken from the form and used only for the life of the job — see
nomic_io.authenticate(), which deliberately never calls nomic.login().

Long work runs on background threads (jobs.py) because Cloudflare cuts off any
response over ~100 seconds.
"""

from __future__ import annotations

import gc
import io
import os
import tempfile

import pandas as pd
from flask import (Blueprint, jsonify, render_template, request, send_file)

from . import jobs, nomic_io, xlsx_io

nomic_relabel_bp = Blueprint("nomic_relabel", __name__, template_folder="templates")

# Pulling source rows back from Atlas materialises the whole dataset in one go
# (the client has no chunked read), so it is capped. Beyond this, the CSV path
# is the only safe option on a memory-constrained host.
MAX_ATLAS_PULL_ROWS = 50_000

# Topics for the most recently inspected dataset, so step 1's export does not
# refetch. Deliberately one entry: this is a memory budget, not a cache.
_topics_cache: dict = {}


def _cache_topics(identifier: str, payload) -> None:
    _topics_cache.clear()
    _topics_cache[identifier] = payload


def _load_topics(dataset, progress=None):
    """Topics for a dataset, reusing the single cache slot when possible.

    -> (topics_df, metadata, id_field, spec). `spec` says where the id came
    from: the dataset's own unique_id_field when it declared one, otherwise a
    column recovered by position (see nomic_io.IdSpec).
    """
    cached = _topics_cache.get(dataset.identifier)
    if cached is not None:
        return cached
    if progress:
        progress(0.1, "resolving the dataset's id field…")
    spec = nomic_io.resolve_id_spec(dataset)
    if progress:
        progress(0.15, "loading topics from Atlas…")
    topics_df, metadata, id_field = nomic_io.load_topics(dataset, spec)
    payload = (topics_df, metadata, id_field, spec)
    _cache_topics(dataset.identifier, payload)
    return payload


def _form_connection():
    api_key = (request.form.get("api_key") or os.environ.get("NOMIC_API_KEY") or "").strip()
    identifier = (request.form.get("identifier") or "").strip()
    if not api_key:
        raise ValueError("Missing Nomic API key. Provide it in the form or set NOMIC_API_KEY.")
    if not identifier:
        raise ValueError("Missing dataset identifier.")
    return api_key, identifier


@nomic_relabel_bp.route("/", methods=["GET"])
def index():
    return render_template("nomic_relabel.html", max_atlas_pull=MAX_ATLAS_PULL_ROWS)


# ── Step 1a: inspect ──────────────────────────────────────────────────


@nomic_relabel_bp.route("/inspect", methods=["POST"])
def inspect():
    try:
        api_key, identifier = _form_connection()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    def work(progress):
        nomic_io.authenticate(api_key)
        dataset = nomic_io.open_dataset(identifier)
        topics_df, metadata, id_field, spec = _load_topics(dataset, progress)
        progress(0.8, "summarising depths…")
        summary = nomic_io.depth_summary(topics_df, metadata)
        return {
            "identifier": dataset.identifier,
            "rows": int(dataset.total_datums),
            "id_field": id_field,
            "id_note": spec.note,
            "id_native": bool(spec.native),
            "fields": [f for f in dataset.dataset_fields if f != id_field],
            "depths": summary.to_dict(orient="records"),
        }

    return jsonify({"job_id": jobs.start(work, "inspect")})


# ── Step 1b: export workbook ──────────────────────────────────────────


@nomic_relabel_bp.route("/export", methods=["POST"])
def export():
    try:
        api_key, identifier = _form_connection()
        depth = int(request.form.get("depth", 1))
        n_per_topic = max(1, min(50, int(request.form.get("n_per_topic", 10))))
        text_field = (request.form.get("text_field") or "").strip()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    def work(progress):
        nomic_io.authenticate(api_key)
        dataset = nomic_io.open_dataset(identifier)
        topics_df, metadata, id_field, spec = _load_topics(dataset, progress)

        progress(0.3, "sampling posts…")
        sampled = nomic_io.sample_ids(topics_df, id_field, depth, n_per_topic)
        report = lambda f, m: progress(0.35 + 0.5 * f, m)
        if spec.native:
            rows = nomic_io.fetch_rows(dataset, sampled[id_field], progress=report)
        else:
            # No declared id field means no datum ids, so get_data() is not
            # available — the sampled rows have to come out of the map data.
            # Only the text column reaches the workbook now, so only it is
            # worth materialising — this path pulls every row of whatever it
            # asks for.
            wanted = [text_field, "full_text"]
            rows = nomic_io.fetch_rows_from_map(
                dataset, spec, sampled[id_field],
                [c for c in wanted if c in dataset.dataset_fields],
                progress=report)
        posts = sampled.merge(rows, on=id_field, how="left")

        progress(0.9, "building workbook…")
        labels = nomic_io.label_table(topics_df, metadata, depth)
        field = text_field if text_field in posts.columns else (
            "full_text" if "full_text" in posts.columns else posts.columns[-1])
        data = xlsx_io.build_workbook(labels, posts, id_field, field)

        return {
            "bytes": data,
            "filename": f"{dataset.slug}_depth{depth}_labels.xlsx",
            "topics": int(len(labels)),
            "sampled_posts": int(len(posts)),
        }

    return jsonify({"job_id": jobs.start(work, "export")})


# ── Step 2: remap and upload ──────────────────────────────────────────


@nomic_relabel_bp.route("/relabel", methods=["POST"])
def relabel():
    try:
        api_key, identifier = _form_connection()
        depth = int(request.form.get("depth_2", 1))
        new_name = (request.form.get("new_name") or "").strip()
        text_field = (request.form.get("text_field_2") or "").strip()
        source = (request.form.get("source") or "csv").strip()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if not new_name:
        return jsonify({"error": "Missing a name for the new dataset."}), 400

    workbook = request.files.get("workbook")
    if not workbook or not workbook.filename:
        return jsonify({"error": "Upload the edited workbook (.xlsx)."}), 400
    try:
        labels = xlsx_io.read_labels(io.BytesIO(workbook.read()))
        specific_map, broad_map, n_blank, n_blank_broad = xlsx_io.build_mapping(labels)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # Spool the CSV to disk rather than holding it in memory: a 275 MB upload
    # read into pandas whole peaks ~1.6 GB, which would OOM the whole site.
    csv_path = None
    if source == "csv":
        csv_file = request.files.get("csv")
        if not csv_file or not csv_file.filename:
            return jsonify({"error": "Upload the original source CSV, or choose "
                                     "to pull rows from Atlas instead."}), 400
        fd, csv_path = tempfile.mkstemp(suffix=".csv", prefix="nomic_relabel_")
        os.close(fd)
        csv_file.save(csv_path)

    def work(progress):
        try:
            nomic_io.authenticate(api_key)
            dataset = nomic_io.open_dataset(identifier)
            topics_df, _metadata, id_field, spec = _load_topics(dataset, progress)

            depth_col = nomic_io.DEPTH_COLS[depth - 1]
            present = set(topics_df[depth_col].dropna().unique())
            unmapped = present - set(specific_map)
            if unmapped:
                raise ValueError(
                    f"{len(unmapped)} topic(s) in the dataset have no row in the "
                    f"workbook — is this the right depth? Examples: "
                    f"{', '.join(list(unmapped)[:5])}"
                )

            total = int(dataset.total_datums)
            lookup = nomic_io.topic_lookup(topics_df, id_field, depth)

            # The full topics frame is ~78 MB on a 175k-row dataset and is dead
            # weight once `lookup` exists. Drop it (and the cache slot holding
            # it) before the upload loop, so it isn't competing with chunks.
            del topics_df, _metadata, present, unmapped
            _topics_cache.clear()
            gc.collect()

            if source == "csv":
                chunks = nomic_io.iter_csv_labelled(
                    csv_path, lookup, spec, specific_map, broad_map)
            else:
                if total > MAX_ATLAS_PULL_ROWS:
                    raise ValueError(
                        f"This dataset has {total:,} rows. Pulling them back from "
                        f"Atlas loads them all at once, which is unsafe above "
                        f"{MAX_ATLAS_PULL_ROWS:,} on this host — upload the "
                        f"original CSV instead."
                    )
                chunks = nomic_io.iter_atlas_labelled(
                    dataset, lookup, spec, specific_map, broad_map)

            progress(0.1, f"uploading up to {total:,} rows…")
            new_ds, sent = nomic_io.upload_chunks(
                chunks, new_name, id_field, text_field or "full_text",
                description=f"Relabelled from {dataset.identifier} at depth {depth}.",
                is_public=False, total_hint=total, progress=progress)

            return {
                "identifier": new_ds.identifier,
                "rows": int(sent),
                "map_link": nomic_io.map_link(new_ds),
                "blank_specific": n_blank,
                "blank_broad": n_blank_broad,
            }
        finally:
            if csv_path and os.path.exists(csv_path):
                os.unlink(csv_path)

    return jsonify({"job_id": jobs.start(work, "relabel")})


# ── Job polling ───────────────────────────────────────────────────────


@nomic_relabel_bp.route("/job/<job_id>", methods=["GET"])
def job_status(job_id):
    status = jobs.public_status(job_id)
    if status is None:
        return jsonify({"error": "Unknown job — it may have expired."}), 404
    return jsonify(status)


@nomic_relabel_bp.route("/job/<job_id>/download", methods=["GET"])
def job_download(job_id):
    job = jobs.get(job_id)
    if not job or job["state"] != "done" or not isinstance(job.get("result"), dict):
        return jsonify({"error": "No file for that job."}), 404
    result = job["result"]
    if "bytes" not in result:
        return jsonify({"error": "That job produced no file."}), 404
    return send_file(
        io.BytesIO(result["bytes"]),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=result.get("filename", "labels.xlsx"),
    )
