# File: image_match.py
import io
import zipfile

from collections import defaultdict

import imagehash
import pandas as pd
import requests
from PIL import Image
from flask import Blueprint, render_template, request, send_file

image_match_bp = Blueprint("image_match", __name__, template_folder="templates")

MAX_IMAGES = 20000
DEFAULT_THRESHOLD = 0  # 0 = exact match only
HASH_SIZE = 16  # higher = more discriminating (default lib uses 8)
FETCH_TIMEOUT = 15  # seconds per URL


def _extract_urls_from_file(upload) -> list[str]:
    """Extract image URLs from an uploaded CSV or TXT file.

    CSV: uses a column named 'url', 'image_url', 'media_url', 'media urls',
    'expanded urls', or 'link' (case-insensitive); falls back to the first column.
    TXT: one URL per line.
    """
    if not upload:
        return []
    filename = (getattr(upload, "filename", "") or "").lower()
    raw = upload.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    if not raw.strip():
        return []

    url_col_names = {
        "url", "urls", "image_url", "image_urls", "media_url", "media_urls",
        "media urls", "expanded urls", "expanded_urls", "link", "links",
        "image", "src",
    }

    if filename.endswith((".csv", ".xlsx", ".xls")):
        try:
            buf = io.StringIO(raw, newline="")
            import csv as _csv
            reader = _csv.DictReader(buf)
            if not reader.fieldnames:
                return []
            # Find the best URL column
            lowered = {c.lower().strip(): c for c in reader.fieldnames}
            col = None
            for candidate in url_col_names:
                if candidate in lowered:
                    col = lowered[candidate]
                    break
            if col is None:
                col = reader.fieldnames[0]
            urls = []
            buf.seek(0)
            reader = _csv.DictReader(buf)
            for row in reader:
                val = (row.get(col) or "").strip()
                if val.startswith(("http://", "https://")):
                    urls.append(val)
            return urls
        except Exception:
            pass

    # Fallback: treat as plain text, one URL per line
    return [
        line.strip() for line in raw.splitlines()
        if line.strip().startswith(("http://", "https://"))
    ]


def _load_images_from_urls(url_text: str) -> list[tuple[str, Image.Image]]:
    """Download images from a newline-separated list of URLs."""
    images = []
    if not url_text or not url_text.strip():
        return images
    urls = [
        line.strip() for line in url_text.splitlines()
        if line.strip().startswith(("http://", "https://"))
    ]
    return _fetch_urls(urls)


def _fetch_urls(urls: list[str]) -> list[tuple[str, Image.Image]]:
    """Download and open images from a list of URLs."""
    images = []
    for url in urls:
        try:
            resp = requests.get(url, timeout=FETCH_TIMEOUT, stream=True)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content))
            img.load()
            images.append((url, img))
        except Exception:
            continue
        if len(images) >= MAX_IMAGES:
            break
    return images


def _load_images_from_files(file_list) -> list[tuple[str, Image.Image]]:
    """Load images from a list of uploaded FileStorage objects."""
    images = []
    for f in file_list:
        fname = f.filename or ""
        if not fname:
            continue
        try:
            if fname.lower().endswith(".zip"):
                images.extend(_load_images_from_zip(f))
            else:
                img = Image.open(f.stream)
                img.load()
                images.append((fname, img))
        except Exception:
            continue
        if len(images) >= MAX_IMAGES:
            break
    return images[:MAX_IMAGES]


def _load_images_from_zip(f) -> list[tuple[str, Image.Image]]:
    """Extract images from a zip archive."""
    images = []
    try:
        raw = f.read() if hasattr(f, "read") else f
        zf = zipfile.ZipFile(io.BytesIO(raw) if isinstance(raw, bytes) else raw)
        for name in zf.namelist():
            low = name.lower()
            if low.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff")):
                try:
                    img = Image.open(io.BytesIO(zf.read(name)))
                    img.load()
                    images.append((name, img))
                except Exception:
                    continue
    except Exception:
        pass
    return images


def _compute_hashes(
    images: list[tuple[str, Image.Image]],
) -> list[tuple[str, imagehash.ImageHash]]:
    """Compute perceptual hash for each image."""
    results = []
    for name, img in images:
        try:
            h = imagehash.phash(img, hash_size=HASH_SIZE)
            results.append((name, h))
        except Exception:
            continue
    return results


def _hash_to_bits(h: imagehash.ImageHash) -> bytes:
    """Return the flat bit array of a hash as a bytes object."""
    return h.hash.flatten().tobytes()


def _band_keys(bits: bytes, band_size: int) -> list[tuple[int, bytes]]:
    """Split a bit-string into bands and return (band_index, band_bytes) keys."""
    n_bands = len(bits) // band_size
    return [
        (b, bits[b * band_size : (b + 1) * band_size])
        for b in range(n_bands)
    ]


# Choose a band size that gives good recall for typical thresholds (0-20).
# With HASH_SIZE=16 we get 256 bits.  16-byte bands → 16 bands.
# Two hashes differing in ≤10 bits share ≥1 band with very high probability.
_LSH_BAND_SIZE = 16


def _find_matches_ingroup(
    hashes: list[tuple[str, imagehash.ImageHash]],
    threshold: int,
) -> list[dict]:
    """Find matching pairs within one set, using LSH to avoid O(n²)."""
    n = len(hashes)

    # Fast path: exact matches via dict grouping — O(n)
    if threshold == 0:
        groups = defaultdict(list)
        for idx, (name, h) in enumerate(hashes):
            groups[str(h)].append((idx, name))
        matches = []
        for members in groups.values():
            if len(members) < 2:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    matches.append({
                        "image_a": members[i][1],
                        "image_b": members[j][1],
                        "distance": 0,
                        "exact": True,
                    })
        return matches

    # Threshold > 0: band-based LSH to find candidate pairs
    bits = [_hash_to_bits(h) for _, h in hashes]

    # Build buckets: band_key → set of indices
    buckets = defaultdict(list)
    for idx in range(n):
        for key in _band_keys(bits[idx], _LSH_BAND_SIZE):
            buckets[key].append(idx)

    # Collect unique candidate pairs from shared buckets
    candidates = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                candidates.add((min(a, b), max(a, b)))

    # Verify candidates with actual hash distance
    matches = []
    for i, j in candidates:
        dist = hashes[i][1] - hashes[j][1]
        if dist <= threshold:
            matches.append({
                "image_a": hashes[i][0],
                "image_b": hashes[j][0],
                "distance": dist,
                "exact": dist == 0,
            })
    matches.sort(key=lambda m: m["distance"])
    return matches


def _find_matches_source_vs_dataset(
    source_hashes: list[tuple[str, imagehash.ImageHash]],
    dataset_hashes: list[tuple[str, imagehash.ImageHash]],
    threshold: int,
) -> list[dict]:
    """Match source images against dataset, using LSH to avoid O(n*m)."""

    # Fast path: exact matches via dict lookup — O(n+m)
    if threshold == 0:
        ds_index = defaultdict(list)
        for name, h in dataset_hashes:
            ds_index[str(h)].append(name)
        matches = []
        for name_s, hash_s in source_hashes:
            for name_d in ds_index.get(str(hash_s), []):
                matches.append({
                    "source_image": name_s,
                    "dataset_image": name_d,
                    "distance": 0,
                    "exact": True,
                })
        return matches

    # Threshold > 0: index dataset bands, probe with source
    ds_bits = [_hash_to_bits(h) for _, h in dataset_hashes]
    ds_buckets = defaultdict(list)
    for idx in range(len(dataset_hashes)):
        for key in _band_keys(ds_bits[idx], _LSH_BAND_SIZE):
            ds_buckets[key].append(idx)

    matches = []
    for name_s, hash_s in source_hashes:
        src_bits = _hash_to_bits(hash_s)
        # Collect candidate dataset indices that share any band
        candidate_ds = set()
        for key in _band_keys(src_bits, _LSH_BAND_SIZE):
            for ds_idx in ds_buckets.get(key, []):
                candidate_ds.add(ds_idx)
        # Verify
        for ds_idx in candidate_ds:
            dist = hash_s - dataset_hashes[ds_idx][1]
            if dist <= threshold:
                matches.append({
                    "source_image": name_s,
                    "dataset_image": dataset_hashes[ds_idx][0],
                    "distance": dist,
                    "exact": dist == 0,
                })
    matches.sort(key=lambda m: m["distance"])
    return matches


def _matches_to_download(matches: list[dict], mode: str, fmt: str):
    """Build downloadable CSV or Excel from match results."""
    df = pd.DataFrame(matches)
    if df.empty:
        df = pd.DataFrame(columns=["No matches found"])

    if fmt == "excel":
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="matches", index=False)
        buf.seek(0)
        return send_file(
            buf,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name="image_matches.xlsx",
        )
    buf = io.StringIO(newline="")
    df.to_csv(buf, index=False)
    mem = io.BytesIO(buf.getvalue().encode("utf-8"))
    return send_file(
        mem, mimetype="text/csv", as_attachment=True,
        download_name="image_matches.csv",
    )


@image_match_bp.route("/", methods=["GET", "POST"])
def image_match():
    matches = []
    total_images = 0
    match_mode = "ingroup"
    threshold = DEFAULT_THRESHOLD
    error = None

    if request.method == "POST":
        match_mode = request.form.get("match_mode", "ingroup")
        output_mode = request.form.get("output_mode", "table")
        try:
            threshold = max(0, int(request.form.get("threshold", DEFAULT_THRESHOLD)))
        except (TypeError, ValueError):
            threshold = DEFAULT_THRESHOLD

        if match_mode == "source_vs_dataset":
            source_files = request.files.getlist("source_images")
            dataset_files = request.files.getlist("dataset_images")
            source_imgs = _load_images_from_files(source_files)
            source_imgs.extend(_load_images_from_urls(request.form.get("source_urls", "")))
            source_imgs.extend(_fetch_urls(_extract_urls_from_file(request.files.get("source_url_file"))))
            dataset_imgs = _load_images_from_files(dataset_files)
            dataset_imgs.extend(_load_images_from_urls(request.form.get("dataset_urls", "")))
            dataset_imgs.extend(_fetch_urls(_extract_urls_from_file(request.files.get("dataset_url_file"))))

            if not source_imgs:
                error = "No source images provided (upload files, paste URLs, or upload a URL list)."
            elif not dataset_imgs:
                error = "No dataset images provided (upload files, paste URLs, or upload a URL list)."
            else:
                source_hashes = _compute_hashes(source_imgs[:MAX_IMAGES])
                dataset_hashes = _compute_hashes(dataset_imgs[:MAX_IMAGES])
                total_images = len(source_hashes) + len(dataset_hashes)
                matches = _find_matches_source_vs_dataset(
                    source_hashes, dataset_hashes, threshold
                )
        else:  # ingroup
            all_files = request.files.getlist("ingroup_images")
            all_imgs = _load_images_from_files(all_files)
            all_imgs.extend(_load_images_from_urls(request.form.get("ingroup_urls", "")))
            all_imgs.extend(_fetch_urls(_extract_urls_from_file(request.files.get("ingroup_url_file"))))
            if not all_imgs:
                error = "No images provided (upload files, paste URLs, or upload a URL list)."
            else:
                hashes = _compute_hashes(all_imgs)
                total_images = len(hashes)
                matches = _find_matches_ingroup(hashes, threshold)

        if output_mode in {"csv", "excel"} and not error:
            return _matches_to_download(matches, match_mode, output_mode)

    return render_template(
        "image_match.html",
        matches=matches,
        total_images=total_images,
        match_mode=match_mode,
        threshold=threshold,
        max_images=MAX_IMAGES,
        error=error,
    )
