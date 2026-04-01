# File: bw_grouping.py
import csv
import io
import re
from collections import Counter

import pandas as pd
from flask import Blueprint, render_template, request, send_file

bw_grouping_bp = Blueprint("bw_grouping", __name__, template_folder="templates")

# Columns that contain 0-n comma+space separated values
MULTI_VALUE_COLUMNS = [
    "Expanded URLs",
    "Media URLs",
    "X Repost of",
    "X Reply to",
]

# English stopwords for token filtering
STOPWORDS = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o",
    "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
    "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan",
    "shouldn", "wasn", "weren", "won", "wouldn", "also", "would",
    "could", "might", "shall", "may", "must", "need", "get", "got",
    "one", "two", "like", "even", "still", "us", "much", "many",
    "well", "really", "right", "back", "going", "know", "think",
    "want", "go", "make", "see", "say", "said", "come", "take",
    "people", "way", "thing", "things", "time", "new", "good", "first",
    "last", "long", "great", "little", "just", "man", "made", "let",
    "put", "old", "big", "use", "look", "give", "help", "tell", "ask",
    "try", "keep", "work", "part", "set", "call", "every", "find",
    "day", "run", "end",
    # single-char noise
    "b", "c", "e", "f", "g", "h", "j", "k", "l", "n", "p", "q", "r",
    "u", "v", "w", "x", "z",
    # common web/social noise
    "http", "https", "www", "com", "rt", "amp",
})

_TOKEN_RE = re.compile(r"[a-zA-Z0-9#@][\w'@#]*")

MAX_ROWS = 100_000


def _read_upload(upload) -> pd.DataFrame:
    """Read CSV or Excel upload into a DataFrame."""
    if not upload:
        return pd.DataFrame()
    filename = (getattr(upload, "filename", "") or "").lower()
    raw = upload.read()
    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw), dtype=str, nrows=MAX_ROWS).fillna("")
    # default: CSV
    return pd.read_csv(
        io.BytesIO(raw), dtype=str, nrows=MAX_ROWS, encoding_errors="ignore"
    ).fillna("")


def _split_values(series: pd.Series) -> list[str]:
    """Split a column of comma+space separated values into a flat list."""
    values = []
    for cell in series:
        cell = str(cell).strip()
        if not cell:
            continue
        values.extend(v.strip() for v in cell.split(", ") if v.strip())
    return values


def _tokenize_texts(series: pd.Series) -> list[str]:
    """Tokenize Full Text column, lowercased and stopword-filtered."""
    tokens = []
    for cell in series:
        cell = str(cell).strip()
        if not cell:
            continue
        for tok in _TOKEN_RE.findall(cell):
            tok_lower = tok.lower()
            if tok_lower not in STOPWORDS and len(tok_lower) > 1:
                tokens.append(tok_lower)
    return tokens


def _aggregate(df: pd.DataFrame) -> dict:
    """Return {dimension_name: Counter} for each multi-value column + tokens."""
    results = {}
    for col in MULTI_VALUE_COLUMNS:
        if col in df.columns:
            vals = _split_values(df[col])
            if vals:
                results[col] = Counter(vals)
    # Tokens from Full Text
    if "Full Text" in df.columns:
        tokens = _tokenize_texts(df["Full Text"])
        if tokens:
            results["Tokens (from Full Text)"] = Counter(tokens)
    return results


def _results_to_download(results: dict, fmt: str):
    """Build a multi-sheet Excel or multi-section CSV from results."""
    if fmt == "excel":
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            for dim, counter in results.items():
                sheet = dim[:31]  # Excel sheet name limit
                rows = [{"Value": val, "Mentions": cnt}
                        for val, cnt in counter.most_common()]
                pd.DataFrame(rows).to_excel(writer, sheet_name=sheet, index=False)
        buf.seek(0)
        return send_file(
            buf,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name="bw_grouping.xlsx",
        )
    # CSV: concatenate all dimensions with a header row per section
    buf = io.StringIO(newline="")
    writer = csv.writer(buf)
    for dim, counter in results.items():
        writer.writerow([dim])
        writer.writerow(["Value", "Mentions"])
        for val, cnt in counter.most_common():
            writer.writerow([val, cnt])
        writer.writerow([])  # blank separator
    mem = io.BytesIO(buf.getvalue().encode("utf-8"))
    return send_file(
        mem, mimetype="text/csv", as_attachment=True,
        download_name="bw_grouping.csv",
    )


@bw_grouping_bp.route("/", methods=["GET", "POST"])
def bw_grouping():
    results = {}
    total_rows = 0

    if request.method == "POST":
        df = _read_upload(request.files.get("bw_file"))
        if df.empty:
            return render_template("bw_grouping.html", results=results,
                                   total_rows=0, error="No file uploaded or file was empty.")

        total_rows = len(df)
        results = _aggregate(df)
        output_mode = request.form.get("output_mode", "table")

        if output_mode in {"csv", "excel"} and results:
            return _results_to_download(results, output_mode)

    return render_template(
        "bw_grouping.html",
        results=results,
        total_rows=total_rows,
    )
