"""Build the review workbook and parse the edited one back."""

from __future__ import annotations

import io

import pandas as pd

LABELS_SHEET = "labels"
POSTS_SHEET = "sampled_posts"

# Columns the analyst fills in, and the Atlas columns they become.
EDITABLE_COLS = ["specific_label", "broad_label"]
ATLAS_SPECIFIC = "Topic (Specific)"
ATLAS_BROAD = "Topic (Broad)"

# Context columns worth showing next to the text when judging a topic label.
# Only those actually present in the dataset are used.
PREFERRED_CONTEXT = [
    "date", "datetime", "platform", "author", "post_type", "sentiment",
    "total_engagement", "views", "url", "domain",
]


def build_workbook(labels: pd.DataFrame, posts: pd.DataFrame,
                   id_field: str, text_field: str) -> bytes:
    """-> xlsx bytes with two tabs: sampled_posts and labels."""
    cols = ["topic", id_field, text_field]
    cols += [c for c in PREFERRED_CONTEXT if c in posts.columns and c not in cols]
    posts_out = posts[[c for c in cols if c in posts.columns]].copy()

    # Order topics exactly as they appear in the labels sheet (largest first) so
    # the two tabs can be read side by side. Within a topic, keep the order the
    # sampler produced — random, but deterministic for a given seed.
    order = labels["nomic_label"].astype(str).tolist()
    posts_out["topic"] = pd.Categorical(posts_out["topic"].astype(str),
                                        categories=order, ordered=True)
    posts_out = (posts_out.sort_values("topic", kind="stable")
                 .reset_index(drop=True))
    posts_out["topic"] = posts_out["topic"].astype(str)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter", datetime_format="yyyy-mm-dd hh:mm") as writer:
        posts_out.to_excel(writer, sheet_name=POSTS_SHEET, index=False)
        labels.to_excel(writer, sheet_name=LABELS_SHEET, index=False)

        wb = writer.book
        wrap = wb.add_format({"text_wrap": True, "valign": "top"})
        editable = wb.add_format({"bg_color": "#FFF7D6", "border": 1})

        ws = writer.sheets[POSTS_SHEET]
        for i, col in enumerate(posts_out.columns):
            ws.set_column(i, i, 60 if col == text_field else 22,
                          wrap if col == text_field else None)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, len(posts_out), max(0, len(posts_out.columns) - 1))

        ws = writer.sheets[LABELS_SHEET]
        for i, col in enumerate(labels.columns):
            width = {"common_keywords": 60, "nomic_label": 30,
                     "specific_label": 30, "broad_label": 30}.get(col, 14)
            # The two label columns are what the analyst fills in — highlight them.
            ws.set_column(i, i, width, editable if col in EDITABLE_COLS else
                          (wrap if col == "common_keywords" else None))
        ws.freeze_panes(1, 0)

    return buf.getvalue()


def read_labels(file) -> pd.DataFrame:
    """Read the edited labels tab back, validating its shape."""
    try:
        df = pd.read_excel(file, sheet_name=LABELS_SHEET, dtype=str).fillna("")
    except ValueError as e:
        raise ValueError(
            f"Could not find a `{LABELS_SHEET}` sheet in that workbook. "
            "Upload the file produced by step 1, with its sheet names intact."
        ) from e

    missing = {"nomic_label", *EDITABLE_COLS} - set(df.columns)
    if missing:
        # Workbooks from before the rename used `new_label`; accept them.
        if "new_label" in df.columns and "specific_label" not in df.columns:
            df = df.rename(columns={"new_label": "specific_label"})
            if "broad_label" not in df.columns:
                df["broad_label"] = ""
            missing = set()
    if missing:
        raise ValueError(
            f"The `{LABELS_SHEET}` sheet is missing required column(s): "
            f"{', '.join(sorted(missing))}. Do not rename or delete columns."
        )

    for c in ["nomic_label", *EDITABLE_COLS]:
        df[c] = df[c].astype(str).str.strip().replace("nan", "")
    return df


def build_mapping(labels: pd.DataFrame) -> tuple[dict[str, str], dict[str, str], int, int]:
    """-> (specific_map, broad_map, n_blank_specific, n_blank_broad).

    A blank `specific_label` falls back to the Nomic label, so a partly-filled
    sheet is valid input. A blank `broad_label` stays empty — there is nothing
    sensible to fall back to, since Nomic never produced one.
    """
    specific, broad = {}, {}
    blank_specific = blank_broad = 0

    for _, row in labels.iterrows():
        old = row["nomic_label"]
        if not old:
            continue

        s = row["specific_label"]
        if not s:
            blank_specific += 1
            s = old
        specific[old] = s

        b = row["broad_label"]
        if not b:
            blank_broad += 1
        broad[old] = b

    return specific, broad, blank_specific, blank_broad
