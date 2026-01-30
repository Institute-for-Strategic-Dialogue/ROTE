# lexicon_bp.py (English only)
import io
import re
import json
import logging
import unicodedata
from collections import Counter
from urllib.parse import urlparse
from typing import Optional, Tuple, List, Dict

from flask import Blueprint, request, render_template, redirect, url_for, flash, send_file, current_app
import pandas as pd
import numpy as np
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import spacy.cli
from io import BytesIO
from datetime import datetime  # fixed import

# NEW: optional lxml import for HTML parsing
try:
    from lxml import html as lxml_html
except Exception:
    lxml_html = None

# NEW: optional KeyBERT import for key phrase extraction
try:
    from keybert import KeyBERT
    # Try to use sentence-transformers for better embeddings
    try:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        kw_model = KeyBERT(model=embedding_model)
    except Exception:
        # Fallback to default KeyBERT model
        kw_model = KeyBERT()
    KEYBERT_AVAILABLE = True
except Exception:
    kw_model = None
    KEYBERT_AVAILABLE = False

def extract_key_phrases(doc, top_n=5):
    """Extract key phrases using KeyBERT. Returns list of (phrase, score) tuples."""
    if not KEYBERT_AVAILABLE or not kw_model or not doc or pd.isna(doc):
        return []
    try:
        # KeyBERT returns [(keyword, score), ...]
        keywords = kw_model.extract_keywords(doc,
                                             keyphrase_ngram_range=(1,3),
                                             top_n=top_n)
        return keywords
    except Exception:
        return []
    


# --- Logging helper ---
_MODULE_LOGGER = logging.getLogger(__name__)

def _log_progress(msg: str):
    """Send progress info to the Flask app logger if present; fallback to module logger."""
    try:
        logger = current_app.logger  # type: ignore[attr-defined]
    except Exception:
        logger = _MODULE_LOGGER
    logger.info(msg)

# NEW: truthy parser for request flags
def _truthy(v) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "on", "yes", "y", "t"}

# --- NLTK resources ---
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

# --- English sentiment (VADER) ---
SIA_EN = SentimentIntensityAnalyzer()

# --- Try loading spaCy model; fall back to blank pipeline if absent ---
def _load_spacy_en():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        # Best-effort auto-download; if it still fails, propagate to surface the error
        spacy.cli.download("en_core_web_sm", quiet=True)
        return spacy.load("en_core_web_sm")

# Pre-load cache; load lazily depending on flags
NLP_CACHE = {"en_full": None, "en_blank": None}

def _get_nlp(use_ner: bool = True):
    key = "en_full" if use_ner else "en_blank"
    if NLP_CACHE[key] is None:
        NLP_CACHE[key] = _load_spacy_en() if use_ner else spacy.blank("en")
    return NLP_CACHE[key]

# --- Stopword set ---
STOP_EN = set(stopwords.words("english"))

def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )

def _get_stopwords():
    return STOP_EN

def _split_terms(q: str) -> List[str]:
    """Split a query string on commas or newlines; trim blanks."""
    parts = re.split(r"[,|\n]+", q or "")
    return [p.strip() for p in parts if p and p.strip()]

def _build_filter_mask(
    series: pd.Series,
    query: str,
    *,
    regex: bool = False,
    logic: str = "any",              # "any" or "all"
    case_insensitive: bool = True,
    accent_insensitive: bool = True,
) -> pd.Series:
    """
    Build a boolean mask over 'series' matching 'query' terms.
    - Multi-term queries are ORed ("any") or ANDed ("all").
    - Optional regex mode.
    - Accent-insensitive by default: both haystack and needles are stripped of accents.
    """
    terms = _split_terms(query)
    if not terms:
        return pd.Series(True, index=series.index)

    text = series.fillna("").astype(str)
    if accent_insensitive:
        text_proc = text.apply(_strip_accents)
        terms_proc = [_strip_accents(t) for t in terms]
    else:
        text_proc = text
        terms_proc = terms

    flags = re.IGNORECASE if case_insensitive else 0
    masks: List[pd.Series] = []

    for term in terms_proc:
        if regex:
            try:
                pat = re.compile(term, flags)
                masks.append(text_proc.str.contains(pat, na=False))
            except re.error:
                # Fallback: escape broken regex to literal match
                pat = re.compile(re.escape(term), flags)
                masks.append(text_proc.str.contains(pat, na=False))
        else:
            masks.append(text_proc.str.contains(term, case=not case_insensitive, regex=False, na=False))

    if logic.lower() == "all":
        return pd.concat(masks, axis=1).all(axis=1) if masks else pd.Series(True, index=series.index)
    return pd.concat(masks, axis=1).any(axis=1) if masks else pd.Series(True, index=series.index)

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    # Remove links, @mentions, bare "#"
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    # Remove non-alphanumeric characters (keep digits and basic punctuation)
    text = re.sub(r'[^a-zA-Z0-9\s.!?,;:\-\(\)]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text.strip()

def extract_meta(text: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    if pd.isna(text):
        return [], [], [], []
    mentions = re.findall(r"@(\w+)", text)
    hashtags = re.findall(r"#(\w+)", text)
    links = re.findall(r"https?://\S+", text)
    domains = []
    for link in links:
        try:
            parsed = urlparse(link)
            domains.append(parsed.netloc)
        except Exception:
            continue
    return mentions, hashtags, links, domains

def extract_embedded_links(article_clean_top_node):
    # Find all URLs in the HTML content
    urls = article_clean_top_node.xpath("//a/@href")

    return urls

def extract_tweet_details(article_clean_top_node):
    # Find all embedded tweets using the appropriate class name
    embedded_tweets = article_clean_top_node.xpath(".//blockquote[contains(@class, 'twitter-tweet')]")

    tweets_data = []

    for tweet in embedded_tweets:
        tweet_data = {}

        # The tweet URL is typically contained within the last <a> tag in the blockquote.
        tweet_link = tweet.xpath(".//a/@href")[-1] if tweet.xpath(".//a/@href") else None
        if tweet_link:
            tweet_data['tweetLink'] = tweet_link
            
            # Extract tweet ID from the tweet URL
            tweet_data['tweetId'] = tweet_link.split('/')[-1]
            
            # Extract the screen name from the tweet URL
            parts = tweet_link.split('/')
            tweet_data['screenName'] = parts[-3] if len(parts) > 3 else None

        # The full text of the tweet is contained in the <p> tag within the blockquote
        tweet_text_list = tweet.xpath(".//p//text()")
        tweet_data['tweetText'] = ''.join(tweet_text_list).strip() if tweet_text_list else None

        tweets_data.append(tweet_data)

    return tweets_data


def tokenize(text: str, lang: str = "en") -> List[str]:
    """
    Language-aware tokenization:
    - Lowercase
    - NLTK word_tokenize fallback to regex on error
    - Remove stopwords, keep alphabetic tokens
    """
    sw = _get_stopwords()
    txt = text.lower()
    try:
        toks = word_tokenize(txt, language="english")
    except Exception:
        toks = re.findall(r"[a-z]+", txt)
    return [t for t in toks if t.isalpha() and t not in sw]

def get_ngrams(tokens, n):
    return ["_".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

def build_frequency_series(list_of_token_lists, include_ngrams=(1, 2, 3)):
    counter = Counter()
    total_equivalent = 0
    for tokens in list_of_token_lists:
        if 1 in include_ngrams:
            counter.update(tokens)
            total_equivalent += len(tokens)
        if 2 in include_ngrams:
            bigrams = get_ngrams(tokens, 2)
            counter.update(bigrams)
            total_equivalent += max(len(tokens) - 1, 0)
        if 3 in include_ngrams:
            trigrams = get_ngrams(tokens, 3)
            counter.update(trigrams)
            total_equivalent += max(len(tokens) - 2, 0)
    raw = pd.Series(counter).sort_values(ascending=False)
    normalized = (raw / total_equivalent * 1000) if total_equivalent > 0 else raw * 0.0
    return {
        "raw": raw,
        "normalized": normalized,
        "total_token_equivalents": total_equivalent,
    }

def differential_dataframe(group_freqs, group_a, group_b, top_n=50):
    a_norm = group_freqs[group_a]["normalized"]
    b_norm = group_freqs[group_b]["normalized"]
    candidates = set(
        list(group_freqs[group_a]["raw"].head(top_n).index)
        + list(group_freqs[group_b]["raw"].head(top_n).index)
    )
    records = []
    for term in candidates:
        a_raw = int(group_freqs[group_a]["raw"].get(term, 0))
        b_raw = int(group_freqs[group_b]["raw"].get(term, 0))
        a_n = float(a_norm.get(term, 0.0))
        b_n = float(b_norm.get(term, 0.0))
        diff_norm = a_n - b_n
        records.append(
            {
                "term": term,
                f"{group_a}_raw": a_raw,
                f"{group_b}_raw": b_raw,
                f"{group_a}_norm": round(a_n, 4),
                f"{group_b}_norm": round(b_n, 4),
                "differential_norm": round(diff_norm, 4),
            }
        )
    df_diff = pd.DataFrame(records).sort_values("differential_norm", ascending=False)
    return df_diff

# --- Sentiment (English only) ---

def _sentiment_score_en(text: str) -> float:
    return float(SIA_EN.polarity_scores(text)["compound"])


def sentiment_summary(df: pd.DataFrame, text_col: str, group_col: str) -> pd.DataFrame:
    # Use a unified sentiment score column (VADER compound for English)
    df["score"] = df[text_col].apply(_sentiment_score_en)

    records = []
    for group, sub in df.groupby(group_col):
        scores = sub["score"].dropna()
        n = len(scores)
        if n == 0:
            continue
        mean = scores.mean()
        std = scores.std(ddof=1)
        se = std / np.sqrt(n) if n > 0 else np.nan
        z = 1.96
        records.append(
            {
                "group": group,
                "count": int(n),
                "mean_score": float(mean),
                "std_dev": float(std),
                "std_error": float(se),
                "95ci_lower": float(mean - z * se),
                "95ci_upper": float(mean + z * se),
            }
        )
    return pd.DataFrame(records)

def cum_counts(series_of_lists: pd.Series) -> pd.Series:
    flat = sum(series_of_lists.tolist(), [])
    return pd.Series(flat).value_counts()

lexicon_bp = Blueprint("lexicon", __name__, template_folder="templates", url_prefix="/lexicon")


def _load_dataframe_from_upload(file_storage, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Load a DataFrame from an uploaded file.
    Supports: CSV (.csv) and Excel (.xlsx, .xlsm, .xlsb, .xls).
    - If 'sheet_name' is provided, tries to load that sheet; otherwise loads the first sheet.
    - For CSV, attempts normal read; if that fails, retries with delimiter sniffing.
    """
    if not file_storage or not file_storage.filename:
        raise ValueError("No file provided.")

    filename = (file_storage.filename or "").lower().strip()
    content = file_storage.read()
    bio = io.BytesIO(content)

    # Excel?
    if filename.endswith((".xlsx", ".xlsm", ".xlsb", ".xls")):
        # Try engines appropriate to extension
        ext = filename.rsplit(".", 1)[-1]
        engine_candidates = []
        if ext in ("xlsx", "xlsm", "xlsb"):
            engine_candidates = ["openpyxl", None]  # None lets pandas choose
        elif ext == "xls":
            engine_candidates = ["xlrd", None]

        last_err = None
        for eng in engine_candidates:
            try:
                bio.seek(0)
                if sheet_name:
                    return pd.read_excel(bio, sheet_name=sheet_name, engine=eng)
                # No sheet specified → load first sheet
                bio.seek(0)
                xls = pd.ExcelFile(bio, engine=eng)
                first = xls.sheet_names[0] if xls.sheet_names else 0
                return pd.read_excel(xls, sheet_name=first)
            except Exception as e:
                last_err = e

        # If we reach here, we failed to read Excel
        hint = ("Install 'openpyxl' for .xlsx/.xlsm/.xlsb or 'xlrd' for .xls "
                "if the default engine is unavailable.")
        raise ValueError(f"Failed to read Excel file. {hint} Error: {last_err}")

    # CSV (default)
    try:
        bio.seek(0)
        return pd.read_csv(bio)
    except Exception:
        # Retry with delimiter sniffing using the Python engine
        bio.seek(0)
        return pd.read_csv(bio, sep=None, engine="python")
    
@lexicon_bp.route("/", methods=["GET", "POST"])
def upload_and_process():
    results: Dict = {}
    columns: List[str] = []
    lang = "en"
    _log_progress("lexicon: request start lang=en")

    # Defaults so we can include them in META even on GET
    filter_query = (request.form.get("filter_query") or "").strip()
    filter_logic = (request.form.get("filter_logic") or "any").lower()   # "any" or "all"
    filter_regex = bool(request.form.get("filter_regex"))
    
    if request.method == "POST":
        print("lexicon: file upload received")
        uploaded = request.files.get("csv_file")
        if not uploaded or not uploaded.filename:
            flash("No file uploaded", "warning")
            return redirect(url_for("lexicon.upload_and_process"))

        try:
            excel_sheet = (request.form.get("excel_sheet") or "").strip() or None
            df = _load_dataframe_from_upload(uploaded, sheet_name=excel_sheet)
            _log_progress(f"lexicon: file loaded rows={len(df)} cols={len(df.columns)}")
        except Exception as e:
            flash(f"Failed to read file: {e}", "danger")
            return redirect(url_for("lexicon.upload_and_process"))

        columns = df.columns.tolist()
        text_col = request.form.get("text_column")
        group_col = request.form.get("group_column") or None
        enrich_only = _truthy(request.form.get("enrich_only"))
        fmt = (request.form.get("format") or "").strip().lower()

        # NEW: toggles for optional NLP work
        enable_tokens = _truthy(request.form.get("enable_tokens", "true"))
        enable_entities = _truthy(request.form.get("enable_entities", "true"))
        enable_sentiment = _truthy(request.form.get("enable_sentiment", "false"))
        # NEW: toggle for key phrase extraction
        enable_key_phrases = _truthy(request.form.get("enable_key_phrases", "false"))
        print("lexicon: processing options -", {
            "enrich_only": enrich_only,
            "enable_tokens": enable_tokens,
            "enable_entities": enable_entities,
            "enable_sentiment": enable_sentiment,
            "enable_key_phrases": enable_key_phrases,
        })
        # Check KeyBERT availability
        if enable_key_phrases and not KEYBERT_AVAILABLE:
            flash("KeyBERT is not available. Install 'keybert' and 'sentence-transformers' to enable key phrase extraction.", "warning")
            enable_key_phrases = False

        # NEW: optional HTML text column name (for XPath parsing)
        html_col = (request.form.get("html_column") or "").strip() or None
        if html_col and html_col not in df.columns:
            flash(f"HTML column '{html_col}' not found in data", "warning")
            html_col = None
        if html_col and lxml_html is None:
            flash("lxml is not available; HTML parsing is disabled. Install 'lxml' to enable.", "warning")
            html_col = None

        if not text_col or text_col not in df.columns:
            flash("Text column not provided or invalid", "danger")
            return redirect(url_for("lexicon.upload_and_process"))

        # ---------------------------
        # NEW: optional filtering
        # ---------------------------
        if filter_query:
            mask = _build_filter_mask(
                df[text_col],
                filter_query,
                regex=filter_regex,
                logic=filter_logic,
                case_insensitive=True,
                accent_insensitive=True,
            )
            kept = int(mask.sum())
            total = int(len(mask))
            df = df[mask].copy()
            print(f"lexicon: filter applied kept={kept}/{total}")
            results["filter_info"] = {
                "query": filter_query,
                "logic": filter_logic,
                "regex": filter_regex,
                "kept": kept,
                "total": total,
            }

            if kept == 0:
                # Still proceed, but give a friendly heads-up
                flash("Filter returned zero rows. Try different terms or disable the filter.", "warning")

        # --- Clean + extract ---
        df["clean_text"] = df[text_col].apply(clean_text)
        print("lexicon: clean text done")

        # Tokens (optional)
        if enable_tokens:
            df["tokens"] = df["clean_text"].apply(lambda t: tokenize(t, lang))
            print("lexicon: tokens generated")
        else:
            df["tokens"] = [[] for _ in range(len(df))]

        # Entities (optional, lazy spaCy)
        if enable_entities:
            nlp = _get_nlp(use_ner=True)
            if not nlp.has_pipe("ner"):
                flash("spaCy NER model unavailable. Install 'en_core_web_sm' to enable entity extraction.", "danger")
                return redirect(url_for("lexicon.upload_and_process"))
            df["entities"] = df[text_col].apply(lambda t: [ent.text for ent in nlp(t).ents])
            print("lexicon: entities extracted via spaCy NER")
        else:
            df["entities"] = [[] for _ in range(len(df))]

        # NEW: Key phrases (optional)
        if enable_key_phrases:
            df["key_phrases_raw"] = df["clean_text"].apply(lambda t: extract_key_phrases(t, top_n=5))
            df["key_phrases"] = df["key_phrases_raw"].apply(lambda kw_list: [phrase for phrase, score in kw_list])
            df["key_phrase_scores"] = df["key_phrases_raw"].apply(lambda kw_list: [f"{phrase}:{score:.3f}" for phrase, score in kw_list])
            print("lexicon: key phrases extracted")
        else:
            df["key_phrases"] = [[] for _ in range(len(df))]
            df["key_phrase_scores"] = [[] for _ in range(len(df))]

        # Links, mentions, hashtags, domains
        meta = df[text_col].apply(extract_meta)
        df["mentions"] = meta.apply(lambda x: x[0])
        df["hashtags"] = meta.apply(lambda x: x[1])
        df["links"] = meta.apply(lambda x: x[2])
        df["domains"] = meta.apply(lambda x: x[3])
        print("lexicon: meta (mentions/hashtags/links/domains) extracted")

        # NEW: HTML parsing with XPath (embedded links + tweets)
        if html_col:
            df["__html_root"] = df[html_col].apply(_safe_parse_html)
            df["embedded_links_xpath"] = df["__html_root"].apply(
                lambda node: (extract_embedded_links(node) if node is not None else [])
            )
            df["embedded_tweets"] = df["__html_root"].apply(
                lambda node: (extract_tweet_details(node) if node is not None else [])
            )
            # Convenience lists for counting
            df["tweet_ids"] = df["embedded_tweets"].apply(
                lambda lst: [d.get("tweetId") for d in lst if isinstance(d, dict) and d.get("tweetId")]
            )
            df["tweet_screen_names"] = df["embedded_tweets"].apply(
                lambda lst: [d.get("screenName") for d in lst if isinstance(d, dict) and d.get("screenName")]
            )
            # Clean up heavy roots to keep memory down
            del df["__html_root"]
            print("lexicon: HTML parsing complete")

        # Grouping
        if group_col and group_col in df.columns:
            groups = df[group_col].dropna().unique()
        else:
            group_col = "__all__"
            df[group_col] = "all"
            groups = ["all"]

        # Frequencies (optional; only when tokens are enabled)
        if enable_tokens and not enrich_only:
            group_freqs = {}
            for g in groups:
                token_lists = df[df[group_col] == g]["tokens"].tolist()
                group_freqs[g] = build_frequency_series(token_lists, include_ngrams=(1, 2, 3))
            print("lexicon: frequency tables built")

            # Summaries per group
            summary_per_group = {}
            for g in group_freqs:
                norm = group_freqs[g]["normalized"].sort_values(ascending=False).head(1000)
                raw = group_freqs[g]["raw"].sort_values(ascending=False).head(1000)
                summary_per_group[g] = {
                    "top_normalized": norm.round(4).to_dict(),
                    "top_raw": raw.to_dict(),
                    "total_token_equivalents": group_freqs[g]["total_token_equivalents"],
                }
            results["group_freqs"] = summary_per_group

            # Differential if exactly two groups
            if len(groups) == 2:
                a, b = list(groups)
                diff_df = differential_dataframe(group_freqs, a, b, top_n=100)
                results["differential"] = diff_df.to_dict(orient="records")
                results["differential_groups"] = (a, b)

        # Sentiment (optional, language-aware)
        if enable_sentiment:
            sentiment_df = sentiment_summary(df, "clean_text", group_col)
            print("lexicon: sentiment computed")
            results["sentiment_summary"] = [] if enrich_only else sentiment_df.to_dict(orient="records")
        else:
            results["sentiment_summary"] = []

        # Meta counts
        meta_summary = {}
        if not enrich_only:
            for g in groups:
                subset = df[df[group_col] == g]
                meta_summary[g] = {
                    "mentions": cum_counts(subset["mentions"]).to_dict(),
                    "hashtags": cum_counts(subset["hashtags"]).to_dict(),
                    "links": cum_counts(subset["links"]).to_dict(),
                    "domains": cum_counts(subset["domains"]).to_dict(),
                    "entities": cum_counts(subset["entities"]).to_dict(),
                    "key_phrases": cum_counts(subset["key_phrases"]).to_dict(),
                }
                # NEW: add HTML-derived meta if present
                if "embedded_links_xpath" in df.columns:
                    meta_summary[g]["embedded_links_xpath"] = cum_counts(subset["embedded_links_xpath"]).to_dict()
                if "tweet_ids" in df.columns:
                    meta_summary[g]["tweet_ids"] = cum_counts(subset["tweet_ids"]).to_dict()
                if "tweet_screen_names" in df.columns:
                    meta_summary[g]["tweet_screen_names"] = cum_counts(subset["tweet_screen_names"]).to_dict()
            _log_progress("lexicon: meta summary built")
        results["meta"] = meta_summary

        # ---------- Export handling (Excel / CSV) ----------
        if request.method == "POST" and fmt in {"excel", "csv"}:
            # Always build an enriched dataframe for export
            def _serialize_cell(val):
                if isinstance(val, (list, tuple)):
                    return "; ".join([json.dumps(v) if isinstance(v, dict) else str(v) for v in val])
                return val

            export_df = df.copy()
            if "clean_text" in export_df.columns:
                export_df = export_df.drop(columns=["clean_text"])
            list_like_cols = [
                "tokens",
                "entities",
                "key_phrases",
                "key_phrase_scores",
                "mentions",
                "hashtags",
                "links",
                "domains",
                "embedded_links_xpath",
                "embedded_tweets",
                "tweet_ids",
                "tweet_screen_names",
            ]
            for col in list_like_cols:
                if col in export_df.columns:
                    export_df[col] = export_df[col].apply(_serialize_cell)

            if fmt == "csv":
                _log_progress("lexicon: exporting enriched CSV")
                output = BytesIO()
                export_df.to_csv(output, index=False)
                output.seek(0)
                filename = "lexical_enriched_en_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"
                return send_file(output, mimetype="text/csv", as_attachment=True, download_name=filename)

            # Excel export path (single or multi-sheet)
            _log_progress("lexicon: exporting Excel")
            sheets: Dict[str, pd.DataFrame] = {}
            sheets["ENRICHED"] = export_df

            if not enrich_only:
                # === Raw & normalized top terms (single tables with 'group' col) ===
                raw_rows, norm_rows = [], []
                for group, summary in results.get("group_freqs", {}).items():
                    for term, count in summary.get("top_raw", {}).items():
                        raw_rows.append({"group": group, "term": term, "count_raw": count})
                    for term, val in summary.get("top_normalized", {}).items():
                        norm_rows.append({"group": group, "term": term, "normalized_per_thousand": val})

                if raw_rows:
                    sheets["TOP_RAW"] = (
                        pd.DataFrame(raw_rows)
                        .sort_values(["group", "count_raw"], ascending=[True, False])
                        .reset_index(drop=True)
                    )
                if norm_rows:
                    sheets["TOP_NORMALIZED"] = (
                        pd.DataFrame(norm_rows)
                        .sort_values(["group", "normalized_per_thousand"], ascending=[True, False])
                        .reset_index(drop=True)
                    )

                # === Group totals ===
                totals_rows = []
                for group, summary in results.get("group_freqs", {}).items():
                    totals_rows.append({
                        "group": group,
                        "total_token_equivalents": summary.get("total_token_equivalents", 0)
                    })
                if totals_rows:
                    sheets["GROUP_TOTALS"] = pd.DataFrame(totals_rows).sort_values("group").reset_index(drop=True)

                # === Meta counts (mentions, hashtags, links, domains, entities, + HTML) ===
                meta_keys = ["mentions", "hashtags", "links", "domains", "entities", "key_phrases"]
                any_meta = results.get("meta") or {}
                if any(k in (any_meta.get(g, {}) or {}) for g in any_meta for k in ["embedded_links_xpath"]):
                    meta_keys.append("embedded_links_xpath")
                if any(k in (any_meta.get(g, {}) or {}) for g in any_meta for k in ["tweet_ids"]):
                    meta_keys.append("tweet_ids")
                if any(k in (any_meta.get(g, {}) or {}) for g in any_meta for k in ["tweet_screen_names"]):
                    meta_keys.append("tweet_screen_names")

                for meta_key in meta_keys:
                    meta_rows = []
                    for group, metas in results.get("meta", {}).items():
                        for val, count in (metas or {}).get(meta_key, {}).items():
                            meta_rows.append({"group": group, meta_key[:-1] if meta_key.endswith("s") else meta_key: val, "count": count})
                    if meta_rows:
                        sheets[meta_key.upper()] = (
                            pd.DataFrame(meta_rows)
                            .sort_values(["group", "count"], ascending=[True, False])
                            .reset_index(drop=True)
                        )

                # === Differential (own sheet) ===
                if "differential" in results and results.get("differential_groups"):
                    sheets["DIFFERENTIAL"] = pd.DataFrame(results["differential"])

                # === Sentiment (own sheet) ===
                if results.get("sentiment_summary"):
                    sheets["SENTIMENT_SUMMARY"] = pd.DataFrame(results["sentiment_summary"])

                # === Detailed embedded tweets sheet (flattened)
                if "embedded_tweets" in df.columns:
                    tweet_rows = []
                    for _, row in df.iterrows():
                        grp = row[group_col]
                        for t in row["embedded_tweets"]:
                            if not isinstance(t, dict):
                                continue
                            tweet_rows.append({
                                "group": grp,
                                "tweetId": t.get("tweetId"),
                                "screenName": t.get("screenName"),
                                "tweetLink": t.get("tweetLink"),
                                "tweetText": t.get("tweetText"),
                            })
                    if tweet_rows:
                        sheets["EMBEDDED_TWEETS"] = pd.DataFrame(tweet_rows).sort_values(["group", "screenName"]).reset_index(drop=True)

            # === META (record filter + language + html column + flags) ===
            fi = results.get("filter_info", {})
            sheets["META"] = pd.DataFrame([{
                "language": "en",
                "filter_query": fi.get("query", ""),
                "filter_logic": fi.get("logic", ""),
                "filter_regex": fi.get("regex", False),
                "filtered_rows_kept": fi.get("kept", None),
                "filtered_rows_total": fi.get("total", None),
                "html_column": (request.form.get("html_column") or "").strip(),
                "enable_tokens": _truthy(request.form.get("enable_tokens", "false")),
                "enable_entities": _truthy(request.form.get("enable_entities", "false")),
                "enable_sentiment": _truthy(request.form.get("enable_sentiment", "false")),
                # NEW: record key phrase flag
                "enable_key_phrases": _truthy(request.form.get("enable_key_phrases", "false")),
                "enrich_only": enrich_only,
                "keybert_available": KEYBERT_AVAILABLE,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }])

            filename = ("lexical_enriched_" if enrich_only else "lexical_sentiment_analysis_")
            filename += "en_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".xlsx"
            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                for sheet_name, data in sheets.items():
                    safe = sheet_name[:31].replace("/", "_").replace("\\", "_")
                    data.to_excel(writer, sheet_name=safe, index=False)
            output.seek(0)
              # SAve file locally for debugging
            with open("debug_output.xlsx", "wb") as f:
                f.write(output.getbuffer())
            
            return send_file(
                output,
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                as_attachment=True,
                download_name=filename,
            )

    return render_template(
        "lexicon.html",
        results=results,
        columns=columns,
        request=request,
    )

