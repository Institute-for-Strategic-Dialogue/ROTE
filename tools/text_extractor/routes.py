# File: text_extractor.py
import io
import json
import re
from collections import Counter
from io import BytesIO
from typing import Iterable, Tuple, Optional, List, Callable

import pandas as pd
from flask import Blueprint, render_template, request, send_file
#from sqlalchemy import create_engine

text_bp = Blueprint('text_extractor', __name__, template_folder='templates')

# ##### DATABASE FUNCTIONS #####
# def get_engine(server: str,
#                database: str,
#                username: str,
#                password: str,
#                driver: str = "ODBC Driver 17 for SQL Server"):
#     conn_str = (
#         f"mssql+pyodbc://{username}:{password}@{server}/{database}"
#         f"?driver={driver.replace(' ', '+')}"
#     )
#     return create_engine(conn_str)


# def fetch_raw_text(engine,
#                    table: str,
#                    column: str,
#                    like_phrase: str = '@',
#                    limit: int = None,
#                    order_by: str = None):
#     where = f"{column} LIKE '%{like_phrase}%'"
#     top = f"TOP {limit}" if limit else ""
#     order = f"ORDER BY {order_by} DESC" if order_by else ""
#     query = f"SELECT {top} {column} FROM {table} WHERE {where} {order};"
#     return pd.read_sql_query(query, engine)

##### XPATTERN EXTRACTION #####
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

def extract_embedded_links(article_clean_top_node):
    # Find all URLs in the HTML content
    urls = article_clean_top_node.xpath("//a/@href")

    return urls

##### TELEGRAM HELPERS #####
def _flatten_telegram_text(msg) -> Tuple[str, list, list, list]:
    """
    Returns: (full_text, hashtags[], mentions[], links[])
    Handles Telegram's list/dict mixed text structure.
    """
    txt = msg.get('text', '')
    hashtags, mentions, links = [], [], []

    def push_entity(part):
        t = part.get('type')
        # Prefer href for links if present, else text
        val = part.get('href') or part.get('text', '')
        if t == 'hashtag':
            hashtags.append(part.get('text', ''))
        elif t == 'mention':
            mentions.append(part.get('text', ''))
        elif t == 'link':
            links.append(val)

    if isinstance(txt, list):
        parts = []
        for part in txt:
            if isinstance(part, dict):
                push_entity(part)
                parts.append(part.get('text', ''))
            else:
                parts.append(str(part))
        return ''.join(parts), hashtags, mentions, links
    elif isinstance(txt, str):
        return txt, hashtags, mentions, links
    else:
        return str(txt), hashtags, mentions, links


def extract_from_telegram_json(data: dict, matcher, filename: str) -> dict:
    """
    Returns per-file raw lists needed for aggregation.
    Keys: name, channel_id, filename, messages, matches, mentions, hashtags, links
    """
    channel_name = data.get('name') or data.get('title') or 'Unknown'
    channel_id = data.get('id') or data.get('channel_id') or 'unknown_id'

    all_matches, all_mentions, all_hashtags, all_links, all_forwarded = [], [], [], [], []
    all_crypto = []
    msgs = data.get('messages', []) or []

    for msg in msgs:
        if msg.get('forwarded_from'):
            all_forwarded.append(msg['forwarded_from'])
        full_text, hashtags, mentions, links = _flatten_telegram_text(msg)
        all_hashtags.extend(hashtags)
        all_mentions.extend(mentions)
        all_links.extend(links)
        all_matches.extend(_run_match_extractor(full_text or "", matcher))
        all_crypto.extend(extract_crypto_wallets(full_text or ""))

    return {
        'name': channel_name,
        'channel_id': channel_id,
        'filename': filename,
        'messages': len(msgs),
        'matches': all_matches,
        'mentions': all_mentions,
        'hashtags': all_hashtags,
        'links': all_links,
        'forwarded': all_forwarded,
        'crypto_wallets': all_crypto,
    }


def _normalize_social_handle(handle: str) -> str:
    if not handle:
        return ""
    cleaned = handle.strip()
    cleaned = cleaned.split("?")[0]
    cleaned = cleaned.strip().strip("/")
    if cleaned.startswith("@"):
        cleaned = cleaned[1:]
    return cleaned

SOCIAL_MEDIA_REGEX = {
    "Facebook": r"https?://(?:www\.)?facebook\.com/([^/?]+)",
    "YouTube": r"https?://(?:www\.)?youtube\.com/(?:user|channel)/([^/?]+)",
    "Instagram": r"https?://(?:www\.)?instagram\.com/(@?[\w.-]+)",
    "TikTok": r"https?://(?:www\.)?tiktok\.com/(@?[\w.-]+)",
    "LinkedIn": r"https?://(?:www\.)?linkedin\.com/in/([\w-]+)",
    "Telegram": r"https?://(?:www\.)?t\.me/([\w-]+)",
    "Douyin": r"https?://(?:www\.)?douyin\.com/(@?[\w.-]+)",
    "QQ": r"https?://user\.qzone\.qq\.com/(\d+)",

    "Snapchat": r"https?://(?:www\.)?snapchat\.com/add/([\w.-]+)",
    "Pinterest": r"https?://(?:www\.)?pinterest\.com/([^/?]+)",
    "Reddit": r"https?://(?:www\.)?reddit\.com/user/([\w-]+)",
    "Twitter": r"https?://(?:www\.)?twitter\.com/([\w-]+)",
    "imo": r"https?://(?:www\.)?imo\.im/([\w.-]+)",
    "Line": r"https?://(?:www\.)?line\.me/R/ti/p/([\w.-]+)",
    "Vevo": r"https?://(?:www\.)?vevo\.com/([^/?]+)",
    "Discord": r"https?://(?:www\.)?discord(?:app)?\.com/([\w.-]+)",
    "Twitch": r"https?://(?:www\.)?twitch\.tv/([\w.-]+)",
    "VK": r"https?://(?:www\.)?vk\.com/([\w.-]+)",
    "Parler": r"https?://(?:www\.)?parler\.com/profile/([\w.-]+)",
    "Gab": r"https?://(?:www\.)?gab\.com/([\w.-]+)",
    "Odysee": r"https?://(?:www\.)?odysee\.com/(@?[\w.-]+)",
    "LBRY": r"https?://(?:www\.)?lbry\.tv/(@?[\w.-]+)",
    "Truth Social": r"https?://(?:www\.)?truthsocial\.com/user/([\w.-]+)",
    "BitChute": r"https?://(?:www\.)?bitchute\.com/channel/([\w.-]+)",
    "Gettr": r"https?://(?:www\.)?gettr\.com/user/([\w.-]+)",
    "Rumble": r"https?://(?:www\.)?rumble\.com/([\w.-]+)",
    "Locals": r"https?://(?:www\.)?locals\.com/([\w.-]+)",
    "Apple Podcasts": r"https?://(?:podcasts\.apple\.com|itunes\.apple\.com)/([^/?]+)",
    "iHeartRadio": r"https?://(?:www\.)?iheart\.com/(?:[^/]+/)?(?:podcast|show)/([\w-]+)",
    "Google Play": r"https?://play\.google\.com/store/apps/details\?id=([\w.]+)",
}
CRYPTO_WALLET_REGEX = {
    "Bitcoin": r"\b(?:bc1[a-z0-9]{25,87}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})\b",
    "Ethereum": r"\b0x[a-fA-F0-9]{40}\b",
    "Litecoin": r"\b[LM3][a-km-zA-HJ-NP-Z1-9]{26,33}\b",
    "Dogecoin": r"\bD[5-9A-HJ-NP-U][1-9A-HJ-NP-Za-km-z]{32}\b",
    "Monero": r"\b4[0-9AB][1-9A-HJ-NP-Za-km-z]{93}\b",
    "Ripple": r"\br[0-9A-Za-z]{24,34}\b",
    "Solana": r"\b[1-9A-HJ-NP-Za-km-z]{43,44}\b",
    "Cardano": r"\baddr1[0-9a-z]{58}\b",
}

def extract_social_profiles(text: str) -> List[Tuple[str, str]]:
    if not text:
        return []
    results: List[Tuple[str, str]] = []
    for platform, pattern in SOCIAL_MEDIA_REGEX.items():
        try:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
        except re.error:
            continue
        for match in matches:
            if isinstance(match, tuple):
                candidate = next((m for m in match if isinstance(m, str) and m), "")
            else:
                candidate = match
            handle = _normalize_social_handle(candidate)
            if handle:
                results.append((platform, handle))
    return results


def social_match_extractor(selected_platform: Optional[str] = None):
    selected_items = (
        [(selected_platform, SOCIAL_MEDIA_REGEX.get(selected_platform, ""))]
        if selected_platform and selected_platform in SOCIAL_MEDIA_REGEX
        else list(SOCIAL_MEDIA_REGEX.items())
    )
    def _extract(text: str) -> List[str]:
        hits: List[str] = []
        for platform, pattern in selected_items:
            if not pattern:
                continue
            try:
                matches = re.findall(pattern, text or "", flags=re.IGNORECASE)
            except re.error:
                continue
            for match in matches:
                candidate = match if isinstance(match, str) else next(
                    (m for m in match if isinstance(m, str) and m), ""
                )
                handle = _normalize_social_handle(candidate)
                if handle:
                    hits.append(f"{platform}:{handle}")
        return hits
    return _extract

def extract_crypto_wallets(text: str) -> List[Tuple[str, str]]:
    if not text:
        return []
    results: List[Tuple[str, str]] = []
    for currency, pattern in CRYPTO_WALLET_REGEX.items():
        try:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
        except re.error:
            continue
        for match in matches:
            candidate = match if isinstance(match, str) else next(
                (m for m in match if isinstance(m, str) and m), ""
            )
            cleaned = candidate.strip()
            if cleaned:
                results.append((currency, cleaned))
    return results

def _run_match_extractor(text: str, matcher) -> List[str]:
    if matcher is None:
        return []
    if callable(matcher):
        return matcher(text or "")
    if not isinstance(matcher, re.Pattern):
        matcher = re.compile(matcher)
    return _extract_matches_single(text or "", matcher)


def _extract_matches_single(text: str, pattern: re.Pattern) -> List[str]:
    found = pattern.findall(text or "")
    normalized: List[str] = []
    for m in found:
        if isinstance(m, tuple):
            s = next((g for g in m if isinstance(g, str) and g), "")
            if not s:
                s = "::".join([str(g) for g in m])
            normalized.append(s)
        else:
            normalized.append(m)
    return normalized


def extract_pattern_from_text(text_or_list, regex) -> list:
    """
    For text/db modes. Accepts a str or iterable of strings and returns all matches.
    """
    if regex is None:
        return []
    matcher = regex if (callable(regex) or isinstance(regex, re.Pattern)) else re.compile(regex)
    out = []
    if isinstance(text_or_list, str):
        items = [text_or_list]
    elif isinstance(text_or_list, Iterable):
        items = text_or_list
    else:
        items = [str(text_or_list)]

    for t in pd.Series(items).dropna():
        out.extend(_run_match_extractor(str(t), matcher))
    return out


def aggregate_counts(items: list) -> pd.DataFrame:
    counts = Counter(items)
    return (
        pd.DataFrame(counts.items(), columns=["value", "count"])
          .sort_values(by="count", ascending=False)
          .reset_index(drop=True)
    )

def _load_tabular_file(file_storage, *, sheet_name: Optional[str] = None) -> pd.DataFrame:
    if not file_storage or not file_storage.filename:
        raise ValueError("No file provided for tabular extraction.")
    filename = (file_storage.filename or "").lower()
    raw = file_storage.read()
    bio = BytesIO(raw)

    if filename.endswith(".csv"):
        try:
            bio.seek(0)
            return pd.read_csv(bio)
        except Exception:
            bio.seek(0)
            return pd.read_csv(bio, sep=None, engine="python")
    if filename.endswith((".xlsx", ".xlsm", ".xlsb", ".xls")):
        ext = filename.rsplit(".", 1)[-1]
        engine_candidates = []
        if ext in ("xlsx", "xlsm", "xlsb"):
            engine_candidates = ["openpyxl", None]
        elif ext == "xls":
            engine_candidates = ["xlrd", None]
        last_err = None
        for eng in engine_candidates or [None]:
            try:
                bio.seek(0)
                if sheet_name:
                    return pd.read_excel(bio, sheet_name=sheet_name, engine=eng)
                bio.seek(0)
                xls = pd.ExcelFile(bio, engine=eng)
                first = xls.sheet_names[0] if xls.sheet_names else 0
                return pd.read_excel(xls, sheet_name=first)
            except Exception as exc:
                last_err = exc
        raise ValueError(f"Failed to read Excel file. Error: {last_err}")
    raise ValueError("Unsupported file type. Upload CSV or Excel.")

##### EXCEL UTIL #####
def _sheet(df: pd.DataFrame, cols: list, sort_cols: list) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = df[cols]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[True if c != 'count' else False for c in sort_cols])
    return df.reset_index(drop=True)

@text_bp.route('/', methods=['GET', 'POST'])
def extract():
    results = {}
    if request.method == 'POST':
        mode = request.form.get('mode')
        regex_mode = (request.form.get('regex_mode') or 'custom').lower()
        social_choice = (request.form.get('social_platform') or '').strip()
        regex_input = (request.form.get('regex') or r"@([A-Za-z0-9_]+)").strip()

        if regex_mode == 'social':
            matcher = social_match_extractor(social_choice if social_choice.lower() != 'all' else None)
        else:
            try:
                matcher = re.compile(regex_input)
            except re.error as exc:
                results['error'] = f"Invalid regular expression: {exc}"
                return render_template('text_extractor.html', results=results)

        filename = 'results.xlsx'
        sheets = {}

        if mode == 'text':
            text = request.form.get('text', '')
            matches = extract_pattern_from_text(text, matcher)
            df = aggregate_counts(matches).rename(columns={'value': 'pattern'})
            sheets['results'] = df
            filename = 'text_extract.xlsx'

        elif mode == 'json':
            files = request.files.getlist('json_files')
            if not files:
                single = request.files.get('json_file')
                if single:
                    files = [single]

            matches_rows, mentions_rows, hashtags_rows, links_rows, forwarded_rows = [], [], [], [], []
            wallet_rows = []
            summary_rows = []

            for f in files:
                try:
                    data = json.load(f)
                except Exception:
                    continue

                ex = extract_from_telegram_json(data, matcher, f.filename or 'file.json')

                summary_rows.append({
                    'name': ex['name'],
                    'channel_id': ex['channel_id'],
                    'filename': ex['filename'],
                    'messages': ex['messages'],
                    'matches_total': len(ex['matches']),
                    'mentions_total': len(ex['mentions']),
                    'hashtags_total': len(ex['hashtags']),
                    'links_total': len(ex['links']),
                    'forwarded_total': len(ex['forwarded']),
                    'crypto_total': len(ex['crypto_wallets']),
                })

                # Per-file tallies → long-form rows
                for val, cnt in Counter(ex['matches']).items():
                    matches_rows.append({
                        'name': ex['name'],
                        'channel_id': ex['channel_id'],
                        'filename': ex['filename'],
                        'match': val,
                        'count': cnt
                    })
                for val, cnt in Counter(ex['mentions']).items():
                    mentions_rows.append({
                        'name': ex['name'],
                        'channel_id': ex['channel_id'],
                        'filename': ex['filename'],
                        'mention': val,
                        'count': cnt
                    })
                for val, cnt in Counter(ex['hashtags']).items():
                    hashtags_rows.append({
                        'name': ex['name'],
                        'channel_id': ex['channel_id'],
                        'filename': ex['filename'],
                        'hashtag': val,
                        'count': cnt
                    })
                for val, cnt in Counter(ex['links']).items():
                    links_rows.append({
                        'name': ex['name'],
                        'channel_id': ex['channel_id'],
                        'filename': ex['filename'],
                        'link': val,
                        'count': cnt
                    })
                for val, cnt in Counter(ex['forwarded']).items():
                    forwarded_rows.append({
                        'name': ex['name'],
                        'channel_id': ex['channel_id'],
                        'filename': ex['filename'],
                        'forwarded': val,
                        'count': cnt
                    })
                for (currency, wallet), cnt in Counter(ex['crypto_wallets']).items():
                    wallet_rows.append({
                        'name': ex['name'],
                        'channel_id': ex['channel_id'],
                        'filename': ex['filename'],
                        'currency': currency,
                        'wallet': wallet,
                        'count': cnt,
                    })


            # Build sheets
            summary_df = pd.DataFrame(summary_rows).sort_values(
                ['name', 'filename']
            ).reset_index(drop=True)

            matches_df  = pd.DataFrame(matches_rows)
            mentions_df = pd.DataFrame(mentions_rows)
            hashtags_df = pd.DataFrame(hashtags_rows)
            links_df    = pd.DataFrame(links_rows)

            sheets['SUMMARY']  = _sheet(summary_df,
                                        ['name', 'channel_id', 'filename', 'messages', 'matches_total', 'mentions_total', 'hashtags_total', 'links_total', 'crypto_total'],
                                        ['name', 'filename'])
            sheets['MATCHES']  = _sheet(matches_df,
                                        ['name', 'channel_id', 'filename', 'match', 'count'],
                                        ['name', 'filename', 'count'])
            sheets['MENTIONS'] = _sheet(mentions_df,
                                        ['name', 'channel_id', 'filename', 'mention', 'count'],
                                        ['name', 'filename', 'count'])
            sheets['HASHTAGS'] = _sheet(hashtags_df,
                                        ['name', 'channel_id', 'filename', 'hashtag', 'count'],
                                        ['name', 'filename', 'count'])
            sheets['LINKS']    = _sheet(links_df,
                                        ['name', 'channel_id', 'filename', 'link', 'count'],
                                        ['name', 'filename', 'count'])
            sheets['FORWARDED'] = _sheet(pd.DataFrame(forwarded_rows),
                                         ['name', 'channel_id', 'filename', 'forwarded', 'count'],
                                         ['name', 'filename', 'count'])
            sheets['CRYPTO_WALLETS'] = _sheet(pd.DataFrame(wallet_rows),
                                              ['name', 'channel_id', 'filename', 'currency', 'wallet', 'count'],
                                              ['name', 'filename', 'count'])

            filename = 'json_extract_multi.xlsx'

        elif mode == 'tabular':
            table_file = (
                request.files.get('table_file')
                or request.files.get('csv_file')
                or request.files.get('excel_file')
                or request.files.get('file')
            )
            if not table_file:
                results['error'] = "No CSV or Excel file uploaded."
                return render_template('text_extractor.html', results=results)

            sheet_name = (request.form.get('sheet_name') or "").strip() or None
            try:
                df = _load_tabular_file(table_file, sheet_name=sheet_name)
            except ValueError as exc:
                results['error'] = str(exc)
                return render_template('text_extractor.html', results=results)

            text_column = (request.form.get('text_column') or "").strip()
            if not text_column:
                results['error'] = "Please specify the text column to analyze."
                return render_template('text_extractor.html', results=results)
            if text_column not in df.columns:
                results['error'] = f"Column '{text_column}' not found in the uploaded file."
                return render_template('text_extractor.html', results=results)

            series = df[text_column].fillna("").astype(str)
            matches_all: List[str] = []
            row_outputs: List[dict] = []
            social_counter: Counter = Counter()
            crypto_counter: Counter = Counter()

            for idx, text in series.items():
                row_matches = _run_match_extractor(text, matcher)
                matches_all.extend(row_matches)
                profiles = extract_social_profiles(text)
                wallets = extract_crypto_wallets(text)
                for platform, handle in profiles:
                    social_counter[(platform, handle)] += 1
                for currency, wallet in wallets:
                    crypto_counter[(currency, wallet)] += 1
                row_outputs.append({
                    'row_index': idx,
                    text_column: text,
                    'matches': "; ".join(row_matches),
                    'match_count': len(row_matches),
                    'social_profiles': "; ".join(f"{platform}:{handle}" for platform, handle in profiles),
                    'social_profile_count': len(profiles),
                    'crypto_wallets': "; ".join(f"{currency}:{wallet}" for currency, wallet in wallets),
                    'crypto_wallet_count': len(wallets),
                })

            match_df = aggregate_counts(matches_all).rename(columns={'value': 'pattern'})
            social_rows = [
                {'platform': platform, 'handle': handle, 'count': count}
                for (platform, handle), count in social_counter.items()
            ]
            social_df = (
                pd.DataFrame(social_rows)
                .sort_values(['platform', 'count'], ascending=[True, False])
                .reset_index(drop=True)
                if social_rows else pd.DataFrame(columns=['platform', 'handle', 'count'])
            )
            wallet_rows = [
                {'currency': currency, 'wallet': wallet, 'count': count}
                for (currency, wallet), count in crypto_counter.items()
            ]
            wallet_df = (
                pd.DataFrame(wallet_rows)
                .sort_values(['currency', 'count'], ascending=[True, False])
                .reset_index(drop=True)
                if wallet_rows else pd.DataFrame(columns=['currency', 'wallet', 'count'])
            )
            row_df = pd.DataFrame(row_outputs)
            if row_df.empty:
                row_df = pd.DataFrame(columns=['row_index', text_column, 'matches', 'match_count', 'social_profiles', 'social_profile_count', 'crypto_wallets', 'crypto_wallet_count'])
            else:
                row_df = row_df[['row_index', text_column, 'matches', 'match_count', 'social_profiles', 'social_profile_count', 'crypto_wallets', 'crypto_wallet_count']]

            sheets['MATCHES'] = match_df
            sheets['SOCIAL_PROFILES'] = social_df
            sheets['CRYPTO_WALLETS'] = wallet_df
            sheets['ROW_RESULTS'] = row_df
            filename = 'tabular_extract.xlsx'
        else:
            results['error'] = "Unsupported extraction mode."
            return render_template('text_extractor.html', results=results)

        # Create Excel file in memory
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for sheet_name, data in sheets.items():
                (data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)).to_excel(
                    writer, sheet_name=sheet_name, index=False
                )
        output.seek(0)

        # Return download response
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )

    # GET: render form
    return render_template('text_extractor.html', results=results)
