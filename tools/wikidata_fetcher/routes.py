from wikidata.client import Client
import requests
import pandas as pd
import time
from io import StringIO
from typing import Dict, Optional, Tuple
import os
from flask import Blueprint, render_template, request

# --- Config ---
URL_OR_CSV = os.environ.get("WDF_INPUT_MODE", "CSV")  # "URL" or "CSV"
CSV_URL = os.environ.get("WDF_CSV_URL", "")           # used when URL_OR_CSV == "URL"
CSV_FILE = os.environ.get("WDF_CSV_FILE", r"C:\Users\PeterBenzoni\repo\social_profiler\Hamilton_Account_List2024.csv")
OUTPUT_FILE = os.environ.get("WDF_OUTPUT_FILE", "updated_domains_with_social_media.csv")
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = os.environ.get("WDF_USER_AGENT", "ResearchToolsWikidataFetcher/1.0 (contact: pb@isdglobal.org)")

# --- Wikidata client and HTTP session ---
client = Client()
session = requests.Session()
session.headers.update({
    "User-Agent": USER_AGENT,
    "Accept": "application/sparql-results+json",
})

# --- Mappings ---
social_media_properties: Dict[str, str] = {
    'P2002':   'Twitter username',
    'P2013':   'Facebook ID',
    'P2003':   'Instagram username',
    'P2397':   'YouTube channel ID',
    'P7085':   'TikTok username',
    'P3789':   'Telegram username',
    'P11892':  'Threads username',
    'P11962':  'Rumble channel',
    'P3185':   'VK ID',
    'P8919':   'Gab username',
    'P9269':   'Odnoklassniki user numeric ID',
    'P8904':   'Parler username',
    'P10858':  'Truth Social username',
    'P856':   'Official website',
    # Add more properties as needed
}

column_to_property: Dict[str, str] = {
    'Twitter ID': 'P2002',
    'Facebook ID': 'P2013',
    'Instagram username': 'P2003',
    'YouTube channel ID': 'P2397',
    'TikTok username': 'P7085',
    'Telegram ID': 'P3789',
    'Threads username': 'P11892',
    'Rumble channel': 'P11962',
    'VK ID': 'P3185',
    'Gab username': 'P8919',
    'Odnoklassniki user numeric ID': 'P9269',
    'Parler username': 'P8904',
    'Truth Social username': 'P10858',
    'Official website': 'P856',
    # Add other mappings as needed
}

property_id_to_column: Dict[str, str] = {v: k for k, v in column_to_property.items()}

# --- Flask Blueprint ---
wikidata_bp = Blueprint("wikidata", __name__, template_folder="templates")

def get_wikidata_id(property_id: str, value: str, retries: int = 3, backoff: float = 1.0) -> Optional[str]:
    """
    Resolve a Wikidata QID via SPARQL by matching a wdt:PROPERTY "value".
    Returns QID like 'Q42' or None.
    """
    sparql_query = f"""
        SELECT DISTINCT ?item WHERE {{
            ?item wdt:{property_id} "{value}".
        }}
        LIMIT 1
    """
    params = {'query': sparql_query, 'format': 'json'}
    for attempt in range(retries):
        try:
            resp = session.get(SPARQL_ENDPOINT, params=params, timeout=30)
            if resp.status_code in (429, 502, 503, 504):
                time.sleep(backoff * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            bindings = data.get('results', {}).get('bindings', [])
            if bindings:
                wikidata_uri = bindings[0]['item']['value']
                return wikidata_uri.rsplit('/', 1)[-1]
            return None
        except Exception as e:
            if attempt == retries - 1:
                print(f"[WARN] SPARQL failed for {property_id}='{value}': {e}")
                return None
            time.sleep(backoff * (attempt + 1))
    return None

def _to_scalar_datavalue(val) -> str:
    # Convert Wikidata datavalue payload into a scalar string
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        # common cases: {'id': 'Q...'}, {'text': '...'}, {'time': '...'}
        for key in ("text", "id", "value"):
            if key in val and isinstance(val[key], str):
                return val[key]
    return str(val)

def extract_social_media_profiles(claims: Dict, props: Dict[str, str]) -> Dict[str, str]:
    """
    Extract property -> scalar value for the given social media properties.
    """
    profiles: Dict[str, str] = {}
    for prop_id, stmts in (claims or {}).items():
        if prop_id not in props:
            continue
        for st in stmts or []:
            try:
                mainsnak = st.get('mainsnak') or {}
                if mainsnak.get('snaktype') != 'value':
                    continue
                dv = mainsnak.get('datavalue', {}).get('value')
                val = _to_scalar_datavalue(dv).strip()
                if val:
                    profiles[prop_id] = val
            except Exception:
                continue
    return profiles

def get_social_media_profiles_from_id(wikidata_id: str) -> Optional[Dict[str, str]]:
    try:
        item = client.get(wikidata_id, load=True)
        claims = item.data.get('claims', {})
        return extract_social_media_profiles(claims, social_media_properties)
    except Exception as e:
        print(f"[WARN] Error fetching item {wikidata_id}: {e}")
        return None

def _read_input_dataframe() -> pd.DataFrame:
    if URL_OR_CSV.upper() == "URL":
        if not CSV_URL:
            raise ValueError("CSV_URL must be set when URL_OR_CSV == 'URL'.")
        r = requests.get(CSV_URL, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        return pd.read_csv(StringIO(r.content.decode('utf-8')), dtype=str, keep_default_na=False)
    if URL_OR_CSV.upper() == "CSV":
        return pd.read_csv(CSV_FILE, dtype=str, keep_default_na=False)
    raise ValueError("URL_OR_CSV must be 'URL' or 'CSV'")

def _read_form_dataframe(req) -> Tuple[pd.DataFrame, str]:
    """Read CSV from form (upload or URL). Returns (df, source_label)."""
    mode = (req.form.get("input_mode") or "CSV").upper()
    ua = req.form.get("user_agent") or USER_AGENT
    # Update session UA if provided
    if ua and session.headers.get("User-Agent") != ua:
        session.headers["User-Agent"] = ua
    if mode == "URL":
        csv_url = req.form.get("csv_url", "").strip()
        if not csv_url:
            raise ValueError("CSV URL is required when Input Source is URL.")
        r = requests.get(csv_url, headers={"User-Agent": ua})
        r.raise_for_status()
        return pd.read_csv(StringIO(r.content.decode("utf-8")), dtype=str, keep_default_na=False), f"URL:{csv_url}"
    # CSV upload
    file = req.files.get("csv_file")
    if not file or file.filename == "":
        raise ValueError("CSV file upload is required when Input Source is CSV.")
    return pd.read_csv(file, dtype=str, keep_default_na=False), f"FILE:{file.filename}"

def _build_mapping_from_form(req) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Create column_to_property and reverse mapping from submitted form fields.
    Ignores empty values.
    """
    mapping: Dict[str, str] = {}
    # Expect inputs like column_map[P2002] = "Twitter ID"
    for prop_id in social_media_properties.keys():
        field_name = f"column_map[{prop_id}]"
        col = (req.form.get(field_name) or "").strip()
        if col:
            mapping[col] = prop_id
    # Fallback: if user left everything blank, use defaults
    if not mapping:
        mapping = dict(column_to_property)
    reverse = {v: k for k, v in mapping.items()}
    return mapping, reverse

def _process_dataframe(df: pd.DataFrame, col_to_prop: Dict[str, str], prop_to_col: Dict[str, str]) -> Dict[str, int]:
    """Enrich dataframe in-place based on provided mappings. Returns summary stats."""
    # Ensure required columns exist (based on mapping provided)
    for col in col_to_prop.keys():
        if col not in df.columns:
            df[col] = None

    if 'updated' not in df.columns:
        df['updated'] = None

    if 'wikidata_id' not in df.columns:
        df['wikidata_id'] = None

    id_cache: Dict[Tuple[str, str], Optional[str]] = {}
    resolved_count = 0
    updated_rows = 0

    for idx, row in df.iterrows():
        wikidata_id = row.get('wikidata_id') or None

        if not wikidata_id:
            for column, prop in col_to_prop.items():
                val = row.get(column)
                if val and isinstance(val, str) and val.strip():
                    key = (prop, val.strip())
                    if key not in id_cache:
                        id_cache[key] = get_wikidata_id(prop, val.strip())
                        time.sleep(0.25)
                    wikidata_id = id_cache[key]
                    if wikidata_id:
                        resolved_count += 1
                        df.at[idx, 'wikidata_id'] = wikidata_id
                        break

        if wikidata_id:
            profiles = get_social_media_profiles_from_id(wikidata_id) or {}
            if profiles:
                any_update = False
                for prop_id, value in profiles.items():
                    col_name = prop_to_col.get(prop_id)
                    if not col_name:
                        continue
                    current = row.get(col_name)
                    if current is None or str(current).strip() == "" or pd.isna(current):
                        df.at[idx, col_name] = value
                        any_update = True
                if any_update:
                    updated_rows += 1
                    df.at[idx, 'updated'] = True
            time.sleep(0.5)
    return {
        "rows": int(len(df)),
        "resolved_qids": int(resolved_count),
        "rows_updated": int(updated_rows),
    }

@wikidata_bp.route("/", methods=["GET", "POST"])
def wikidata_view():
    results = None
    if request.method == "POST":
        try:
            output_file = request.form.get("output_file") or OUTPUT_FILE
            col_map, rev_map = _build_mapping_from_form(request)
            df, source = _read_form_dataframe(request)

            if request.form.get("format") == "preview":
                # Show basic preview without processing
                results = {
                    "mode": request.form.get("input_mode") or "CSV",
                    "source": source,
                    "rows": int(len(df)),
                    "mapping": col_map,
                    "note": "Preview only; no enrichment run.",
                }
            else:
                summary = _process_dataframe(df, col_map, {v: k for k, v in col_map.items()})
                df.to_csv(output_file, index=False)
                results = {
                    **summary,
                    "output_file": output_file,
                    "source": source,
                    "user_agent": session.headers.get("User-Agent"),
                }
        except Exception as e:
            results = {"error": str(e)}
    return render_template("wikidata.html", results=results)

def main():
    df = _read_input_dataframe()
    summary = _process_dataframe(df, column_to_property, property_id_to_column)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[OK] Updated data saved to {OUTPUT_FILE}")
    print(summary)

if __name__ == "__main__":
    main()
