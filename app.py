# app.py
import os
import secrets
from flask import Flask, render_template
from text_extractor import text_bp
from lexicon import lexicon_bp
from yt_oembed import yt_oembed_bp
from temporal_analyzer import temporal_bp
from tiktok_ads import tiktok_ads_bp
from wp_scraper import wp_scraper_bp
from wikidata_fetcher import wikidata_bp
from serp_social import serp_social_bp
from link_unshorten import link_unshorten_bp
from crypto_transfers import crypto_transfers_bp
from site_liveness import site_liveness_bp
from site_downloader import site_downloader_bp  
from text_cluster import text_cluster_bp

app = Flask(__name__)
# Use a stable env var in prod; fall back to a strong random for dev.
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY") or secrets.token_urlsafe(32)
app.register_blueprint(text_bp, url_prefix='/extract')
app.register_blueprint(lexicon_bp, url_prefix='/lexicon')
app.register_blueprint(yt_oembed_bp, url_prefix='/youtube')
app.register_blueprint(temporal_bp, url_prefix='/temporal')
app.register_blueprint(tiktok_ads_bp, url_prefix="/tiktok")
app.register_blueprint(wp_scraper_bp, url_prefix="/wordpress")
app.register_blueprint(wikidata_bp, url_prefix="/wikidata")
app.register_blueprint(serp_social_bp, url_prefix="/serp_social")
app.register_blueprint(link_unshorten_bp, url_prefix="/link_unshorten")
app.register_blueprint(crypto_transfers_bp, url_prefix="/crypto_transfers")
app.register_blueprint(site_liveness_bp, url_prefix="/site_liveness")
app.register_blueprint(site_downloader_bp, url_prefix="/site_downloader")
app.register_blueprint(text_cluster_bp, url_prefix="/text_cluster") 
# TODO: add a blueprint/tool that clusters text documents based on semantic similarity e.g. "This is insane! check out tthis AI nudifier" vs "Look at this crazy deepfake porn generator" should be similar, where "cats are friendly and cute" is very different; clusters can be of any size. Maybe use KNN?
# TODO: Add a GDELT wrapper 

@app.route('/')
def index():
    tools = [
        {
            "name": "Text Extractor",
            "description": "Extract and count patterns from text, JSON, or database.",
            "url": "/extract",
            "icon": "fa-regular fa-file-lines",
            "bg": "bg-tool-indigo"
        },
        {
            "name": "Lexicon CSV Enricher",
            "description": "Extract links, “@” mentions, named entities, sentiment, and hashtags from CSV files.",
            "url": "/lexicon",
            "icon": "fa-solid fa-spell-check",
            "bg": "bg-tool-teal"
        },
        {
            "name": "YouTube Metadata Extractor",
            "description": "Extract metadata from YouTube videos.",
            "url": "/youtube",
            "icon": "fa-brands fa-youtube",
            "bg": "bg-tool-red"
        },
        {
            "name": "Temporal Analyzer",
            "description": "Analyze temporal patterns, bursts, and coordination signals.",
            "url": "/temporal",
            "icon": "fa-solid fa-clock-rotate-left",
            "bg": "bg-tool-amber"
        },
        {
            "name": "TikTok Ad Library Search (UNDER CONSTRUCTION)",
            "description": "UNDER CONSTRUCTION: Query TikTok’s Ad Library for multiple search terms, from text or CSV.",
            "url": "/tiktok",
            "icon": "fa-brands fa-tiktok",
            "bg": "bg-tool-black"
        },
        {
            "name": "WordPress Article Scraper",
            "description": "Fetch posts from any WordPress site via the REST API, with optional name resolution.",
            "url": "/wordpress",
            "icon": "fa-solid fa-newspaper",
            "bg": "bg-tool-gold"
        },
        {
            "name": "Wikidata Social Media Fetcher",
            "description": "Fetch social media profiles from Wikidata based on existing usernames.",
            "url": "/wikidata",
            "icon": "fa-brands fa-wikipedia-w",
            "bg": "bg-tool-purple"
        },
        {
            "name": "SerpAPI Social Media Search",
            "description": "Search social media sites using SerpAPI based on input names or terms.",
            "url": "/serp_social",
            "icon": "fa-solid fa-magnifying-glass",
            "bg": "bg-tool-green"
        },
        {
            "name": "Link Unshortener",
            "description": "Unshorten lists of URLs from text or CSV files.",
            "url": "/link_unshorten",
            "icon": "fa-solid fa-link",
            "bg": "bg-tool-gray"
        },
        {
            "name": "Crypto Transaction Extractor",
            "description": "UNDER CONSTRUCTION: Extract cryptocurrency transactions from public wallet addresses.",
            "url": "/crypto_transfers",
            "icon": "fa-brands fa-bitcoin",
            "bg": "bg-tool-orange"
        },
        {
            "name": "Site Liveness Checker",
            "description": "Check if websites are live or down from lists of URLs.",
            "url": "/site_liveness",
            "icon": "fa-solid fa-signal",
            "bg": "bg-tool-black"
        },
        {
            "name": "Site Downloader",
            "description": "UNDER CONSTRUCTION: Download website content for offline analysis.",
            "url": "/site_downloader",
            "icon": "fa-solid fa-download",
            "bg": "bg-tool-cyan"
        },
        {
            "name": "Text Clustering Tool",
            "description": "Cluster text documents based on semantic similarity.",
            "url": "/text_cluster",
            "icon": "fa-solid fa-object-group",
            "bg": "bg-tool-indigo"
        },
        
    ]
    return render_template('index.html', tools=tools)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render sets $PORT
    app.run(host="0.0.0.0", port=port, debug=False)
