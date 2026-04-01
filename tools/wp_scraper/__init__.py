from .routes import wp_scraper_bp as blueprint

TOOL_META = {
    "name": "WordPress Article Scraper",
    "description": "Fetch posts from any WordPress site via the REST API, with optional name resolution.",
    "url": "/wordpress",
    "icon": "fa-solid fa-newspaper",
    "bg": "bg-tool-gold",
}
