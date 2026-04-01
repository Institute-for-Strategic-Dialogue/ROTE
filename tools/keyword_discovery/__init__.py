from .routes import keyword_discovery_bp as blueprint

TOOL_META = {
    "name": "Keyword Discovery",
    "description": "Discover co-occurring, semantically similar, and coded/dogwhistle terms from a corpus via chained iterative expansion.",
    "url": "/keyword_discovery",
    "icon": "fa-solid fa-diagram-project",
    "bg": "bg-tool-rose",
}
