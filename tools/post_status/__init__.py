from .routes import post_status_bp as blueprint

TOOL_META = {
    "name": "Post & Account Status",
    "description": "Check whether X posts and accounts are still live, deleted, suspended, or protected. Designed to extend to other platforms.",
    "url": "/post_status",
    "icon": "fa-solid fa-heart-pulse",
    "bg": "bg-tool-slate",
}
