from .routes import site_liveness_bp as blueprint

TOOL_META = {
    "name": "Site Liveness Checker",
    "description": "Check if websites are live or down from lists of URLs.",
    "url": "/site_liveness",
    "icon": "fa-solid fa-signal",
    "bg": "bg-tool-black",
}
