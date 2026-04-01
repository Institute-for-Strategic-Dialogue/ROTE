from .routes import image_match_bp as blueprint

TOOL_META = {
    "name": "Image Matcher",
    "description": "Find exact and near-duplicate images using perceptual hashing, within one set or across two sets.",
    "url": "/image_match",
    "icon": "fa-solid fa-images",
    "bg": "bg-tool-purple",
}
