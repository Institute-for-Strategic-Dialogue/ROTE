from .routes import nomic_relabel_bp as blueprint

TOOL_META = {
    "name": "Nomic Relabeller",
    "description": "Rename Nomic Atlas topic labels in a spreadsheet, then rebuild the dataset with your own specific and broad topics.",
    "url": "/nomic_relabel",
    "icon": "fa-solid fa-tags",
    "bg": "bg-tool-cyan",
}
