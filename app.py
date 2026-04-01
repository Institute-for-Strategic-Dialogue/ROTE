# app.py
import os
import secrets
from flask import Flask, render_template
from tools import discover_tools

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY") or secrets.token_urlsafe(32)

# Auto-discover and register all tool blueprints
tool_registry = []
for bp, url_prefix, meta in discover_tools():
    app.register_blueprint(bp, url_prefix=url_prefix)
    tool_registry.append(meta)

# External tools (linked out, no local code)
EXTERNAL_TOOLS = [
    {
        "name": "ISD Archiver",
        "description": "Archive and preserve web pages, social media posts, and other online content.",
        "url": "https://archive.isd.ngo",
        "icon": "fa-solid fa-box-archive",
        "bg": "bg-tool-slate",
        "external": True,
    },
]
# TODO: add a blueprint/tool that clusters text documents based on semantic similarity e.g. "This is insane! check out tthis AI nudifier" vs "Look at this crazy deepfake porn generator" should be similar, where "cats are friendly and cute" is very different; clusters can be of any size. Maybe use KNN?
# TODO: Add a GDELT wrapper

@app.route('/')
def index():
    return render_template('index.html', tools=tool_registry + EXTERNAL_TOOLS)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render sets $PORT
    app.run(host="0.0.0.0", port=port, debug=False)
