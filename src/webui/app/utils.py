from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


def get_visualizations(folder):
    valid_extensions = {".html", ".png", ".jpg", ".jpeg", ".webp"}

    return sorted([
        f
        for f in Path(folder).iterdir()
        if f.is_file() and f.suffix.lower() in valid_extensions
    ])

# Render a visualization based on file type. Used in dashboards.py
def render_visualization(file_path):
    if file_path is None:
        st.warning("No visualization selected.")
        return

    if file_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
        st.image(file_path, use_container_width=True)

    elif file_path.suffix.lower() == ".html":
        with open(file_path, "r", encoding="utf-8") as f:
            html = f.read()

        components.html(
            html,
            height=900,
            scrolling=True
        )

    else:
        st.info(f"Unsupported file type: {file_path.suffix}")
