from pathlib import Path

import streamlit as st
from utils import get_visualizations, render_visualization

st.title("📊 Dashboards")

st.markdown("""
This section lets you explore the conversations and emotions surrounding climate anxiety.
Think of it as moving from the headline to the full story.

Use the **Topics** tab to see what people are discussing, and the **Emotions** tab
to understand the emotional patterns behind those conversations.
""")

with st.expander("📖 Introduction & Guide"):
    st.write("""
    This section lets you explore the conversations and emotions surrounding climate anxiety.
    Think of it as moving from the headline to the full story.
    """)

with st.expander("🔍 Understanding the Data Visualization Tools"):
    st.write("""
    Each tool shows the same conversations from a different angle.

    Some focus on what people discuss (topics), while others focus on how people feel (emotions).

    Together, they provide a richer understanding of climate-related conversations.
    """)

with st.expander("🧭 How to Use This Dashboard"):
    st.write("""
    Start with the Barchart or Intertopic Map to identify major themes.

    Use Heatmaps and Hierarchies to explore relationships between topics.

    Then switch to the Emotions tab to understand the emotional signals associated
    with those conversations.

    You don't need to explore everything—follow your curiosity.
    """)

## Main Tabs
topic_tab, emotion_tab = st.tabs(
    ["📊 Topics", "🎭 Emotions"]
)

## Topics
with topic_tab:

    try: # works on HF
        topic_folders = sorted(
        [f for f in Path("src/assets/topics").iterdir() if f.is_dir()]
    )
    except Exception as e: # works locally
        topic_folders = sorted(
        [f for f in Path("assets/topics").iterdir() if f.is_dir()]
    )

    if not topic_folders:
        st.warning("No topic visualization folders found.")
    else:

        selected_folder = st.selectbox(
            "Visualization Type",
            topic_folders,
            format_func=lambda x: x.name.replace("_", " ").title(),
            key="topic_folder"
        )

        files = get_visualizations(selected_folder)

        if not files:
            st.warning("No visualizations found in this folder.")
        else:

            selected_file = st.selectbox(
                "Visualization",
                files,
                format_func=lambda x: x.stem.replace("_", " ").title(),
                key="topic_file"
            )

            render_visualization(selected_file)

## Emotions
with emotion_tab:
    try: # works on HF
        emotion_folders = sorted(
        [f for f in Path("src/assets/emotions").iterdir() if f.is_dir()]
    )
    except Exception as e: # works locally
        emotion_folders = sorted(
        [f for f in Path("assets/emotions").iterdir() if f.is_dir()]
    )

    if not emotion_folders:
        st.warning("No emotion visualization folders found.")
    else:

        selected_folder = st.selectbox(
            "Visualization Type",
            emotion_folders,
            format_func=lambda x: x.name.replace("_", " ").title(),
            key="emotion_folder"
        )

        files = get_visualizations(selected_folder)

        if not files:
            st.warning("No visualizations found in this folder.")
        else:

            selected_file = st.selectbox(
                "Visualization",
                files,
                format_func=lambda x: x.stem.replace("_", " ").title(),
                key="emotion_file"
            )

            render_visualization(selected_file)
