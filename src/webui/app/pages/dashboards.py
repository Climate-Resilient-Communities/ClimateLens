import streamlit as st
import streamlit.components.v1 as components

from utils import get_visualizations

st.title("📊 Dashboards")

st.markdown("""
This section lets you explore the conversations and feelings around climate anxiety.
Think of it as moving from the headline to the full story.

Use the Topics tab to see what people are talking about, and the Sentiments tab
to understand how they feel.
""")

with st.expander("Introduction & Guide"):
    st.write("""
    This section lets you explore the conversations and feelings around climate anxiety.
    Think of it as moving from the headline to the full story.
    """)

with st.expander("Understanding the Data Visualization Tools"):
    st.write("""
    Each tool shows the same conversations from a different angle.
    Some focus on what people discuss (topics), others on how they feel (sentiments).
    Together, they offer a fuller picture.
    """)

with st.expander("How to Use This Dashboard"):
    st.write("""
    Start with the Barchart or Intertopic Map to see the main conversations.

    Use the Heatmap and Hierarchy to explore connections and branches.

    Then open the Sentiments tab to check the feelings behind the words.
    """)

topic_tab, sentiment_tab = st.tabs(
    ["📊 Topics", "🎭 Sentiments"]
)

with topic_tab:

    files = get_visualizations("assets/topics/dtm")

    selected_file = st.selectbox(
        "Choose Visualization",
        files,
        format_func=lambda x: x.stem.replace("_", " ").title(),
        key="topics"
    )

    if selected_file.suffix in [".png", ".jpg", ".jpeg"]:
        st.image(selected_file, use_container_width=True)

    elif selected_file.suffix == ".html":
        with open(selected_file, encoding="utf-8") as f:
            html = f.read()

        components.html(
            html,
            height=900,
            scrolling=True
        )
'''
with sentiment_tab:

    files = get_visualizations("assets/sentiments")

    selected_file = st.selectbox(
        "Choose Visualization",
        files,
        format_func=lambda x: x.stem.replace("_", " ").title(),
        key="sentiments"
    )

    if selected_file.suffix in [".png", ".jpg", ".jpeg"]:
        st.image(selected_file, use_container_width=True)

    elif selected_file.suffix == ".html":
        with open(selected_file, encoding="utf-8") as f:
            html = f.read()

        components.html(
            html,
            height=900,
            scrolling=True
        )
        '''