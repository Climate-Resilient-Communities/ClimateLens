import streamlit as st

st.title("📊 Dashboards")

topic_tab, sentiment_tab = st.tabs(
    ["Topics", "Sentiments"]
)

with topic_tab:
    st.write("Topic visualizations")

with sentiment_tab:
    st.write("Sentiment visualizations")

view = st.selectbox(
    "Choose Visualization",
    [
        "Intertopic Map",
        "Heatmap",
        "Barchart",
        "Hierarchy"
    ]
)