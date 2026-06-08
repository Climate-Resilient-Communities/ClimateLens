import streamlit as st

st.title("📊 Dashboards")

st.markdown("""
This section lets you explore the conversations and feelings around climate anxiety. Think of it as moving from the headline to the full story. Use the Topics tab to see what people are talking about, and the Sentiments tab to understand how they feel.        
""")

with st.expander("Introduction & Guide"):
    st.write("""
             This section lets you explore the conversations and feelings around climate anxiety. Think of it as moving from the headline to the full story. Use the Topics tab to see what people are talking about, and the Sentiments tab to understand how they feel.
""")

with st.expander("Understanding the Data Visualization Tools"):
    st.write("""
Each tool shows the same conversations from a different angle. Some focus on what people discuss (topics), others on how they feel (sentiments). Together, they offer a fuller picture.
""")

with st.expander("How to Use This Dashboard"):
    st.write("""
Start with the Barchart or Intertopic Map to see the main conversations. Use the Heatmap and Hierarchy to explore connections and branches. Then open the Sentiments tab to check the feelings behind the words. You don't need to view everything—follow your curiosity.
""")

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