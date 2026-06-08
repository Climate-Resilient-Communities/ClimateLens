import streamlit as st

st.set_page_config(
page_title="ClimateLens",
page_icon="🌍",
layout="wide"
)

st.title("🌍 ClimateLens")
st.caption("Understanding climate-related emotions through social media analysis")

st.markdown("""
ClimateLens is an open-source research platform that explores climate-related
emotions, concerns, and discussions through computational social science and
natural language processing (NLP).

The project helps researchers, educators, mental health professionals, and
community organizations better understand how climate-related concerns are
expressed in online conversations.
""")

st.info("""
Use the navigation menu in the sidebar to explore key insights,
interactive dashboards, frequently asked questions, project information,
and terms of use.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 Explore Key Insights")

    st.markdown("""
    Review high-level findings from our analysis of climate-related
    conversations and emotional themes.
    """)

with col2:
    st.subheader("📊 Interactive Dashboards")

    st.markdown("""
    Explore topic models, sentiment visualizations, and other
    interactive analytics generated from the research pipeline.
    """)

st.subheader("What ClimateLens Does")

st.markdown("""
ClimateLens is designed to:

* Identify and explore climate-related emotional themes
* Visualize patterns in climate conversations
* Support research and educational initiatives
* Improve understanding of climate anxiety and related experiences
* Provide interpretable insights through interactive dashboards
  """)

st.subheader("What ClimateLens Does Not Do")

st.markdown("""
ClimateLens is **not** a clinical assessment tool and should not be used to:

* Diagnose mental health conditions
* Provide treatment recommendations
* Replace professional mental health services
* Evaluate or score individual people
* Make clinical decisions without professional oversight
  """)

st.caption(
"ClimateLens is an open-source research project focused on climate-related emotions and wellbeing."
)

st.caption("Last updated: June 2026")