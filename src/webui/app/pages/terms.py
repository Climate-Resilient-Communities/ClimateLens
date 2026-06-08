import streamlit as st

st.title("📋 Terms of Use")

st.markdown("""
These terms govern the use of ClimateLens and its associated data,
models, visualizations, and research outputs.
""")

with st.expander("🎯 Scope of Acceptable Use"):
    st.markdown("""By using ClimateLens, you agree that your use is limited to:
    - Academic, clinical, or policy research
    - Educational and training activities
    - Professional development and learning
    - Research related to climate mental health and wellbeing

    Users are responsible for complying with all applicable laws,
    regulations, and professional standards.
    """)

with st.expander("🚫 Prohibited Uses"):
    st.markdown("""
    ClimateLens and its outputs must not be used to:

    - Provide direct clinical care or psychological services
    - Make diagnoses or treatment decisions
    - Replace professional mental health assessment
    - Infer or claim an individual's mental health status from social media activity
    - Violate privacy, confidentiality, or ethical obligations
    """)

with st.expander("🩺 Professional Responsibility"):
    st.markdown("""
    Mental health professionals remain fully responsible for their
    own professional practice and decision-making.

    ClimateLens is intended as a research and decision-support tool
    and does not replace clinical judgment, training, supervision,
    or ethical obligations.
    """)

with st.expander("📚 Attribution & Citation"):
    st.markdown("""
    When using ClimateLens outputs in research, publications,
    presentations, or professional work, users should:

    - Provide appropriate citation and attribution
    - Acknowledge the open-source nature of the project
    - Acknowledge the limitations of computational analysis
    - Avoid misrepresenting the project team or their credentials
    """)

with st.expander("🔒 Data Use & Privacy"):
    st.markdown("""
    ClimateLens analyzes publicly available content and research data.

    Users are responsible for ensuring compliance with relevant:

    - Privacy laws
    - Research ethics requirements
    - Institutional policies
    - Professional regulations

    Users may not attempt to extract or identify individuals from
    project outputs.
    """)

with st.expander("⚖️ Liability"):
    st.markdown("""
    ClimateLens is provided for research and educational purposes.

    The project team, contributors, and affiliated organizations are
    not responsible for decisions, actions, or outcomes resulting from
    the use of the platform or its outputs.

    This includes:
    - Clinical decisions
    - Policy decisions
    - Research interpretations
    - Technical errors or inaccuracies
    """)

with st.expander("🔄 Changes to These Terms"):
    st.markdown("""
    These terms may be updated periodically as the project evolves.

    Continued use of ClimateLens constitutes acceptance of any updated
    terms.
    """)

with st.expander("🍁 Governing Law"):
    st.markdown("""
    These terms are governed by the laws of Canada and applicable
    provincial regulations.

    Any disputes arising from the use of ClimateLens shall be resolved
    through appropriate legal processes within the relevant jurisdiction.
    """)