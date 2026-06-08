import streamlit as st

st.title("❓ Support & FAQs")
st.markdown("""
            Get help using ClimateLens and find answers to common questions.
""")

with st.expander("What will your tool not do?"):
    st.write("""
    It won't run live surveillance or score individual youth in real time.
    And it won't auto-prescribe personalized interventions. Those are explicitly out-of-scope for this phase.
    """)

with st.expander("Will this replace counselors?"):
    st.write("""
    No. It's decision support, not decision maker.
    The deliverables include an interpretable model + UI meant to surface patterns and speed human judgment.
    KHP leads on intervention strategy and practice.
    """)

with st.expander("If it doesn't flag individuals, what does a counselor actually see?"):
    st.write("""A visual dashboard of aggregate themes and interpretable signals (e.g., prevalent concerns, representative phrases), with an interface to explore how a prediction was made; KHP designs the human intervention guidance that sits beside these insights.""")

with st.expander("What counts as a 'good' model here?"):
    st.write("""Not just accuracy. We'll judge success on:
- (a) detection quality against baselines
- (b) interpretability: clear linguistic cues you can inspect, and
- (c) fairness across subgroups.
""")

with st.expander("Where does the data come from and do youth consent?"):
    st.write("""Phase I uses open-source, climate-related text.
Phase II contemplates anonymized/de-identified Kids Help Phone transcripts that are used only after de-identification and governance review.""")

with st.expander("How do you keep the system from drifting or getting brittle over time (slang, memes, shifts)?"):
    st.write("""
We treat this like a living system: evaluate routinely, watch for distribution shift, and use human-in-the-loop review to update lexicons/models when new patterns appear. The lifecycle includes explicit monitoring and testing for changes.
""")

with st.expander("Could this be biased against certain groups?"):
    st.write("""
Bias risk is real. We address it by measuring subgroup performance, requiring interpretable rationales (so reviewers can spot skewed cues), and baking fairness checks into testing alongside clear documentation of limits.
""")

with st.expander("What happens when the model is wrong?"):
    st.write("""
We plan for error. Baselines and evaluation tests are set up to catch regressions; the UI shows why a prediction happened; and there's a human feedback loop to correct mistakes and refine the system.
""")

with st.expander("Can communities reuse this without heavy ML expertise?"):
    st.write("""
Yes! Code and models are released with usage guidelines, and the interface ships with built-in visualizations so non-technical teams can explore results safely and meaningfully.
""")



st.subheader("🧠 For Mental Health Professionals")

st.markdown("""
ClimateLens is designed as a research and decision-support tool, not a clinical
assessment or intervention system.

If you are a psychologist, counselor, social worker, educator, or other mental
health professional, we encourage you to:

- Use your professional judgment when interpreting findings and visualizations.
- Maintain appropriate boundaries between research insights and clinical practice.
- Consider the broader context of individuals and communities when applying research findings.
- Follow applicable ethical, legal, and professional guidelines within your field.
- Seek additional training or consultation when incorporating climate psychology research into professional work.

While ClimateLens can help surface trends and themes in climate-related conversations,
human expertise remains essential for understanding, supporting, and responding to
mental health concerns.
""")



st.subheader("📬 Questions, Feedback, or Concerns")

st.markdown("""
We welcome questions about the project, methodology, data sources,
limitations, and appropriate use of the platform.

If you have feedback, suggestions, or concerns, please reach out to the project team:

**Email:** programs@crcgreen.com

Your input helps us improve ClimateLens and ensure it remains a useful,
transparent, and responsible resource for researchers, educators,
mental health professionals, and community organizations.
""")

st.link_button(
    "💬 Submit Feedback",
    "https://forms.gle/o7QQBZNijJo9E76E7",
    use_container_width=True
)
