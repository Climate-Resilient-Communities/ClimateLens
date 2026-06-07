import streamlit as st

st.title("❓ FAQ")

with st.expander("What will your tool not do?"):
    st.write("""
    It won't run live surveillance or score
    individual youth in real time.
    """)

with st.expander("Will this replace counselors?"):
    st.write("""
    No. This is a decision-support tool.
    """)