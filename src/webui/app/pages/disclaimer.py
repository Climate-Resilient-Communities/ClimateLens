import streamlit as st

st.title("About & Legal")

about_tab, disclaimer_tab = st.tabs([
    "About",
    "Disclaimer"
    ])


st.markdown("""
This project is intended for research and educational purposes only.

It is not a clinical tool and should not be used
for diagnosis or treatment.
""")