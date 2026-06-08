import streamlit as st

## Page Setup
home_page = st.Page(page="pages/home.py", title="Home Page", icon=":material/globe:", default=True)

insights_page = st.Page(
    page="pages/insights.py", title="Key Insights", icon=":material/search_insights:"
)

dashboards_page = st.Page(
    page="pages/dashboards.py", title="Dashboards", icon=":material/analytics:"
)

faq_page = st.Page(page="pages/FAQ.py", title="Support/FAQ", icon=":material/help:")

disclaimer_page = st.Page(
    page="pages/disclaimer.py", title="Disclaimer", icon=":material/warning:"
)

terms_page = st.Page(page="pages/terms.py", title="Terms of Use", icon=":material/policy:")

## Navigation Setup
pg = st.navigation(
    {
        "Main": [home_page, insights_page, dashboards_page],
        "Help & Legal": [faq_page, disclaimer_page, terms_page],
    }
)

st.logo("assets/dummy-logo.png")
st.sidebar.text("Made with ❤️ by the ClimateLens Team")

pg.run()
