import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
)

pages = {
    "🏠 Home": [
        st.Page("pages/home.py"),
    ],
}


pg = st.navigation(pages)
pg.run()