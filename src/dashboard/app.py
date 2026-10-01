import streamlit as st


st.set_page_config(
    page_title="AI Traffic Digital Twin",
    layout="wide",
)


st.title("AI-Powered Indian Urban Traffic Digital Twin")

st.subheader("System Status")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Phase", "0")

with col2:
    st.metric("Junctions", "4")

with col3:
    st.metric("Digital Twin", "Offline")

with col4:
    st.metric("AI Controller", "Offline")


st.divider()

st.info(
    "Project foundation initialized. "
    "CCTV, YOLO, tracking, traffic-state, prediction, "
    "SUMO and signal-control modules will be added in later phases."
)