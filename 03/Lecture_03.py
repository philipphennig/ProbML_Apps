import streamlit as st

from matplotlib import pyplot as plt

from tueplots import bundles

plt.rcParams.update(bundles.beamer_moml())

st.set_page_config(
    page_title="Lecture 03",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "(c) Philipp Hennig, 2023"},
    primaryColor="#A51E37",
    backgroundColor="#FFFFFF",
    secondaryBackgroundColor="#E1D7C1",
    textColor="#32414B",
    font="sans serif",
)
