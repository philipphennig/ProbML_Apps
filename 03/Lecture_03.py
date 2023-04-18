import streamlit as st

import jax
from jax import numpy as jnp
from jax import vmap, jit, jacrev, jacfwd, value_and_grad
from matplotlib import pyplot as plt

from tueplots import bundles
from tueplots.constants.color import rgb

from jax.scipy.stats import norm, beta

plt.rcParams.update(bundles.beamer_moml())
plt.rcParams.update({"figure.figsize":(7,4)})

st.set_page_config(
    page_title="Lecture 03",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "(c) Philipp Hennig, 2023"},
)

