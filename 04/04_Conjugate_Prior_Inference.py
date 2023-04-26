import streamlit as st

import jax
from jax import numpy as jnp
from jax import vmap, jit, jacrev, jacfwd, value_and_grad
from matplotlib import pyplot as plt

from tueplots import bundles
from tueplots.constants.color import rgb

from jax.scipy.stats import norm, beta

st.set_page_config(
    page_title="Lecture 04",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "(c) Philipp Hennig, 2023"},
)


'''A conjugate prior to the likelihood $p(x\mid D)=\ell(D;x)$ is a probability measure with pdf $p(x)=g(x;\\theta)$, such that

$$ p(x\mid D) = \\frac{\ell(D;x)g(x;\\theta)}{\int \ell(D;x) g(x;\\theta) dx} = g(x; \\theta + \phi(D)) $$ 

'''