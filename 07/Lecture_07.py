import streamlit as st

import scipy.io
import functools
from matplotlib_inline.backend_inline import set_matplotlib_formats
from tueplots.constants.color import rgb
from tueplots import bundles
from matplotlib import pyplot as plt
import jax
import array_to_latex as a2l
import numpy as np
from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)
def ltx(a, fmt="{:6.5f}"):
    return a2l.to_ltx(np.array(a), frmt=fmt, print_out=False)

plt.rcParams.update(bundles.beamer_moml())

from gaussians import Gaussian

st.set_page_config(
    page_title="Lecture 06",
    page_icon="🧊",
    initial_sidebar_state="expanded",
    menu_items={"About": "(c) Philipp Hennig, 2023"},
)

# st.sidebar.title("Lecture 06: Parametric Regression")

st.markdown("Sampling from a Gaussian distribution (given [standard Gaussian](https://upload.wikimedia.org/wikipedia/commons/1/1f/Box-Muller\_transform\_visualisation.svg) RVs)")

M0 = st.sidebar.slider("Mu_0", min_value=-2., max_value=2., value=0., step=.1)
M1 = st.sidebar.slider("Mu_1", min_value=-2., max_value=2., value=0., step=.1)

S1 = st.sidebar.slider("Sigma_11", min_value=0., max_value=10., value=1., step=.1)
S2 = st.sidebar.slider("Sigma_22", min_value=0., max_value=10., value=1., step=.1)
rho = st.sidebar.slider("rho", min_value=-1., max_value=1., value=0., step=.1)

N = st.sidebar.slider("Number of samples", min_value=1, max_value=100, value=10, step=1)

method = st.sidebar.radio("Sampling method", options=["Cholesky", "Eigenvalue"])

st.markdown("$$p(x) = \mathcal{N}\\left(\\begin{pmatrix} x_1 \\\\ x_2 \\end{pmatrix}; \\begin{pmatrix}" + f"{M0:.01f}" + "\\\\" + f"{M1}" + "\\end{pmatrix}, \\begin{pmatrix} " + f"{S1}" + "&" + f"{rho * jnp.sqrt(S1*S2):.01f}" + "\\\\" + f"{rho * jnp.sqrt(S1*S2):.01f}" + "&" + f"{S2}" + " \\end{pmatrix}\\right)$$")

mu = jnp.asarray([M0,M1])
Sigma = jnp.asarray([[S1,rho],[rho,S2]])

circle = jnp.stack(
    [jnp.cos(jnp.linspace(0, 2 * jnp.pi, 100)),
     jnp.sin(jnp.linspace(0, 2 * jnp.pi, 100))]
)



key = jax.random.PRNGKey(0)
raw = jax.random.normal(key, shape=(N, 2))

if method == "Cholesky":
    L = jnp.linalg.cholesky(Sigma)
    transformed = raw @ L.T 
    transformed_circle = circle.T @ L.T

    st.markdown("$ L = " + ltx(L,fmt="{:6.2f}") + " $")

    st.markdown("""
    ```python
    L = jnp.linalg.cholesky(Sigma)
    key = jax.random.PRNGKey(0)
    samples = jnp.random.normal(key, shape=(N, 2)) @ L.T + mu
    ```
    """)
    
elif method == "Eigenvalue":
    D, V = jnp.linalg.eigh(Sigma)
    transformed = raw @ jnp.diag(jnp.sqrt(D)) @ V.T
    transformed_circle = circle.T @ jnp.diag(jnp.sqrt(D)) @ V.T

    st.markdown("$ D = " + ltx(D,fmt="{:6.2f}") + " $")

    st.markdown("$ V = " + ltx(V,fmt="{:6.2f}") + " $")

    st.markdown("""
    ```python
    D, V = jnp.linalg.eigh(Sigma)
    key = jax.random.PRNGKey(0)
    samples = jnp.random.normal(key, shape=(N, 2)) @ jnp.diag(jnp.sqrt(D)) @ V.T + mu
    ```
    """)

shifted = transformed + mu
shifted_circle = transformed_circle + mu

fig,ax = plt.subplots(figsize=(4,4))
ax.plot(circle[0,:], circle[1,:], color=rgb.tue_dark, label="unit circle")
ax.plot(2 * circle[0,:], 2 * circle[1,:], color=rgb.tue_dark, alpha=0.5)
ax.plot(transformed_circle[:,0], transformed_circle[:,1], color=rgb.tue_blue, label="transformed circle")
ax.plot(2 * transformed_circle[:,0], 2 * transformed_circle[:,1], color=rgb.tue_blue, alpha=0.5)
ax.plot(shifted_circle[:,0], shifted_circle[:,1], color=rgb.tue_red, label="shifted circle")
ax.plot((2 * transformed_circle + jnp.asarray([M0,M1]))[:,0], (2 * transformed_circle + jnp.asarray([M0,M1]))[:,1], color=rgb.tue_red, alpha=0.5)

ax.scatter(raw[:,0], raw[:,1], color=rgb.tue_dark, s=10, label="raw samples")
ax.scatter(transformed[:,0], transformed[:,1], color=rgb.tue_blue, s=10, label="transformed samples")
ax.scatter(shifted[:,0], shifted[:,1], color=rgb.tue_red, s=10, label="shifted samples")

ax.set_aspect("equal")
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
ax.legend()
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")

st.pyplot(fig)