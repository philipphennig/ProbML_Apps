import streamlit as st

import jax
from jax import numpy as jnp
from jax import vmap, jit, jacrev, jacfwd, value_and_grad
from matplotlib import pyplot as plt

from tueplots import bundles
from tueplots.constants.color import rgb

from jax.scipy.stats import norm, beta

plt.rcParams.update(bundles.beamer_moml())
plt.rcParams.update({"figure.figsize": (6, 3)})

st.set_page_config(
    page_title="Lecture 03",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "(c) Philipp Hennig, 2023"},
)
jax.config.update("jax_enable_x64", True)

st.sidebar.header("Inference on Probabilities")

a0 = st.sidebar.slider(
    "prior success observations", min_value=0.1, max_value=1.0, value=1.0, step=0.1
)

b0 = st.sidebar.slider(
    "prior failure observations", min_value=0.1, max_value=1.0, value=1.0, step=0.1
)

a = st.sidebar.number_input(
    "number of successes",
    min_value=0,
    max_value=None,
    value=0,
    step=1,
    key="a_select",
)

b = st.sidebar.number_input(
    "number of failures",
    min_value=0,
    max_value=None,
    value=0,
    step=1,
    key="b_select",
)

a = a + a0
b = b + b0

Laplace = st.sidebar.checkbox(
    "show Laplace's approximation", value=False, key="laplace"
)

### plotting
x = jnp.linspace(0, 1, 200)
p = beta.pdf(x, a, b)
logp = beta.logpdf(x, a, b)

mean_p = a / (a + b)
var_p = a * b / (a + b) ** 2 / (a + b + 1)
std_p = jnp.sqrt(var_p)

fig, axs = plt.subplots(1, 2, sharex=True)
ax = axs[0]
ax.plot(x, p, label="$p(\pi\mid a,b)$", color=rgb.tue_red)
ax.fill_between(x, y1=0, y2=p, label="$p(\pi\mid a,b)$", alpha=0.2)
ax.axvline(mean_p, color=rgb.tue_green, label="mean")
ax.plot([mean_p - std_p, mean_p + std_p], [0.8, 0.8], "|-", label="std-deviation")

ax.axhline(1, linewidth=0.5, color=rgb.tue_dark)
ax.plot([0, 1], [0, 1], linewidth=0.5, color=rgb.tue_gray)
ax.plot([0, 1], [1, 0], linewidth=0.5, color=rgb.tue_gray)

if (a > 1) & (b > 1):
    mode_p = (a - 1) / (a + b - 2)
    # Laplace approximation:
    hess_logpmode = jax.hessian(lambda x: beta.logpdf(x, a, b))(mode_p)
    Laplace_std = jnp.sqrt(1.0 / -hess_logpmode)

    ax.plot(
        [mode_p, mode_p],
        [0, beta.pdf(mode_p, a, b)],
        "--",
        color=rgb.tue_red,
        label="mode",
    )

    if Laplace:
        ax.plot(
            x,
            norm.pdf(x, loc=mode_p, scale=Laplace_std),
            label="Laplace's approximation",
        )

ax.set_xlabel("$\pi$")
ax.set_ylabel("$p(\pi\mid a,b)$")
ax.set_xlim([0, 1])
ax.set_ylim([0, 5])
ax.legend(loc="upper right", fontsize="x-small")

# the log p plot on the right
if Laplace:
    ax = axs[1]
    ax.plot(x, logp, label="$\log p(\pi\mid a,b)$")
    ax.set_xlim([0, 1])
    ax.set_ylabel("$\log p(\pi\mid a,b)$")
    if (a > 1) & (b > 1):

        def quad(x):
            return beta.logpdf(mode_p, a, b) + 0.5 * (x - mode_p) ** 2 * hess_logpmode

        ax.axvline(mode_p, linestyle="--", color=rgb.tue_red, label="mode")
        ax.plot(x, quad(x), label="Laplace's approximation")

    ax.legend(loc="lower right", fontsize="x-small")

st.pyplot(fig)
