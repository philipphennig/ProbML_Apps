import streamlit as st

from jax import vmap
from jax import random
from jax import numpy as jnp
from matplotlib import pyplot as plt

from scipy.stats import norm

from tueplots import bundles
from tueplots.constants.color import rgb

plt.rcParams.update(bundles.beamer_moml())
plt.rcParams.update({"figure.figsize": (6, 3)})

st.set_page_config(
    page_title="Conjugate Priors",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "(c) Philipp Hennig, 2023"},
)

st.sidebar.header("Inference on the mean of a Gaussian distribution")

N = st.sidebar.slider("N", min_value=0, max_value=100, step=1)

mu = st.sidebar.slider("mu", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)

cp = st.sidebar.checkbox("Conjugate Prior?", value=False, key="cp")

if cp:
    m = st.sidebar.slider("m", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)

    v = st.sidebar.slider("v", min_value=0.01, max_value=10.0, value=1.0, step=0.1)

key = random.PRNGKey(0)
mp = jnp.linspace(-3, 3, 250)
if N > 0:
    X = random.normal(key, shape=(N, 1)) * 1.0 + mu
    counts, bins = jnp.histogram(X, density=True)

    def loglik(x, m, s):
        return -0.5 * (x - m) ** 2 / s**2 - jnp.log(s)

    LL = vmap(lambda m: (vmap(lambda x: loglik(x, m, 1.0))(X)))(mp)
    L = LL.sum(axis=1) / N
else:
    X = jnp.array([])

fig, axs = plt.subplots(1, 2)
axs[0].set_title("observed")
axs[1].set_title("latent")
xp = jnp.linspace(-5, 5, 250)
if N > 0:
    axs[0].bar(
        bins[:-1], counts, width=0.8 * (bins[1] - bins[0]), alpha=0.3, label="data"
    )
    axs[0].set_ylim(0, 0.7)
    axs[0].plot(xp, norm.pdf(xp, mu, 1.0), color=rgb.tue_red, label="$p_{true}$")

axs[0].set_xlabel("$x$")
axs[0].set_ylabel("count")

if N > 0:
    axs[1].set_ylim(0, 2)
    axs[1].axvline(mu, color=rgb.tue_dark, label="true $\mu$")
    axs[1].plot(mp, jnp.exp(L), color=rgb.tue_gray, label="$\hat{p}(x\mid \mu)$")
    axs[1].set_xlabel("$\mu$")
    axs[1].set_ylabel("$p$")
    axs[1].legend()

if cp:
    prior_prec = 1 / v
    prior_meanprec = m / v
    prior = norm.pdf(mp, m, jnp.sqrt(v))
    posterior_prec = prior_prec + N
    posterior_meanprec = prior_meanprec + X.sum()
    posterior = norm.pdf(
        mp, posterior_meanprec / posterior_prec, jnp.sqrt(1 / posterior_prec)
    )

    def predictive(x):
        return norm.pdf(
            x, posterior_meanprec / posterior_prec, jnp.sqrt(1 / posterior_prec) + 1.0
        )

    axs[0].plot(xp, predictive(xp), color=rgb.tue_gold, label="predictive")

    axs[1].plot(mp, prior, color=rgb.tue_gold, label="prior $p(\mu)$)")
    axs[1].plot(mp, posterior, color=rgb.tue_red, label="posterior $p(\mu\mid x)$")
    axs[1].legend()

axs[0].legend()

st.pyplot(fig)
