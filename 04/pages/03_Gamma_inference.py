import streamlit as st

from jax import vmap
from jax import random
from jax import numpy as jnp
from matplotlib import pyplot as plt

from scipy.stats import gamma
from scipy.special import gamma as gamma_func
from jax.scipy.stats import norm

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

st.sidebar.header("Inference on the variance (actually, the precision) of a Gaussian distribution")

N = st.sidebar.slider("N", min_value=0, max_value=100, step=1)

sigma = st.sidebar.slider("sigma", min_value=0.01, max_value=5.0, value=1.0, step=0.1)

cp = st.sidebar.checkbox("Conjugate Prior?", value=False, key="cp")

if cp:
    alpha = st.sidebar.slider(
        "alpha", min_value=0.01, max_value=2.0, value=1.0, step=0.1
    )

    beta = st.sidebar.slider("beta", min_value=0.01, max_value=2.0, value=1.0, step=0.1)


key = random.PRNGKey(0)
sp = jnp.linspace(0.01, 10, 250)
if N > 0:
    X = random.normal(key, shape=(N, 1)) * sigma
    counts, bins = jnp.histogram(X, density=True)

    def loglik(x, m, s):
        return -0.5 * (x - m) ** 2 / s**2 - jnp.log(s)

    LL = vmap(lambda s: (vmap(lambda x: loglik(x, 0, s))(X)))(sp)
    L = LL.sum(axis=1) / N
else:
    X = jnp.array([])


fig, axs = plt.subplots(1, 2)
xp = jnp.linspace(-3 * sigma, 3 * sigma, 250)
if N > 0:
    axs[0].bar(bins[:-1], counts, width=0.8 * (bins[1] - bins[0]), alpha=0.3,label="data")
    axs[0].set_ylim(0, .7)
    axs[0].plot(xp, norm.pdf(xp, 0, sigma), color=rgb.tue_red, label="$p_{true}$")

axs[0].set_xlabel("$x$")
axs[0].set_ylabel("count")
axs[0].set_title("observed")
axs[1].set_title("latent")

if N > 0:
    axs[1].set_ylim(0, .3)
    axs[1].axvline(sigma, color=rgb.tue_dark, label="true $\sigma$")
    axs[1].plot(sp, jnp.exp(L), color=rgb.tue_gray, label="$\hat{p}(x\mid \sigma)$")
    axs[1].set_xlabel("$\sigma$")
    axs[1].set_ylabel("$p$")
    axs[1].legend()

if cp:
    prior = gamma(a=alpha, scale=1 / beta)
    alphapost = alpha + N / 2
    betapost = beta + 0.5 * (X**2).sum()
    posterior = gamma(a=alphapost, scale=1 / betapost)

    def predictive(x):
        constants = 1 / jnp.sqrt(betapost * 2 * jnp.pi) * gamma_func(alphapost+0.5) / gamma_func(alphapost)
        return constants * (1 + x**2 / (2 * betapost))**(-alphapost-0.5)

    axs[0].plot(xp, predictive(xp), color=rgb.tue_gold, label="predictive")

    axs[1].plot(sp,prior.pdf(1/sp**2) / (2*sp), color=rgb.tue_gold, label="prior $p(\sigma)$")
    axs[1].plot(sp, posterior.pdf(1/sp**2) / (2*sp), color=rgb.tue_red, label="posterior $p(\sigma\mid x)$")
    axs[1].legend()

axs[0].legend()

st.pyplot(fig)
