import streamlit as st

from jax import numpy as jnp
from matplotlib import pyplot as plt

from tueplots import bundles
from tueplots.constants.color import rgb

from jax.scipy.stats import beta, betabinom
from scipy.stats import bernoulli

plt.rcParams.update(bundles.beamer_moml())
plt.rcParams.update({"figure.figsize": (6, 3)})

st.set_page_config(
    page_title="Conjugate Priors",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "(c) Philipp Hennig, 2023"},
)

st.sidebar.header("Inference on Binary Probabilities")

a0 = st.sidebar.slider("alpha", min_value=0.1, max_value=1.0, value=1.0, step=0.1)

b0 = st.sidebar.slider("beta", min_value=0.1, max_value=1.0, value=1.0, step=0.1)

N = st.sidebar.slider("N", min_value=0, max_value=100, step=1)

ftrue = st.sidebar.slider("f_true", min_value=0.0, max_value=1.0, step=0.1, value=0.5)


r = bernoulli.rvs(ftrue, size=N, random_state=1)
a = r.sum()
b = N - a

y = jnp.arange(N + 1)
predictive = betabinom.pmf(y, N, a + a0, b + b0)


# plotting
x = jnp.linspace(0, 1, 200)
p = beta.pdf(x, a + a0, b + b0)

fig, axs = plt.subplots(1, 2)
axs[0].bar([0, 1], [b, a])
axs[0].set_xticks([0, 1])
axs[0].set_xticklabels(["$n_0$ (failures)", "$n_1$ (successes)"])
axs[0].barh(y, left=1, width=predictive, height=0.8, color=rgb.tue_gold, alpha=0.7)
axs[0].barh(
    y,
    left=0,
    width=predictive[::-1],
    height=0.8,
    color=rgb.tue_gold,
    alpha=0.7,
    label="posterior predictive",
)
axs[0].axhline(N, color=rgb.tue_dark)
axs[0].axvline(0, color=rgb.tue_dark, lw=0.5)
axs[0].axvline(1, color=rgb.tue_dark, lw=0.5)
axs[0].axhline(N * ftrue, color=rgb.tue_dark, lw=0.5)
axs[0].set_xlim([-1, 2])
axs[0].set_title("observations")
axs[0].legend(loc="lower right", fontsize="x-small", framealpha=0.7, facecolor="white")

axs[1].plot(x, p, label="posterior $p(f\mid n_0,n_1)$", color=rgb.tue_red)
axs[1].fill_between(x, y1=0, y2=p, alpha=0.2)

from scipy.special import binom
likelihood = binom(N, a) * x**a * (1 - x) ** b

axs[1].plot(x, likelihood, label="likelihood $p(x\mid f)$", color=rgb.tue_blue)
axs[1].axvline(ftrue, linewidth=0.5, color=rgb.tue_dark, label="$f_{true}$")

axs[1].set_xlabel("$f$")
axs[1].set_ylabel("$p$")
axs[1].set_xlim([0, 1])
# ax.set_ylim([0, 5])
axs[1].legend(loc="upper right", fontsize="x-small")
axs[1].set_title("latent")

st.markdown(
    """
    $$
    p(x\mid f) = \prod_{i=1} ^N f^{\: x} \cdot(1-f)^{1-x} = 
    \\begin{pmatrix} N \\\\ n_1 \\end{pmatrix}
    f^{\: n_1} \cdot (1-f)^{n_0} \qquad n_0 := N - n_1 \qquad x\in \{0;1\}
    $$
    
    $$
    p(f\mid \\alpha,\\beta) = \mathcal{B}(\\alpha,\\beta) =\\frac{1}{B(\\alpha,\\beta)} f^{\\alpha-1}(1-f)^{\\beta-1}
    $$
    
    $$
    p(f\mid x, \\alpha,\\beta) = \mathcal{B}(\\alpha + n_1,\\beta + n_0) = \\frac{1}{B(\\alpha + n_1,\\beta + n_0)} f^{\\alpha + n_1 -1}(1-f)^{\\beta + n_0 -1}
    $$
"""
)

st.pyplot(fig)
