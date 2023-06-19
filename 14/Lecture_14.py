import streamlit as st

import functools
from matplotlib_inline.backend_inline import set_matplotlib_formats
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import ticker
from tueplots.constants.color import rgb
from tueplots import bundles
from matplotlib import pyplot as plt
import jax
from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)
plt.rcParams.update(bundles.beamer_moml())

cmap_rw = LinearSegmentedColormap.from_list(
    "rw", [(1, 1, 1), rgb.tue_red], N=1024)
cmap_dw = LinearSegmentedColormap.from_list(
    "dw", [(1, 1, 1), rgb.tue_dark], N=1024)
cmap_bw = LinearSegmentedColormap.from_list(
    "bw", [(1, 1, 1), rgb.tue_blue], N=1024)
cmap_gw = LinearSegmentedColormap.from_list(
    "gw", [(1, 1, 1), rgb.tue_green], N=1024)
cmap_bwr = LinearSegmentedColormap.from_list(
    "bwr", [rgb.tue_blue, (1, 1, 1), rgb.tue_red], N=1024
)

from gaussians import *

st.set_page_config(
    page_title="Lecture 14",
    page_icon="🧊",
    initial_sidebar_state="expanded",
    menu_items={"About": "(c) Philipp Hennig, 2023"},
)

st.sidebar.title("Lecture 14: Logistic regression")

M = st.sidebar.slider("mu(x)", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
theta = st.sidebar.slider("theta", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
ell = st.sidebar.slider("ell", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

num_samples = st.sidebar.slider("number of samples", min_value=1, max_value=20, value=3, step=1)

key = jax.random.PRNGKey(0)

import functools


def constant_mean(x, c=0.0):
    return c * jnp.ones_like(x[:, 0])


def RQ_kernel(a, b, theta=1.0, ell=1.0, alpha=1.0):
    return theta**2 * (
        1 + jnp.sum((a - b) ** 2, axis=-1) / (2 * alpha * ell**2)
    ) ** (-alpha)


st.markdown(
    """
    To compute the pdf p(y=1), we remember that
    $$
    p_y(y(x)) = p_x(x(y))  \\left|\\frac{dx}{dy}\\right|
    $$
    Here, $y(x) = \sigma(f(x)) = 1/(1+e^{-f(x)})$. The inverse logistic is
    $$
    x(y) = \log(y/(1-y))
    $$
    with derivative $dx/dy = 1/(y(1-y))$.
    """
)

# Define mean, kernel and prior GP for f
mean = functools.partial(constant_mean, c=M)
kernel = functools.partial(RQ_kernel, theta=theta, ell=ell)
prior = GaussianProcess(mean, kernel)
logistic = lambda x: 1 / (1 + jnp.exp(-x))

fig, axs = plt.subplots(3, 1, figsize=(4, 3), sharex=True)

axs[1].axhline(0.5, color=rgb.tue_dark, lw=0.5)
axs[2].axhline(0, color=rgb.tue_dark, lw=0.5)
axs[0].yaxis.set_major_locator(ticker.MultipleLocator(1))

x = jnp.linspace(-5, 5, 400)[:, None]
xs = jnp.vstack([x, jnp.arange(5)[:,None]])


prior_x = prior(x)
samples = prior(xs).sample(key, num_samples=num_samples)
# st.text(samples.shape)
u = jax.random.uniform(key, shape=(5,), minval=0, maxval=1)
X_samples = samples[:,-5:]
p_samples = logistic(X_samples)
samples = samples[:,:-5]

for s in range(num_samples):
    for i in jnp.arange(5):
        if p_samples[s, i] > u[i]:
            axs[0].plot(i+(s-(num_samples-1)/2) * 0.4/num_samples, 1, "o", color=rgb.tue_dark, ms=2)
        else:
            axs[0].plot(i+(s-(num_samples-1)/2) * 0.4/num_samples, 0, "o", color=rgb.tue_dark, ms=2)

# computing the density on the output space. Remember 
# p_y(y(x)) = p_x(x(y)) * |dx/dy|
# here, y(x) = logistic(x). The inverse logistic is
# x(y) = log(y/(1-y)), with derivative dx/dy = 1/(y*(1-y))
pyy = jnp.linspace(0, 1, 1000)[:,None]
def logit(y):
    return jnp.log(y/(1-y))
PY = gp_shading(logit(pyy), prior_x.mu, prior_x.std) / (pyy * (1-pyy))

for ax in axs:
    for i in jnp.arange(5):
        ax.axvline(i, color=rgb.tue_dark, lw=0.5)
    
axs[1].plot(jnp.arange(5), u, "o", color=rgb.tue_green, ms=4, label="u")
axs[1].plot(jnp.arange(5), p_samples.T, "o", color=rgb.tue_blue, ms=2)

ax = axs[1]
ax.imshow(PY, extent=[x[0, 0], x[-1, 0], 0, 1],
            cmap=cmap_bw,
            alpha=0.5,
            aspect="auto",
            origin="lower"
        )

ax.plot(x, logistic(prior_x.mu), color=rgb.tue_blue, label="$p(\sigma(f))$")
ax.plot(x, logistic(samples.T), color=rgb.tue_blue, alpha=0.4)

ax.set_ylabel("$p(y=1)$")
ax.set_ylim(0, 1)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
ax.legend(loc="upper left", facecolor="white", framealpha=1.0)

ax = axs[2]
prior.plot_shaded(
    ax,
    x,
    color=rgb.tue_red,
    yrange=(-3, 3),
    yres=1000,
    mean_kwargs={"label": "$p(f)=\mathcal{GP}$"},
    std_kwargs={"alpha": 0.5, "cmap": cmap_rw},
    num_samples=0,
    rng_key=key,
)

ax.plot(x, samples.T, color=rgb.tue_red, alpha=0.4)

ax.set_xlabel("$x$")
ax.set_ylabel("$f(x)$")
ax.set_ylim(-3, 3)
ax.set_xlim(-5, 5)
ax.legend(loc="upper left", facecolor="white", framealpha=1.0)

ax.yaxis.set_major_locator(ticker.MultipleLocator(1))


st.pyplot(fig)
