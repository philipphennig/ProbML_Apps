import streamlit as st

import scipy.io
import functools
from matplotlib_inline.backend_inline import set_matplotlib_formats
from tueplots.constants.color import rgb
from tueplots import bundles
from matplotlib import pyplot as plt
import jax
import dataclasses
from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)

plt.rcParams.update(bundles.beamer_moml())

from gaussians import Gaussian

data = scipy.io.loadmat("lindata.mat")
X = data["X"]  # inputs
Y = data["Y"][:, 0]  # outputs
sigma = data["sigma"][0].flatten()
N = X.shape[0]

st.set_page_config(
    page_title="Lecture 06",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "(c) Philipp Hennig, 2023"},
)

st.sidebar.title("Lecture 06: Parametric Regression")

M0 = st.sidebar.slider("Mu_0", min_value=-2., max_value=2., value=0., step=.1)
M1 = st.sidebar.slider("Mu_1", min_value=-2., max_value=2., value=0., step=.1)

S1 = st.sidebar.slider("Sigma_11", min_value=0., max_value=10., value=1., step=.1)
S2 = st.sidebar.slider("Sigma_22", min_value=0., max_value=10., value=1., step=.1)
rho = st.sidebar.slider("rho", min_value=-1., max_value=1., value=0., step=.1)

select_data = st.sidebar.multiselect(
    "Select data",
    options = jnp.arange(N),
    default = [i for i in range(N)]
    )

X_select = X[select_data]
Y_select = Y[select_data]

prior = Gaussian(mu=jnp.asarray([M0,M1]), Sigma=jnp.asarray([[S1,rho],[rho,S2]]))
phi = lambda x: jnp.hstack([jnp.ones_like(x), x])
x = jnp.linspace(-5, 5, 100)[:, None]
f_prior = phi(x) @ prior

std_f = f_prior.std

key = jax.random.PRNGKey(0)
w_samples = prior.sample(key, num_samples=3)

likelihood = Gaussian(mu = jnp.zeros(2), Sigma = 10e3**2 * jnp.eye(2)).condition(phi(X_select), Y_select, sigma**2 * jnp.eye(len(select_data)))
posterior = prior.condition(phi(X_select), Y_select, sigma**2 * jnp.eye(len(select_data)))
f_posterior = phi(x) @ posterior

st.markdown('''
```python
prior = Gaussian(mu=jnp.asarray([Mu_0,Mu_1]), Sigma=jnp.asarray([[Sigma_11,rho * jnp.sqrt(Sigma_11 * Sigma_22)],[rho * jnp.sqrt(Sigma_11 * Sigma_22),Sigma_22]]))
phi = lambda x: jnp.hstack([jnp.ones_like(x), x])
posterior = prior.condition(phi(X), Y, sigma**2 * jnp.eye(X.size)))))
x = jnp.linspace(-5, 5, 100)[:, None]
f_prior = phi(x) @ prior
f_posterior = phi(x) @ posterior
```
''')

fig, axs = plt.subplots(1, 2)

ax = axs[0]
ax.plot(prior.mu[0], prior.mu[1], "o", color=rgb.tue_gray, ms = 4 ,label="prior")
ax.set_xlim([-3,3])
ax.set_ylim([-3,3])

circle = jnp.linspace(0, 2 * jnp.pi, 100)
circle = jnp.stack([jnp.cos(circle), jnp.sin(circle)], axis=1)
prior_cov = prior.L @ circle.T
for i in range(1,3):
    ax.plot(
        prior.mu[0] + i * prior_cov[0, :],
        prior.mu[1] + i * prior_cov[1, :],
        color=rgb.tue_gray,
        alpha=0.5,
        linewidth= 1 / i,
    )

ax.plot(w_samples[:, 0], w_samples[:, 1], "o", color=rgb.tue_gray, ms=2, alpha=1.0, mec="none")

if len(select_data) > 0:

    likelihood_cov = likelihood.L @ circle.T
    ax.plot(likelihood.mu[0], likelihood.mu[1], "o", color=rgb.tue_blue, ms = 4, label="likelihood")
    for i in range(1,3):
        ax.plot(
            likelihood.mu[0] + i * likelihood_cov[0, :],
            likelihood.mu[1] + i * likelihood_cov[1, :],
            color=rgb.tue_blue,
            alpha=0.5,
            linewidth= 1 / i,
        )


    ax.plot(posterior.mu[0], posterior.mu[1], "o", color=rgb.tue_red, ms = 4, label="posterior")
    posterior_cov = posterior.L @ circle.T
    for i in range(1,3):

        ax.plot(
            posterior.mu[0] + i * posterior_cov[0, :],
            posterior.mu[1] + i * posterior_cov[1, :],
            color=rgb.tue_red,
            alpha=0.5,
            linewidth= 1 / i,
        )

    w_post = posterior.sample(key, num_samples=3)
    ax.plot(w_post[:, 0], w_post[:, 1], "o", color=rgb.tue_red, ms=2, alpha=1.0, mec="none")



ax.set_xlabel("$w_0$")
ax.set_ylabel("$w_1$")

ax.legend(loc="lower left")

ax = axs[1]
ax.errorbar(
    X, Y, yerr=sigma * jnp.ones_like(Y), fmt="o", ms=2, color=rgb.tue_dark
)
ax.plot(x, f_prior.mu, color=rgb.tue_gray, label="prior")
ax.fill_between(
    x[:, 0],
    f_prior.mu - 2 * std_f,
    f_prior.mu + 2 * std_f,
    color=rgb.tue_gray,
    alpha=0.5,
)
ax.plot(x, phi(x) @ w_samples.T, color=rgb.tue_gray, alpha=0.4)


if len(select_data) > 0:
    ax.errorbar(
        X_select, Y_select, yerr=sigma * jnp.ones_like(Y_select), fmt="o", ms=2, color=rgb.tue_red
    )

    likelihood_x = phi(x) @ likelihood
    ax.plot(x, likelihood_x.mu, color=rgb.tue_blue, label="likelihood")
    ax.fill_between(
        x[:, 0],
        likelihood_x.mu - 2 * likelihood_x.std,
        likelihood_x.mu + 2 * likelihood_x.std,
        color=rgb.tue_blue,
        alpha=0.2,
    )

    ax.plot(x, f_posterior.mu, color=rgb.tue_red, label="posterior")
    ax.fill_between(
        x[:, 0],
        f_posterior.mu - 2 * f_posterior.std,
        f_posterior.mu + 2 * f_posterior.std,
        color=rgb.tue_red,
        alpha=0.5,
    )

    ax.plot(x, phi(x) @ posterior.sample(key, num_samples=3).T, color=rgb.tue_red, alpha=0.4)


ax.set_ylim([-10,10])
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

ax.legend(loc="upper left")

st.pyplot(fig)