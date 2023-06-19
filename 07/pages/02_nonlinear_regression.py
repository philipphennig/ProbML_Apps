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

data = scipy.io.loadmat("nlindata.mat")
X = data["X"]  # inputs
Y = data["Y"][:, 0]  # outputs
N = X.shape[0]
sigma = data["sigma"][0].flatten()

st.set_page_config(
    page_title="Lecture 06",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "(c) Philipp Hennig, 2023"},
)

# st.sidebar.title("Parametric Regression")

st.markdown(
    """
    $$ p(w) = \mathcal{N}(w\\mid 0, \\theta^2/F \cdot \mathbf{I}) \qquad f(x)=\phi(x)^T w \qquad  p(y \\mid w) = \\prod_i \mathcal{N}(y_i|\phi(x_i)^T w,\\sigma^2 ) $$
    """
)

### A bunch of features:
def polynomial_features(x, num_features=2):
    """Feature map for a polynomial basis function."""
    # output shape: (n_samples, order)
    return (
        x ** jnp.arange(num_features)
        / jnp.exp(jax.scipy.special.gammaln(jnp.arange(num_features) + 1))
        / jnp.sqrt(num_features)
    )


def gaussian_features(x, num_features=2, sigma=1.0):
    """Feature map for a Gaussian basis function."""
    # output shape: (n_samples, order)
    return jnp.exp(
        -(((x - jnp.linspace(-8, 8, num_features)) / sigma) ** 2)
    ) / jnp.sqrt(num_features)


def relu_features(x, num_features=2):
    """Feature map for a ReLU basis function."""
    # output shape: (n_samples, order)
    return jnp.maximum(
        0,
        (jnp.sign(jnp.arange(0, num_features) % 2 - 0.5))
        * (x - jnp.linspace(-8, 8, num_features)),
    ) / jnp.sqrt(num_features)

def one_sided_relu(x, num_features=2):
    """Feature map for a ReLU basis function."""
    # output shape: (n_samples, order)
    return jnp.maximum(
        0,(x - jnp.linspace(-8, 8, num_features)),
    ) / jnp.sqrt(num_features)

def cosine_features(x, num_features=2, ell=1.0):
    """Feature map for a cosine basis function."""
    # output shape: (n_samples, order)
    return jnp.cos(jnp.pi * x / jnp.arange(1, num_features) / ell) / jnp.sqrt(
        num_features
    )


def trig_features(x, num_features=2, ell=1.0):
    """Feature map for a combination of cosine and sine basis functions."""
    # output shape: (n_samples, order)
    return jnp.hstack(
        [
            jnp.cos(jnp.pi * x / jnp.arange(1, jnp.floor(num_features / 2.0)) / ell),
            jnp.sin(jnp.pi * x / jnp.arange(1, jnp.ceil(num_features / 2.0)) / ell),
        ]
    ) / jnp.sqrt(num_features)


def sigmoid_features(x, num_features=2, ell=1.0):
    """Feature map for a sigmoid basis function."""
    # output shape: (n_samples, order)
    return (
        1
        / (1 + jnp.exp(-(x - jnp.linspace(-8, 8, num_features)) / ell))
        / jnp.sqrt(num_features)
    )


def step_features(x, num_features=2):
    """Feature map for a step basis function."""
    # output shape: (n_samples, order)
    return jnp.sign(x - jnp.linspace(-8, 8, num_features)) / jnp.sqrt(num_features)


def switch_features(x, num_features=2):
    """Feature map for a switch basis function."""
    # output shape: (n_samples, order)
    return (x > jnp.linspace(-8, 8, num_features)) / jnp.sqrt(num_features)


select_features = st.sidebar.selectbox(
    "Select features",
    [
        "Polynomial",
        "Gaussian",
        "Switch",
        "ReLU",
        "one-sided ReLU",
        "Step",
        "Cosine",
        "Trig",
        "Sigmoid",
    ],
)

n_features = select_num_features = st.sidebar.slider(
    "Select number of features",
    min_value=2,
    max_value=60,
    value=2,
    step=1,
)

theta = st.sidebar.select_slider(
    "Select theta",
    options=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0],
    value=1.0,
)

if select_features == "Polynomial":
    phi = functools.partial(polynomial_features, num_features=n_features)
    st.markdown(
        '''
    ```python
    def polynomial_features(x, num_features=2):
        """Feature map for a polynomial basis function."""
        # output shape: (n_samples, order)
        return x ** jnp.arange(num_features) / jnp.exp(jax.scipy.special.gammaln(jnp.arange(num_features)+1)) / jnp.sqrt(num_features)
    ```
    '''
    )
 
elif select_features == "Gaussian":
    ell = st.sidebar.slider(
        "Select length-scale for Gaussian features",
        min_value=0.05,
        max_value=5.0,
        value=1.0,
        step=0.1,
    )
    phi = functools.partial(gaussian_features, num_features=n_features, sigma=ell)
    st.markdown(
        '''
    ```python
    def gaussian_features(x, num_features=2, sigma=1.0):
        """Feature map for a Gaussian basis function."""
        # output shape: (n_samples, order)
        return jnp.exp(-(((x - jnp.linspace(-8, 8, num_features)) / sigma) ** 2)) / jnp.sqrt(num_features)
    ```
    '''
    )
elif select_features == "ReLU":
    phi = functools.partial(relu_features, num_features=n_features)
    st.markdown(
        '''
    ```python
    def relu_features(x, num_features=2):
        """Feature map for a ReLU basis function."""
        # output shape: (n_samples, order)
        return jnp.maximum(
            0,
            (jnp.sign(jnp.arange(0, num_features) % 2 - 0.5))
            * (x - jnp.linspace(-8, 8, num_features)),
        ) / jnp.sqrt(num_features)
    ```
    '''
    )

elif select_features == "one-sided ReLU":
    phi = functools.partial(one_sided_relu, num_features=n_features)
    st.markdown(
        '''
    ```python
    def one_sided_relu(x, num_features=2):
        """Feature map for a ReLU basis function."""
        # output shape: (n_samples, order)
        return jnp.maximum(
            0,(x - jnp.linspace(-8, 8, num_features)),
        ) / jnp.sqrt(num_features)
    ```
    '''
    )

elif select_features == "Cosine":
    ell = st.sidebar.slider(
        "Select length-scale for cosine features",
        min_value=0.01,
        max_value=5.0,
        value=1.0,
        step=0.1,
    )
    phi = functools.partial(cosine_features, num_features=n_features, ell=ell)
    st.markdown(
        '''
    ```python
    def cosine_features(x, num_features=2, ell=1.0):
        """Feature map for a cosine basis function."""
        # output shape: (n_samples, order)
        return jnp.cos(jnp.pi * x / jnp.arange(1, num_features) / ell) / jnp.sqrt(num_features)
    ```
    '''
    )
elif select_features == "Trig":
    ell = st.sidebar.slider(
        "Select length-scale for trig features",
        min_value=0.01,
        max_value=5.0,
        value=1.0,
        step=0.1,
    )
    phi = functools.partial(trig_features, num_features=n_features, ell=ell)
    st.markdown(
        '''
    ```python
    def trig_features(x, num_features=2, ell=1.0):
        """Feature map for a combination of cosine and sine basis functions."""
        # output shape: (n_samples, order)
        return jnp.hstack(
            [
                jnp.cos(jnp.pi * x / jnp.arange(1, jnp.floor(num_features / 2.0)) / ell),
                jnp.sin(jnp.pi * x / jnp.arange(1, jnp.ceil(num_features / 2.0)) / ell),
            ]
        ) / jnp.sqrt(num_features)
    ```
    '''
    )

elif select_features == "Sigmoid":
    ell = st.sidebar.slider(
        "Select length-scale for sigmoid features",
        min_value=0.01,
        max_value=5.0,
        value=1.0,
        step=0.1,
    )
    phi = functools.partial(sigmoid_features, num_features=n_features, ell=ell)
    st.markdown(
        '''
    ```python
    def sigmoid_features(x, num_features=2, ell=1.0):
        """Feature map for a sigmoid basis function."""
        # output shape: (n_samples, order)
        return (
            1
            / (1 + jnp.exp(-(x - jnp.linspace(-8, 8, num_features)) / ell))
            / jnp.sqrt(num_features)
        )
    ```
    '''
    )
elif select_features == "Step":
    phi = functools.partial(step_features, num_features=n_features)
    st.markdown(
        '''
    ```python
    def step_features(x, num_features=2):
        """Feature map for a step basis function."""
        # output shape: (n_samples, order)
        return (x > jnp.linspace(-8, 8, num_features)) / jnp.sqrt(num_features)
    ```
    '''
    )
elif select_features == "Switch":
    phi = functools.partial(switch_features, num_features=n_features)
    st.markdown(
        '''
    ```python
    def switch_features(x, num_features=2):
        """Feature map for a switch basis function."""
        # output shape: (n_samples, order)
        return (x > jnp.linspace(-8, 8, num_features)) * 1.0 / jnp.sqrt(num_features)
    ```
    '''
    )

st.markdown(
    """
    ```python
    phi = functools.partial(feature_function, num_features=n_features)
    prior = Gaussian(mu=jnp.zeros(F), Sigma=theta**2 * jnp.eye(F))
    posterior = prior.condition(phi(X), Y, sigma**2 * jnp.eye(len(X)))
    prior_x = phi(x) @ prior
    posterior_x = phi(x) @ posterior
    ```
    """
)

select_data = st.sidebar.multiselect(
    "Select data",
    options = jnp.arange(N),
    default = [i for i in range(N)]
    )

X = X[select_data]
Y = Y[select_data]

phi_X = phi(X)
F = phi_X.shape[1]

prior = Gaussian(mu=jnp.zeros(F), Sigma=theta**2 * jnp.eye(F))
posterior = prior.condition(phi_X, Y, sigma**2 * jnp.eye(len(X)))
dmudy = jax.jacfwd(
    lambda y: prior.condition(phi_X, y, sigma**2 * jnp.eye(len(X))).mu
)(Y)


plot_prior = st.sidebar.checkbox("plot prior", value=False)
plot_features = st.sidebar.checkbox("plot features", value=False)
plot_posterior = st.sidebar.checkbox("plot posterior", value=False)
plot_derivative = st.sidebar.checkbox("plot Jacobian dmu / dy", value=False)

fig, ax = plt.subplots()
ax.errorbar(
    X, Y, yerr=sigma * jnp.ones_like(Y), fmt="o", ms=2, color=rgb.tue_dark
)

x = jnp.linspace(-8, 8, 300)[:, None]

if plot_features:
    # plot the basis functions
    lines = ax.plot(x[:, 0], phi(x) * jnp.sqrt(F), color=rgb.tue_green, alpha=0.8)

if plot_prior:
    # plot the prior
    prior_x = phi(x) @ prior
    ax.plot(x[:, 0], prior_x.mu, color=rgb.tue_dark, label="prior")
    ax.fill_between(
        x[:, 0],
        prior_x.mu - 2 * prior_x.std,
        prior_x.mu + 2 * prior_x.std,
        color=rgb.tue_dark,
        alpha=0.2,
    )

    # plot posterior samples
    key = jax.random.PRNGKey(0)
    ax.plot(
        x[:, 0],
        phi(x) @ prior.sample(key, num_samples=30).T,
        color=rgb.tue_dark,
        alpha=0.2,
    )

if plot_posterior:
    # plot the posterior
    posterior_x = phi(x) @ posterior
    ax.plot(x[:, 0], posterior_x.mu, color=rgb.tue_red, label="posterior")
    ax.fill_between(
        x[:, 0],
        posterior_x.mu - 2 * posterior_x.std,
        posterior_x.mu + 2 * posterior_x.std,
        color=rgb.tue_red,
        alpha=0.2,
    )

    # plot posterior samples
    key = jax.random.PRNGKey(0)
    ax.plot(
        x[:, 0],
        phi(x) @ posterior.sample(key, num_samples=30).T,
        color=rgb.tue_red,
        alpha=0.2,
    )

    if plot_derivative:
        index = st.sidebar.slider(
            "i for dmu(x) / dy[i]", min_value=0, max_value=len(X) - 1
        )
        scale = st.sidebar.checkbox("scale by y?", value=False)
        if scale:
            ax.plot(x[:, 0], phi(x) @ (dmudy * Y), color=rgb.tue_orange, alpha=0.2)
        else:   
            ax.plot(x[:, 0], phi(x) @ (dmudy), color=rgb.tue_orange, alpha=0.2)
        ax.plot(X[index], Y[index], "o", color=rgb.tue_blue)
        ax.plot(x[:, 0], phi(x) @ dmudy[:, index] * Y[index], color=rgb.tue_blue)
        
# # sanity check
# kxx = phi(x) @ prior.Sigma @ phi(x).T
# vxx = kxx - phi(x) @ prior.Sigma @ phi(X).T @ (phi(x) @ dmudy).T
# ax.plot(x[:, 0], 2 * jnp.sqrt(jnp.diag(vxx)) + posterior_x.mu, color=rgb.tue_green)

ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_ylim(-12, 18)
ax.set_xlim(-8, 8)
ax.legend(loc="lower right")

st.pyplot(fig, dpi=400)
