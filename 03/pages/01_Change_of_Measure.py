import streamlit as st
from typing import Callable
from matplotlib_inline.backend_inline import set_matplotlib_formats

from jax import numpy as jnp
from jax import vmap, value_and_grad
from matplotlib import pyplot as plt
from tueplots import bundles
from tueplots.constants.color import rgb

from jax.scipy.stats import norm

plt.rcParams.update(bundles.beamer_moml())
plt.rcParams.update({"figure.figsize": (5, 3)})
set_matplotlib_formats("svg")

st.set_page_config(
    page_title="Change of Measure",
    page_icon="🧊",
    # layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "(c) Philipp Hennig, 2023"},
)


def sigmoid(loc: float, gain: float) -> Callable:
    return value_and_grad(lambda a: 1.0 / (1.0 + jnp.exp(-(a - loc) / gain)))


c1 = st.sidebar.container()
c1.write("Feature 1")
# l1, l2, l3, g1, g2, g3, with_correction=False
l1 = c1.slider(
    "Location", key="loc1", min_value=-3.0, max_value=3.0, value=-1.0, step=0.1
)
g1 = c1.select_slider(
    "Gain",
    key="gain1",
    options=jnp.logspace(-2, 1, 100),
    format_func=lambda x: f"{x:2.2e}",
    value=1,
)

c2 = st.sidebar.container()
c2.write("Feature 2")
# l1, l2, l3, g1, g2, g3, with_correction=False
l2 = c2.slider(
    "Location", key="loc2", min_value=-3.0, max_value=3.0, value=0.0, step=0.1
)
g2 = c2.select_slider(
    "Gain",
    key="gain2",
    options=jnp.logspace(-2, 1, 100),
    format_func=lambda x: f"{x:2.2e}",
    value=1,
)

c3 = st.sidebar.container()
c3.write("Feature 3")
# l1, l2, l3, g1, g2, g3, with_correction=False
l3 = c3.slider(
    "Location", key="loc3", min_value=-3.0, max_value=3.0, value=1.0, step=0.1
)
g3 = c3.select_slider(
    "Gain",
    key="gain3",
    options=jnp.logspace(-2, 1, 100),
    format_func=lambda x: f"{x:2.2e}",
    value=1,
)

show_features = st.sidebar.checkbox("show features", value=True)

corr = st.sidebar.checkbox("use correction", value=False)

locs = jnp.asarray([l1, l2, l3])
gains = jnp.asarray([g1, g2, g3])
N = 400
x = jnp.linspace(-3, 3, N)

# I do not know how to do this list comprehension (efficiently) in jax. Improvements welcome!
Fs = jnp.asarray([vmap(sigmoid(l, g))(x) for (l, g) in zip(locs, gains)])  # [3, 2, N]
F = Fs.sum(axis=0)
# unpack
f = F[0, :]
df = F[1, :]

p = norm.pdf(x, loc=0, scale=1)
# p_y(y=u(x)) = p_x(v(y)) * |du/dx|^{-1} for v(y) = u^{-1}(x)
py = p / jnp.abs(df)

# = sum p[i] * d(x[i]), with each "box" d(x[i]) of size 6 / N
px_int = jnp.sum(p) * 6 / (N - 1)
f_int = jnp.sum(f) * 3 / (N - 1)
py_int = jnp.sum(py) * 3 / (N - 1)

fig, ax = plt.subplots()

ax.text(-2.5, 2.5, f"The integral over p_x(x) is approximately {px_int:.2f}.")
ax.text(-2.5, 2.25, f"The integral over f(x) is approximately {f_int:.2f}.")

# plot the features:
if show_features:
    ax.plot(x, Fs[:, 0, :].T, "-", color=rgb.tue_gray, lw=0.5)
    ax.plot(x, Fs[:, 1, :].T, "--", color=rgb.tue_gray, lw=0.5)

ax.plot(x, f, label="$f$", color=rgb.tue_red)
ax.plot(x, df, "--", label="$df/dx$", color=rgb.tue_red)
ax.plot(x, p, "-", color=rgb.tue_gold, label="$p(x)$")
ax.plot(p - 3, f, "-.", color=rgb.tue_orange, label="$p_x(y=f(x))$")
if corr:
    ax.plot(py - 3, f, "-", color=rgb.tue_green, label="$p_y(y=f(x))$")
    ax.text(-2.5, 2.0, f"The integral over p_y(y) is approximately {py_int:.2f}.")
ax.axvline(-3, color=rgb.tue_dark)
ax.axhline(0, color=rgb.tue_dark)

ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_xlim([-3, 3])
ax.set_ylim([-0.1, 3.1])
ax.legend(loc="center right")

st.pyplot(fig)  # ,dpi=300)
