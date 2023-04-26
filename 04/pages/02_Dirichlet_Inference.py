import streamlit as st

from jax import numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from mpltern.datasets import get_dirichlet_pdfs

from tueplots import bundles
from tueplots.constants.color import rgb

from scipy.stats import multinomial, betabinom

N = 256
vals = np.ones((N, 4))
vals[:, 0] = np.linspace(1, rgb.tue_red[0], N)
vals[:, 1] = np.linspace(1, rgb.tue_red[1], N)
vals[:, 2] = np.linspace(1, rgb.tue_red[2], N)
redCM = colors.ListedColormap(vals)

plt.rcParams.update(bundles.beamer_moml())
plt.rcParams.update({"figure.figsize": (6, 3)})
# plt.rcParams.update({"figure.dpi": 300})


st.set_page_config(
    page_title="Conjugate Priors",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "(c) Philipp Hennig, 2023"},
)

st.sidebar.header("Inference on Multivariate Probabilities")

a1 = st.sidebar.slider("alpha_1", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

a2 = st.sidebar.slider("alpha_2", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

a3 = st.sidebar.slider("alpha_3", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

N = st.sidebar.slider("N", min_value=0, max_value=100, step=1)

p1 = st.sidebar.slider("[p_true]_1", min_value=0.0, max_value=1.0, step=0.1, value=0.3)
p2 = st.sidebar.slider(
    "[p_true]_2", min_value=0.0, max_value=1.0 - p1, step=0.1, value=(1.0 - p1) / 3
)
p3 = 1.0 - p1 - p2
p = np.array([p1, p2, p3])
a = np.array([a1, a2, a3])

r = multinomial.rvs(n=N, p=p, random_state=1)
alpha = a + r

# predictive = betabinom.pmf(y, N, a+a0, b+b0)
p_true = np.array([p1, p2, p3])

# plotting
x = jnp.linspace(0, 1, 200)
# p = beta.pdf(x, a + a0, b + b0)

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.bar(np.arange(1, 4), r)
ax.set_xticks(np.arange(1, 4))
ax.set_xticklabels([f"$n_{i}$" for i in np.arange(1, 4)])
ax.axhline(N, color=rgb.tue_dark)
ax.axhline(0, color=rgb.tue_dark, linewidth=0.5)
for i in range(1,4):
    ax.plot(i,p_true[i-1]*N, "o", color=rgb.tue_gold,ms=5)
    ax.plot([i,i],[0,p_true[i-1]*N], color=rgb.tue_gold,marker="None",ms=5)
ax.set_title("observed")

# marginals
if N > 0:
    for i in range(3):
        y = jnp.arange(N + 1)
        predictive = betabinom.pmf(y, N, alpha[i], alpha.sum() - alpha[i])
        ax.barh(
            y, left=i + 1, width=predictive, height=0.8, color=rgb.tue_gold, alpha=0.7
        )

ax = fig.add_subplot(1, 2, 2, projection="ternary")
t, l, r, v = get_dirichlet_pdfs(n=91, alpha=alpha)
v[np.isnan(v) | np.isinf(v)] = 0
cmap = redCM
shading = "gouraud"
cs = ax.tripcolor(t, l, r, v, cmap=cmap, shading=shading, rasterized=True)
ax.tricontour(t, l, r, v, colors="k", linewidths=0.5)
ax.plot(p[0], p[1], p[2], "o", color=rgb.tue_green)
ax.set_rlabel('$f_1$')
ax.set_tlabel('$f_2$')
ax.set_llabel('$f_3$')

ax.taxis.set_label_position("tick2")
ax.laxis.set_label_position("tick2")
ax.raxis.set_label_position("tick2")

ax.set_title("latent")

st.markdown(
    """
    $$
    p(x) = \prod_{i=1} ^n f_{x_i} = \prod_{k=1} ^K f_k ^{\:n_k} \qquad x\in \{0;\dots,K\}, \\text{ and } n_k := |\{x_i \mid x_i = k \}|
    $$

    $$
    p(f\mid \\alpha) = \mathcal{D}(\\alpha) =\\frac{1}{B(\\alpha)} \prod_{k=1} ^K f_k ^{\\alpha_k-1}
    $$
    
    $$
    p(f\mid x) = \mathcal{D}(\\alpha + n)
    $$"""
)

st.pyplot(fig)
