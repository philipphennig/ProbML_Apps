import functools
import jax
import dataclasses
from jax import numpy as jnp
from collections.abc import Callable
from tueplots.constants.color import rgb


jax.config.update("jax_enable_x64", True)


@dataclasses.dataclass
class Gaussian:
    # Gaussian distribution with mean mu and covariance Sigma
    mu: jnp.ndarray  # shape (D,)
    Sigma: jnp.ndarray  # shape (D,D)

    @functools.cached_property
    def L(self):
        """Cholesky decomposition of the covariance matrix"""
        return jnp.linalg.cholesky(self.Sigma)

    @functools.cached_property
    def L_factor(self):
        """Cholesky factorization of the covariance matrix
        (for use in jax.scipy.linalg.cho_solve)"""
        return jax.scipy.linalg.cho_factor(self.Sigma, lower=True)

    @functools.cached_property
    def logdet(self):
        """log-determinant of the covariance matrix
        e.g. for computing the log-pdf
        """
        return 2 * jnp.sum(jnp.log(jnp.diag(self.L)))

    @functools.cached_property
    def prec(self):
        """precision matrix.
        you probably don't want to use this directly, but rather prec_mult
        """
        return jnp.linalg.inv(self.Sigma)

    def prec_mult(self, x):
        """precision matrix multiplication
        implements Sigma^{-1} @ x. For numerical stability, we use the Cholesky factorization
        """
        return jax.scipy.linalg.cho_solve(self.L_factor, x)

    @functools.cached_property
    def mp(self):
        """precision-adjusted mean"""
        return self.prec_mult(self.mu)

    def log_pdf(self, x):
        """log N(x;mu,Sigma)"""
        return (
            -0.5 * (x - self.mu) @ self.prec_mult(x - self.mu)
            - 0.5 * self.logdet
            - 0.5 * len(self.mu) * jnp.log(2 * jnp.pi)
        )

    def __mult__(self, other):
        """
        Products of Gaussian pdfs are Gaussian pdfs!
        Multiplication of two Gaussian PDFs  (not RVs!)
        other: Gaussian RV
        """
        Sigma = jnp.linalg.inv(self.prec + other.prec)
        mu = Sigma @ (self.mp + other.mp)
        return Gaussian(mu=mu, Sigma=Sigma)

    def __rmatmul__(self, A):
        """Linear maps of Gaussian RVs are Gaussian RVs
        A: linear map, shape (N,D)
        """
        return Gaussian(mu=A @ self.mu, Sigma=A @ self.Sigma @ A.T)

    @functools.singledispatchmethod
    def __add__(self, other):
        """Affine maps of Gaussian RVs are Gaussian RVs
        shift of a Gaussian RV by a constant.
        We implement this as a singledispatchmethod, because jnp.ndarrays can not be dispatched on,
        and register the addition of two RVs below
        """
        other = jnp.asarray(other)
        return Gaussian(mu=self.mu + other, Sigma=self.Sigma)

    def condition(self, A, y, Lambda):
        """Linear conditionals of Gaussian RVs are Gaussian RVs
        Conditioning of a Gaussian RV on a linear observation
        A: observation matrix, shape (N,D)
        y: observation, shape (N,)
        Lambda: observation noise covariance, shape (N,N)
        """
        Gram = A @ self.Sigma @ A.T + Lambda
        L = jax.scipy.linalg.cho_factor(Gram, lower=True)
        mu = self.mu + self.Sigma @ A.T @ jax.scipy.linalg.cho_solve(L, y - A @ self.mu)
        Sigma = self.Sigma - self.Sigma @ A.T @ jax.scipy.linalg.cho_solve(
            L, A @ self.Sigma
        )
        return Gaussian(mu=mu, Sigma=Sigma)

    @functools.cached_property
    def std(self):
        # standard deviation
        return jnp.sqrt(jnp.diag(self.Sigma))

    def sample(self, key, num_samples=1):
        """
         sample from the Gaussian
        # alternative implementation: works because the @ operator contracts on the second-to-last axis on the right
        # return (self.L @ jax.random.normal(key, shape=(num_samples, self.mu.shape[0], 1)))[...,0] + self.mu
        # or like this, more explicit, but not as easy to read
        # return jnp.einsum("ij,kj->ki", self.L, jax.random.normal(key, shape=(num_samples, self.mu.shape[0]))) + self.mu
        # or the scipy version:
        """
        return jax.random.multivariate_normal(
            key, mean=self.mu, cov=self.Sigma, shape=(num_samples,), method="svd"
        )


@Gaussian.__add__.register
def _add_gaussians(self, other: Gaussian):
    # sum of two Gaussian RVs
    return Gaussian(mu=self.mu + other.mu, Sigma=self.Sigma + other.Sigma)


@dataclasses.dataclass
class GaussianProcess:
    # mean function
    m: Callable[[jnp.ndarray], jnp.ndarray]
    # covariance function
    k: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]

    def __call__(self, x):
        return Gaussian(mu=self.m(x), Sigma=self.k(x[:, None, :], x[None, :, :]))

    def condition(self, y, X, sigma):
        return ConditionalGaussianProcess(
            self, y, X, Gaussian(mu=jnp.zeros_like(y), Sigma=sigma * jnp.eye(len(y)))
        )

    def plot(
        self,
        ax,
        x,
        color=rgb.tue_gray,
        mean_kwargs={},
        std_kwargs={},
        num_samples=0,
        rng_key=None,
    ):
        gp_x = self(x)
        ax.plot(x[:, 0], gp_x.mu, color=color, **mean_kwargs)
        ax.fill_between(
            x[:, 0],
            gp_x.mu - 2 * gp_x.std,
            gp_x.mu + 2 * gp_x.std,
            color=color,
            **std_kwargs
        )
        if num_samples > 0:
            ax.plot(
                x[:, 0],
                gp_x.sample(rng_key, num_samples=num_samples).T,
                color=color,
                alpha=0.2,
            )

    def plot_shaded(
        self,
        ax,
        x,
        color=rgb.tue_gray,
        yrange=None,
        yres=1000,
        mean_kwargs={},
        std_kwargs={},
        num_samples=0,
        rng_key=None,
    ):
        if yrange is None:
            yrange = ax.get_ylim()

        gp_x = self(x)
        ax.plot(x[:, 0], gp_x.mu, color=color, **mean_kwargs)

        yy = jnp.linspace(*yrange, yres)[:, None]
        ax.imshow(
            gp_shading(yy, gp_x.mu, gp_x.std),
            extent=[x[0, 0], x[-1, 0], *yrange],
            **std_kwargs,
            aspect="auto",
            origin="lower"
        )

        ax.plot(x[:, 0], gp_x.mu - 2 * gp_x.std, color=color, lw=0.25)
        ax.plot(x[:, 0], gp_x.mu + 2 * gp_x.std, color=color, lw=0.25)
        if num_samples > 0:
            ax.plot(
                x[:, 0],
                gp_x.sample(rng_key, num_samples=num_samples).T,
                color=color,
                alpha=0.2,
            )

def gp_shading(yy, mu, std):
        return jnp.exp(-((yy - mu) ** 2) / (2 * std**2)) # / (std * jnp.sqrt(2 * jnp.pi))

class ConditionalGaussianProcess(GaussianProcess):
    """
    A Gaussian process conditioned on data.
    Implented as a proper python class, which allows inheritance from the GaussianProcess superclass:
    A conditional Gaussian process contains a Gaussian process prior, provided at instantiation.
    """

    def __init__(self, prior, y, X, epsilon: Gaussian):
        self.prior = prior
        self.y = jnp.atleast_1d(y)  # shape: (n_samples,)
        self.X = jnp.atleast_2d(X)  # shape: (n_samples, n_features)
        self.epsilon = epsilon
        # initialize the super class
        super().__init__(self._mean, self._covariance)

    @functools.cached_property
    def predictive_covariance(self):
        return self.prior.k(self.X[:, None, :], self.X[None, :, :]) + self.epsilon.Sigma

    @functools.cached_property
    def predictive_mean(self):
        return self.prior.m(self.X) + self.epsilon.mu

    @functools.cached_property
    def predictive_covariance_cho(self):
        return jax.scipy.linalg.cho_factor(self.predictive_covariance)

    @functools.cached_property
    def representer_weights(self):
        return jax.scipy.linalg.cho_solve(
            self.predictive_covariance_cho,
            self.y - self.predictive_mean,
        )

    def _mean(self, x):
        x = jnp.asarray(x)
        return (
            self.prior.m(x)
            + self.prior.k(x[..., None, :], self.X[None, :, :])
            @ self.representer_weights
        )

    @functools.partial(jnp.vectorize, signature="(d),(d)->()", excluded={0})
    def _covariance(self, a, b):
        return self.prior.k(a, b) - self.prior.k(
            a, self.X
        ) @ jax.scipy.linalg.cho_solve(
            self.predictive_covariance_cho,
            self.prior.k(self.X, b),
        )

    def _m_proj(self, x, projection, projection_mean):
        x = jnp.asarray(x)

        if projection_mean is None:
            projection_mean = self.prior.m

        return (
            projection_mean(x)
            + projection(x[..., None, :], self.X[None, :, :]) @ self.representer_weights
        )

    @functools.partial(jnp.vectorize, signature="(d),(d)->()", excluded={0, 3})
    def _cov_proj(self, a, b, projection):
        return projection(a, b) - projection(a, self.X) @ jax.scipy.linalg.cho_solve(
            self.predictive_covariance_cho,
            projection(self.X, b),
        )

    def project(self, k_proj, m_proj=None):
        return GaussianProcess(
            lambda x: self._m_proj(x, k_proj, m_proj),
            lambda x0, x1: self._cov_proj(x0, x1, k_proj),
        )


class ParametricGaussianProcess(GaussianProcess):
    """Parametric special case of a Gaussian Process"""

    def __init__(self, phi: Callable, prior: Gaussian):
        self.phi = phi
        self.prior = prior
        super().__init__(self._mean, self._covariance)

    def _mean(self, x):
        return self.phi(x) @ self.prior.mu

    def _covariance(self, x, y):
        return self.phi(x) @ self.prior.Sigma @ self.phi(y).T
