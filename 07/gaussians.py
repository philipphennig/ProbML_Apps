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
    mu: jnp.ndarray
    Sigma: jnp.ndarray

    def condition(self, A, y, Lambda):
        # conditioning of a Gaussian RV on a linear observation
        # A: observation matrix
        # y: observation
        # Lambda: observation noise covariance
        Gram = A @ self.Sigma @ A.T + Lambda
        L = jax.scipy.linalg.cho_factor(Gram, lower=True)
        mu = self.mu + self.Sigma @ A.T @ jax.scipy.linalg.cho_solve(L, y - A @ self.mu)
        Sigma = self.Sigma - self.Sigma @ A.T @ jax.scipy.linalg.cho_solve(
            L, A @ self.Sigma
        )
        return Gaussian(mu=mu, Sigma=Sigma)

    # def learn(self, A, y, Lambda):
    #     def loss(w):
    #         (A @ w - y)**2 / Lambda + w @ self.prec @ w

    @functools.cached_property
    def L(self):
        # Cholesky decomposition of the covariance matrix
        return jnp.linalg.cholesky(self.Sigma)

    @functools.cached_property
    def logdet(self):
        # log-determinant of the covariance matrix
        return 2 * jnp.sum(jnp.log(jnp.diag(self.L)))

    @functools.cached_property
    def prec(self):
        # precision matrix
        return jnp.linalg.inv(self.Sigma)

    @functools.cached_property
    def mp(self):
        # precision-adjusted mean
        return self.prec @ self.mu

    def __mult__(self, other):
        # multiplication of two Gaussian pdfs
        Sigma = jnp.linalg.inv(self.prec + other.prec)
        mu = Sigma @ (self.mp + other.mp)
        return Gaussian(mu=mu, Sigma=Sigma)

    @functools.singledispatchmethod
    def __add__(self, other):
        other = jnp.asarray(other)
        # shifting a Gaussian RV by a constant
        # we implement this as the default, because jnp.ndarrays can not be dispatched on
        # we register the addition of two RVs below
        return Gaussian(mu=self.mu + other, Sigma=self.Sigma)

    def __rmatmul__(self, other):
        # linear projection of a Gaussian RV
        return Gaussian(mu=other @ self.mu, Sigma=other @ self.Sigma @ other.T)

    @functools.cached_property
    def std(self):
        # standard deviation
        return jnp.sqrt(jnp.diag(self.Sigma))

    def sample(self, key, num_samples=1):
        # sample from the Gaussian
        # alternative implementation: works because the @ operator contracts on the second-to-last axis on the right
        # return (self.L @ jax.random.normal(key, shape=(num_samples, self.mu.shape[0], 1)))[...,0] + self.mu
        # or like this, more explicit, but not as easy to read
        # return jnp.einsum("ij,kj->ki", self.L, jax.random.normal(key, shape=(num_samples, self.mu.shape[0]))) + self.mu
        # or the scipy version:
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
    def predictive_covariance_cho(self):
        return jax.scipy.linalg.cho_factor(self.predictive_covariance)

    @functools.cached_property
    def representer_weights(self):
        return jax.scipy.linalg.cho_solve(
            self.predictive_covariance_cho,
            self.y - self.prior(self.X).mu - self.epsilon.mu,
        )

    def _mean(self, x):
        x = jnp.asarray(x)
        return (
            self.prior(x).mu
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
