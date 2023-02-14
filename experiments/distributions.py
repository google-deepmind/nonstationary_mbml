# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Probability distributions used to generate trajectories.

A distribution can produce samples of shape (batch_size, ...) with multiple
parameters at once, of shape (batch_size, parameter_size).
It also has a density method, returning the positive density for given sets of
points, in the support of the distribution.

For now, the distributions are only one dimension, sufficient for our
experiments. It means that you can only produce iid samples, no multivariate
distributions.
"""

import abc

import chex
import jax
import jax.nn as jnn
import jax.numpy as jnp

_EPSILON = 1e-7
# Constant to be added to avoid inputs close to asymptotes
# e.g. to ensure that the variance of a Gaussian >= _EPSILON


def split_params(parameters: chex.Array) -> tuple[chex.Array, ...]:
  """Returns a tuple of batches of individual parameters.

  Args:
    parameters: a (batch_size, parameter_size, ...) tensor

  Returns:
    param_tuple: tuple whose elements are (batch_size, ...) tensors
  """
  parameters = jnp.swapaxes(parameters, 1, 0)
  param_tuple = tuple(x for x in parameters)
  return param_tuple


def broadcast_params_with_ones(
    parameters: chex.Array,
    num_dims_to_match: int,
) -> chex.Array:
  """Expand dimension of parameters num_dims times.

  Args:
    parameters: a (batch_size, parameter_size) tensor
    num_dims_to_match: the number of dimensions to match

  Returns:
    broadcasted: a (batch_size, parameter_size, 1, 1, ...) tensor, with shape
      length of num_dims_to_match
  """
  num_dims_to_add = num_dims_to_match - len(parameters.shape)
  return parameters[(...,) + (None,) * num_dims_to_add]


class Distribution(abc.ABC):
  """Abstract class for random distributions."""

  @property
  @abc.abstractmethod
  def parameter_size(self) -> int:
    """The number of parameters expected by the distribution."""

  @property
  @abc.abstractmethod
  def feature_size(self) -> int:
    """The size of an iid sample of the distribution."""

  @abc.abstractmethod
  def sample(
      self,
      rng: chex.PRNGKey,
      parameters: chex.Array,
      shape: tuple[int, ...],
  ) -> chex.Array:
    """Sample data with given parameters.

    Args:
      rng: the random key
      parameters: a (batch_size, parameter_size) tensor used to sample the data
      shape: the shape of the output tensor. Must be of the form (batch_size,
        ..., feature_size).

    Returns:
      samples: an output_shape tensor, dtype float64
    """

  @abc.abstractmethod
  def density(
      self,
      parameters: chex.Array,
      x: chex.Array,
      logits: bool = False,
  ) -> chex.Array:
    """Returns the evaluation of the density function at x.

    Args:
      parameters: a (batch_size, parameter_size) tensor of density parameters.
      x: (batch_size, ..., feature_size) tensor of points to be evaluated
      logits: If True, interprets the `parameters` arg as the logits and
        performs the appropriate conversions internally. Default is False.

    Returns:
      densities: a (batch_size, ..., 1) tensor, containing the values of the
        density function evaluated at each entry of x. dtype float32
    """

  def log_density(
      self,
      parameters: chex.Array,
      x: chex.Array,
      logits: bool = False,
  ) -> chex.Array:
    """Computes the log of the density at x. Override where appropriate."""
    return jnp.log(self.density(parameters, x, logits=logits) + _EPSILON)

  @abc.abstractmethod
  def mean(self, parameters: chex.Array) -> chex.Array:
    """(batch_size, feature_size) array of the means of the given parameters."""

  @abc.abstractmethod
  def std(self, parameters: chex.Array) -> chex.Array:
    """(batch_size, feature_size) array of the std devs of the given parameters.
    """

  @abc.abstractmethod
  def entropy(self, parameters: chex.Array) -> chex.Array:
    """(batch_size, 1) array of the entropies of the given parameters."""

  @abc.abstractmethod
  def logits_to_params(self, logits: chex.Array) -> chex.Array:
    """Given the final pre-activation output, compute appropriate parameters.

    E.g., for a gaussian, map the 2nd column into positive values above a
    certain epsilon, to avoid divergence due to zero variance.

    Args:
      logits: a (batch_size, parameter_size) pre-activation output of the final
        layer of a neural net.

    Returns:
      a (batch_size, parameter_size) tensor of valid parameters for the
        distribution.
    """

  def _validate_parameters_shape(
      self,
      parameters: chex.Array,
      output_shape: tuple[int, ...],
  ) -> None:
    """Checks that `parameters` has shape (batch_size, parameter_size)."""
    expected_shape = (output_shape[0], self.parameter_size)
    if parameters.shape != expected_shape:
      raise ValueError("Parameters shape mismatch. "
                       f"Expected {expected_shape}, got {parameters.shape}.")

  def _validate_output_shape(
      self,
      parameters: chex.Array,
      output_shape: tuple[int, ...],
  ) -> None:
    """Checks that `output_shape` has form (batch_size, ..., feature_size)."""
    leading_and_trailing_dims = (output_shape[0], output_shape[-1])
    batch_size = parameters.shape[0]
    if leading_and_trailing_dims != (batch_size, self.feature_size):
      raise ValueError(f"Bad shape. "
                       f"Expected ({batch_size}, ..., {self.feature_size}). "
                       f"Got {output_shape}")


class PrecisionGaussianDistribution(Distribution):
  """Gaussian Distribution parameterised by precision.

  The parameters of this distribution are the mean and 1/var = rho.

  This parameterisation results in a stable log_density computation.
  """

  parameter_size = 2
  feature_size = 1

  def sample(
      self,
      rng: chex.PRNGKey,
      parameters: chex.Array,
      shape: tuple[int, ...],
  ) -> chex.Array:
    self._validate_parameters_shape(parameters, shape)
    self._validate_output_shape(parameters, shape)
    parameters = broadcast_params_with_ones(parameters, len(shape) + 1)
    mu, rho = split_params(parameters)
    sigma = 1 / jnp.sqrt(rho)
    batch = mu + sigma * jax.random.normal(rng, shape=shape)
    return batch

  def log_density(
      self,
      parameters: chex.Array,
      x: chex.Array,
      logits: bool = False,
  ) -> chex.Array:
    self._validate_parameters_shape(parameters, x.shape)
    parameters = broadcast_params_with_ones(parameters, len(x.shape) + 1)
    if logits:
      mu, log_rho = split_params(parameters)
      rho = jnp.exp(log_rho)
    else:
      mu, rho = split_params(parameters)
      log_rho = jnp.log(rho + _EPSILON)

    return 1 / 2 * (log_rho - jnp.log(2 * jnp.pi) - (rho * (x - mu)**2))

  def density(
      self,
      parameters: chex.Array,
      x: chex.Array,
      logits: bool = False,
  ) -> chex.Array:
    return jnp.exp(self.log_density(parameters, x))

  def logits_to_params(self, logits: chex.Array) -> chex.Array:
    mean = logits[..., 0]
    rho = jnp.exp(logits[..., 1])
    return jnp.stack([mean, rho], axis=-1)

  def mean(self, parameters: chex.Array) -> chex.Array:
    # Index with list to not lose the dimension.
    return parameters[..., [0]]

  def std(self, parameters: chex.Array) -> chex.Array:
    return 1 / jnp.sqrt(parameters[..., [1]])

  def entropy(self, parameters: chex.Array) -> chex.Array:
    _, rho = split_params(parameters)
    return 1 / 2 * (jnp.log(2 * jnp.pi * jnp.e) - jnp.log(rho))

  def kl(self, p: chex.Array, q: chex.Array) -> chex.Array:
    """Computes the KL between the Gaussians parameterised by p and q.

    Args:
      p: [..., 2] tensor of means and precisions.
      q: [..., 2] tensor of means and precisions.

    Returns:
      [...,] tensor of KL between p and q.
    """

    mu_p, mu_q = p[..., 0], q[..., 0]
    var_p, var_q = 1 / p[..., 1], 1 / q[..., 1]
    std_p, std_q = jnp.sqrt(var_p), jnp.sqrt(var_q)

    return jnp.log(std_q / std_p) + (var_p +
                                     (mu_p - mu_q)**2) / (2 * var_q) - 1 / 2


class BetaDistribution(Distribution):
  """Beta distribution.

  Parameters are alpha and beta.

  The pdf is p(x; alpha, beta) = x^(alpha-1) * (1-x)^(beta-1) / B(alpha, beta)
  where B(alpha, beta) = G(alpha)*G(beta) / G(alpha + beta) and
        G is the Gamma function.
  """

  parameter_size = 2
  feature_size = 1

  def sample(
      self,
      rng: chex.PRNGKey,
      parameters: chex.Array,
      shape: tuple[int, ...],
  ) -> chex.Array:
    self._validate_parameters_shape(parameters, shape)
    self._validate_output_shape(parameters, shape)
    parameters = broadcast_params_with_ones(parameters, len(shape) + 1)
    alpha, beta = split_params(parameters)
    batch = jax.random.beta(rng, alpha, beta, shape)
    return batch.astype(jnp.float32)

  def density(
      self,
      parameters: chex.Array,
      x: chex.Array,
      logits: bool = False,
  ) -> chex.Array:
    self._validate_parameters_shape(parameters, x.shape)
    parameters = self.logits_to_params(parameters) if logits else parameters
    parameters = broadcast_params_with_ones(parameters, len(x.shape) + 1)
    alpha, beta = split_params(parameters)
    return jax.scipy.stats.beta.pdf(x, alpha, beta)

  def logits_to_params(self, logits: chex.Array) -> chex.Array:
    raise NotImplementedError()

  def mean(self, parameters: chex.Array) -> chex.Array:
    alpha, beta = split_params(parameters)
    return alpha / (alpha + beta)

  def std(self, parameters: chex.Array) -> chex.Array:
    a, b = split_params(parameters)
    variance = a * b / ((a + b)**2 * (a + b + 1))
    return jnp.sqrt(variance)

  def entropy(self, parameters: chex.Array) -> chex.Array:
    raise NotImplementedError()


class GammaDistribution(Distribution):
  """Gamma distribution.

  Parameters are alpha (shape) and beta (rate).

  The pdf is p(x; a, b) = b^a / G(a) * x^(a-1) exp(-b*x)
  where G is the Gamma function.
  """

  parameter_size = 2
  feature_size = 1

  def sample(
      self,
      rng: chex.PRNGKey,
      parameters: chex.Array,
      shape: tuple[int, ...],
  ) -> chex.Array:
    self._validate_parameters_shape(parameters, shape)
    self._validate_output_shape(parameters, shape)
    parameters = broadcast_params_with_ones(parameters, len(shape) + 1)
    alpha, beta = split_params(parameters)
    # jax.random.gamma samples from Gamma(alpha, 1). To obtain a sample from
    # Gamma(alpha, beta), we rescale:
    batch = jax.random.gamma(rng, alpha, shape) / beta
    return batch.astype(jnp.float32)

  def density(
      self,
      parameters: chex.Array,
      x: chex.Array,
      logits: bool = False,
  ) -> chex.Array:
    self._validate_parameters_shape(parameters, x.shape)
    parameters = self.logits_to_params(parameters) if logits else parameters
    parameters = broadcast_params_with_ones(parameters, len(x.shape) + 1)
    alpha, beta = split_params(parameters)
    # jax.scipy.stats.gamma.pdf is the pdf of Gamma(alpha, 1). To obtain the
    # pdf for Gamma(alpha, beta), we reparameterise:
    return jax.scipy.stats.gamma.pdf(beta * x, alpha) * beta

  def logits_to_params(self, logits: chex.Array) -> chex.Array:
    raise NotImplementedError()

  def mean(self, parameters: chex.Array) -> chex.Array:
    alpha, beta = split_params(parameters)
    return alpha / beta

  def std(self, parameters: chex.Array) -> chex.Array:
    alpha, beta = split_params(parameters)
    return jnp.sqrt(alpha) / beta

  def entropy(self, parameters: chex.Array) -> chex.Array:
    raise NotImplementedError()


class LomaxDistribution(Distribution):
  """Lomax distribution.

  Parameters are alpha (shape) and lambda (scale).

  alpha > 0 and scale > 0.
  The pdf is p(x; a, l) = (a * l**a) / (x + l)**(a + 1)
  """

  parameter_size = 2
  feature_size = 1

  def sample(
      self,
      rng: chex.PRNGKey,
      parameters: chex.Array,
      shape: tuple[int, ...],
  ) -> chex.Array:
    self._validate_parameters_shape(parameters, shape)
    self._validate_output_shape(parameters, shape)
    parameters = broadcast_params_with_ones(parameters, len(shape) + 1)
    alpha, lamb = split_params(parameters)
    # jax.random.pareto samples from Pareto(alpha). To obtain a sample from
    # Lomax(alpha, lambda), we rescale:
    batch = (jax.random.pareto(rng, alpha, shape) + 1) * lamb
    return batch.astype(jnp.float32)

  def log_density(
      self,
      parameters: chex.Array,
      x: chex.Array,
      logits: bool = False,
  ) -> chex.Array:
    self._validate_parameters_shape(parameters, x.shape)
    parameters = broadcast_params_with_ones(parameters, len(x.shape) + 1)
    if logits:
      log_alpha, log_lamb = split_params(parameters)
      alpha, lamb = jnp.exp(log_alpha), jnp.exp(log_lamb)
    else:
      alpha, lamb = split_params(parameters)
      log_alpha, log_lamb = jnp.log(alpha + _EPSILON), jnp.log(lamb + _EPSILON)

    log_dens_num = log_alpha + alpha * log_lamb
    log_dens_den = (alpha + 1) * jnp.log(x + lamb)
    return log_dens_num - log_dens_den

  def density(
      self,
      parameters: chex.Array,
      x: chex.Array,
      logits: bool = False,
  ) -> chex.Array:
    return jnp.exp(self.log_density(parameters, x, logits=logits))

  def logits_to_params(self, logits: chex.Array) -> chex.Array:
    return jnp.exp(logits)

  def mean(self, parameters: chex.Array) -> chex.Array:
    alpha, lamb = split_params(parameters)
    return jnp.where(alpha > 1, lamb / (alpha - 1), jnp.nan)

  def std(self, parameters: chex.Array) -> chex.Array:
    raise NotImplementedError()

  def entropy(self, parameters: chex.Array) -> chex.Array:
    raise NotImplementedError()

  def kl(self, p: chex.Array, q: chex.Array) -> chex.Array:
    """Computes approximate KL between p and q.

    Since Lomax KL is analytically intractable, we approximate the KL using
    the Exponential distribution, converting the Lomax parameters to Exponential
    parameters with the same mean. When doing so, the alpha parameter is clipped
    to be >1, as the Lomax mean is undefined for alpha ≤ 1.

    This is a hacky approximation. Use at your own risk.

    Args:
      p: [..., 2] parameter array for distribution p.
      q: [..., 2] parameter array for distribution q.

    Returns:
      a [...,] array of approximated KL between p and q.
    """

    lambda_p = _approximate_lomax_params_to_exponential_params(p) + _EPSILON
    lambda_q = _approximate_lomax_params_to_exponential_params(q) + _EPSILON

    return jnp.log(lambda_p / lambda_q) + (lambda_q / lambda_p) - 1


def _approximate_lomax_params_to_exponential_params(params):
  """Given Lomax params, returns the Exponential params with the same mean."""
  alpha, beta = jnp.maximum(1 + _EPSILON, params[..., 0]), params[..., 1]
  return (alpha - 1) / (beta + _EPSILON)


class DirichletDistribution(Distribution):
  """Dirichlet distribution with given parameter_size and feature_size."""

  def __init__(self, size):
    self._size = size

  @property
  def parameter_size(self) -> int:
    return self._size

  @property
  def feature_size(self) -> int:
    return self._size

  def sample(
      self,
      rng: chex.PRNGKey,
      parameters: chex.Array,
      shape: tuple[int, ...],
  ) -> chex.Array:
    self._validate_parameters_shape(parameters, shape)
    self._validate_output_shape(parameters, shape)
    parameters = broadcast_params_with_ones(parameters, len(shape))

    # jax.random.dirichlet expects the alphas to be in axis=-1, so we rearrange:
    alphas = jnp.moveaxis(parameters, source=1, destination=-1)

    # a sample from jax.random.dirichlet has shape (shape + alphas.shape[-1]),
    # so we remove the trailing dimension to preserve the final output shape:
    shape = shape[:-1]

    batch = jax.random.dirichlet(rng, alphas, shape)
    return batch.astype(jnp.float32)

  def density(
      self,
      parameters: chex.Array,
      x: chex.Array,
      logits: bool = False,
  ) -> chex.Array:
    raise NotImplementedError()

  def logits_to_params(self, logits: chex.Array) -> chex.Array:
    raise NotImplementedError()

  def mean(self, parameters: chex.Array) -> chex.Array:
    return parameters / jnp.sum(parameters, axis=1, keepdims=True)

  def std(self, parameters: chex.Array) -> chex.Array:
    raise NotImplementedError()

  def entropy(self, parameters: chex.Array) -> chex.Array:
    raise NotImplementedError()


class CategoricalDistribution(Distribution):
  """Categorical Distribution with given parameter_size and feature_size.

  The support of CategoricalDistribution(n) are one-hot vectors of size n.

  The parameter vector contains the probabilities corresponding to classes.
  """

  def __init__(self, size):
    self._size = size

  @property
  def parameter_size(self) -> int:
    return self._size

  @property
  def feature_size(self) -> int:
    return self._size

  def sample(
      self,
      rng: chex.PRNGKey,
      parameters: chex.Array,
      shape: tuple[int, ...],
  ) -> chex.Array:
    self._validate_parameters_shape(parameters, shape)
    self._validate_output_shape(parameters, shape)

    batch_size = shape[0]
    keys = jax.random.split(rng, batch_size)

    # jax.random.categorical expects logits, so we convert to log space.
    # we add an epsilon for stability. Regrettably, the epsilon means
    # that we can sample from classes with zero probability.
    log_probabilities = jnp.log(parameters + _EPSILON)

    # sample each trajectory in the batch separately to work around weird
    # shape behaviours in jax.random.categorical.
    def unbatched_categorical(
        log_p: chex.Array,
        rng: chex.PRNGKey,
    ) -> chex.Array:
      # Sample every trajectory in the batch separately but not the feature size
      # dimension since it will be one-hot encoded.
      return jax.random.categorical(rng, log_p, shape=shape[1:-1])

    batch = jax.vmap(unbatched_categorical)(log_probabilities, keys)

    return jnn.one_hot(batch, self.feature_size, dtype=jnp.float32)

  def density(
      self,
      parameters: chex.Array,
      x: chex.Array,
      logits: bool = False,
  ) -> chex.Array:
    self._validate_parameters_shape(parameters, x.shape)
    parameters = self.logits_to_params(parameters) if logits else parameters
    parameters = broadcast_params_with_ones(parameters, len(x.shape))
    # align parameter axis to feature axis.
    probabilities = jnp.moveaxis(parameters, 1, -1)

    # since x is one-hot, this gives us the likelihoods of the observed data
    # with zeros in non-observed entries. The sum removes these zeros.
    return jnp.sum(probabilities * x, axis=-1, keepdims=True)

  def mean(self, parameters: chex.Array) -> chex.Array:
    raise NotImplementedError()

  def std(self, parameters: chex.Array) -> chex.Array:
    raise NotImplementedError()

  def entropy(self, parameters: chex.Array) -> chex.Array:
    return -jnp.sum(parameters * jnp.log(parameters), axis=-1, keepdims=True)

  def kl(self, p: chex.Array, q: chex.Array) -> chex.Array:
    """Computes the KL between the Categoricals parameterised by p and q.

    Args:
      p: [..., parameter_size] tensor of means and precisions.
      q: [..., parameter_size] tensor of means and precisions.

    Returns:
      [...,] tensor of KL between p and q.
    """
    return jnp.sum(p * jnp.log(p / q), axis=-1)

  def logits_to_params(self, logits: chex.Array) -> chex.Array:
    return jax.nn.softmax(logits, axis=-1)


class ExponentialDistribution(Distribution):
  """Exponential implementation of abstract Distribution."""

  parameter_size = 1
  feature_size = 1

  def sample(
      self,
      rng: chex.PRNGKey,
      parameters: chex.Array,
      shape: tuple[int, ...],
  ) -> chex.Array:
    self._validate_parameters_shape(parameters, shape)
    self._validate_output_shape(parameters, shape)
    parameters = broadcast_params_with_ones(parameters, len(shape) + 1)
    (tau,) = split_params(parameters)
    batch = 1 / tau * jax.random.exponential(rng, shape)
    return batch.astype(jnp.float32)

  def density(
      self,
      parameters: chex.Array,
      x: chex.Array,
      logits: bool = False,
  ) -> chex.Array:
    self._validate_parameters_shape(parameters, x.shape)
    parameters = self.logits_to_params(parameters) if logits else parameters
    parameters = broadcast_params_with_ones(parameters, len(x.shape) + 1)
    (tau,) = split_params(parameters)
    return (tau * jnp.exp(-tau * x)) * (x >= 0)

  def logits_to_params(self, logits: chex.Array) -> chex.Array:
    return jnp.exp(logits)

  def mean(self, parameters: chex.Array) -> chex.Array:
    return 1 / parameters

  def std(self, parameters: chex.Array) -> chex.Array:
    return 1 / parameters

  def entropy(self, parameters: chex.Array) -> chex.Array:
    tau = parameters
    return 1 - jnp.log(tau)


class UniformDistribution(Distribution):
  """Uniform implementation of abstract Distribution."""

  parameter_size = 2
  feature_size = 1

  def sample(
      self,
      rng: chex.PRNGKey,
      parameters: chex.Array,
      shape: tuple[int, ...],
  ) -> chex.Array:
    self._validate_parameters_shape(parameters, shape)
    self._validate_output_shape(parameters, shape)
    parameters = broadcast_params_with_ones(parameters, len(shape) + 1)
    min_val, max_val = split_params(parameters)
    normal_uniform = jax.random.uniform(rng, shape, minval=0., maxval=1.)
    batch = min_val + (max_val - min_val) * normal_uniform
    return batch

  def density(
      self,
      parameters: chex.Array,
      x: chex.Array,
      logits: bool = False,
  ) -> chex.Array:
    self._validate_parameters_shape(parameters, x.shape)
    parameters = self.logits_to_params(parameters) if logits else parameters
    parameters = broadcast_params_with_ones(parameters, len(x.shape) + 1)
    min_val, max_val = split_params(parameters)

    return jnp.where(
        (x - min_val) * (max_val - x) >= 0,  # <=> min_val >= x >= max_val
        1 / (max_val - min_val),
        0.)

  def logits_to_params(self, logits: chex.Array) -> chex.Array:
    raise UserWarning("Uniform distribution cannot de differentiated "
                      "in the current implementation.")

  def mean(self, parameters: chex.Array) -> chex.Array:
    min_val, max_val = split_params(parameters)
    return (min_val + max_val) / 2

  def std(self, parameters: chex.Array) -> chex.Array:
    min_val, max_val = split_params(parameters)
    return (max_val - min_val) / jnp.sqrt(12)

  def entropy(self, parameters: chex.Array) -> chex.Array:
    min_val, max_val = split_params(parameters)
    return jnp.log(max_val - min_val)


class ConstantDistribution(Distribution):
  """Constant (Dirac) implementation of abstract Distribution.

  Not intended to be used for modelling, but can be used to fix some
  parameters of a TrajectoryGenerator to, e.g., generate a distribution
  over Gaussian distributions with fixed variance but different means.
  """

  parameter_size = 1
  feature_size = 1

  def sample(
      self,
      rng: chex.PRNGKey,
      parameters: chex.Array,
      shape: tuple[int, ...],
  ) -> chex.Array:
    self._validate_parameters_shape(parameters, shape)
    self._validate_output_shape(parameters, shape)
    parameters = broadcast_params_with_ones(parameters, len(shape) + 1)
    (constant,) = split_params(parameters)
    return constant * jnp.ones(shape, dtype=jnp.float32)

  def density(
      self,
      parameters: chex.Array,
      x: chex.Array,
      logits: bool = False,
  ) -> chex.Array:
    self._validate_parameters_shape(parameters, x.shape)
    raise UserWarning("Attempting to use ConstantDistribution.density(). "
                      "This is most likely an error, as this function would "
                      "not tell you anything meaningful.")
    # If this function was supposed to return a value, it would be:
    # jnp.where(parameters == x, jnp.inf, 0)

  def mean(self, parameters: chex.Array) -> chex.Array:
    return parameters

  def std(self, parameters: chex.Array) -> chex.Array:
    return jnp.zeros_like(parameters)

  def entropy(self, parameters: chex.Array) -> chex.Array:
    raise NotImplementedError("No entropy for Dirac distribution.")

  def logits_to_params(self, logits: chex.Array) -> chex.Array:
    raise UserWarning("Constant distribution cannot be differentiated.")
