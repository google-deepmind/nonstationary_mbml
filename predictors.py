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

"""Predictors used in the project.

The interface is general and accept any 'unrolling' predictor, basically
implementing a jax.lax.scan function under the cover.
"""

import abc
from typing import Any, Callable, Optional

import chex
import haiku as hk
import jax
import jax.numpy as jnp

from nonstationary_mbml.experiments import distributions


class Predictor(abc.ABC):
  """Predictors used for predictions."""

  @abc.abstractmethod
  def initial_state(
      self,
      params: Any,
      rng: chex.PRNGKey,
      batch_size: int,
  ) -> Any:
    """Sample an initial state for the predictor.

    Can be independent of params as well as rng.
    Args:
      params: the parameters of the predictor, can be anything
      rng: a random key
      batch_size: the number of initial states to return

    Returns:
      init_state: a list or array of size batch_size, containing the states of
        the predictor
    """

  @abc.abstractmethod
  def init_params(self, rng: chex.PRNGKey, batch_init: chex.Array,
                  state_init: chex.Array) -> hk.Params:
    """Initialise parameters.

    Args:
      rng: Seed operation.
      batch_init: Dummy data to create network's parameters.
      state_init: Dummy state to create network's parameters.

    Returns:
      Parameters of the network.
    """

  @abc.abstractmethod
  def unroll(
      self,
      params: Any,
      rng: chex.PRNGKey,
      batch: chex.Array,
      init_state: Any,
  ) -> chex.Array:
    """Unroll our predictor on a batch of trajectories.

    Args:
      params: the parameters of the predictor, can be anything
      rng: a random key
      batch: a (batch_size, seq_length,)+obs_shape tensor, containing the
        observations for the predictor
      init_state: the initial state of the predictor

    Returns:
      predictions: a (batch_size, seq_length, parameter_size) tensor, containing
      the output of the predictor, i.e., predictions of the next value in the
      sequence
    """


class InContextPredictor(Predictor):
  """A predictor without state that only looks at the current context."""

  def __init__(self, predictor: Callable[[chex.Array], chex.Array]):
    self._predictor_init, self._predictor_apply = hk.transform(predictor)

  def initial_state(self, params: hk.Params, rng: chex.PRNGKey,
                    batch_size: int) -> Optional[Any]:
    # No state for this predictor
    return None

  def init_params(self, rng: chex.PRNGKey, batch_init: chex.Array,
                  state_init: Optional[chex.Array]) -> hk.Params:
    del state_init
    return self._predictor_init(rng, batch_init)

  def unroll(self, params: hk.Params, rng: chex.PRNGKey, batch: chex.Array,
             init_state: Optional[chex.Array]) -> chex.Array:
    del init_state
    output = self._predictor_apply(params, rng, batch)
    return output, None  # pytype: disable=bad-return-type  # numpy-scalars


class RNNPredictor(Predictor):
  """A predictor implementing an RNN.

  This class doesn't inherit ScanPredictor because it is using its own haiku
  behaviour.
  """

  def __init__(self, unroll_factory, initial_state_factory):
    self._init_params, self._unroll = hk.transform(unroll_factory)
    _, self._initial_state = hk.transform(initial_state_factory)

  def initial_state(
      self,
      params: Any,
      rng: chex.PRNGKey,
      batch_size: int,
  ) -> Any:
    return self._initial_state(params, rng, batch_size)

  def init_params(self, rng: chex.PRNGKey, batch_init: chex.Array,
                  state_init: Optional[chex.Array]) -> hk.Params:
    return self._init_params(rng, batch_init, state_init)

  def unroll(
      self,
      params: Any,
      rng: chex.PRNGKey,
      batch: chex.Array,
      init_state: Any,
  ) -> chex.Array:
    return self._unroll(params, rng, x=batch, initial_state=init_state)


class ScanPredictor(Predictor, abc.ABC):
  """Implementation of predictors using jax.lax.scan and an update function.

  The prior is the output in the initial state.
  The only things the predictor has to provide is how to update its
  state, and what to output from this state.
  """

  @abc.abstractmethod
  def output_from_state(
      self,
      rng: chex.PRNGKey,
      state: chex.Array,
  ) -> chex.Array:
    """Returns what the predictor will output at a given state."""

  @abc.abstractmethod
  def update_state(
      self,
      rng: chex.PRNGKey,
      state: chex.Array,
      x: chex.Array,
  ) -> chex.Array:
    """Returns state at time t+1 based on state at time t."""

  def unroll(
      self,
      params: Any,
      rng: chex.PRNGKey,
      batch: chex.Array,
      init_state: chex.Array,
      jittable: bool = True,
  ) -> chex.Array:
    del params

    def scan_update_output(
        state: chex.Array,
        x: chex.Array,
    ) -> tuple[chex.Array, chex.Array]:
      new_state = self.update_state(rng, state, x)
      return new_state, self.output_from_state(rng, new_state)

    # Change to time-major layout since lax.scan unrolls over leading dimension.
    batch = batch.swapaxes(0, 1)
    if jittable:
      _, predictions = jax.lax.scan(scan_update_output, init_state, batch)
    else:
      state = init_state
      predictions = []
      for x in batch:
        state, pred = scan_update_output(state, x)
        predictions.append(pred)
      predictions = jnp.stack(predictions, axis=0)
    predictions = predictions.swapaxes(0, 1)
    return predictions


class OptimalPredictor(ScanPredictor, abc.ABC):
  """Abstract class for optimal predictors.

  The predictor must also define what prior and posterior distribution it uses.
  They are chosen carefully to mathematically match.
  Params are the parameters of the prior distribution.
  """

  def init_params(self, rng: chex.PRNGKey, batch_init: chex.Array,
                  state_init: chex.Array) -> hk.Params:
    raise NotImplementedError(
        'Optimal predictors do not provide parameter initialization.'
    )

  def initial_state(
      self,
      params: list[tuple[float, ...]],
      rng: chex.PRNGKey,
      batch_size: int,
  ) -> chex.Array:
    """Computes the initial state based on parameters.

    Args:
      params: the parameter tuples of the parameter distributions used to sample
        the true parameters of the observations.
      rng: an unused random key
      batch_size: the number of states returned

    Returns:
      state: a (batch_size, parameter_size) array
    """
    state = jnp.concatenate([jnp.array(p) for p in params], axis=0)
    state = jnp.stack([state] * batch_size, axis=0)
    return state.astype(jnp.float32)

  def unpack_state(self, state: chex.Array) -> tuple[chex.Array, ...]:
    """Splits a (batch_size, parameter_size) array to (batch_size, 1) elements.
    """
    state = jnp.expand_dims(state, axis=-1)
    return distributions.split_params(state)

  def pack_state(self, *state_elements: tuple[chex.Array, ...]) -> chex.Array:
    """Converts individual state elements into a single array.

    Args:
      *state_elements: parameter_size arguments, each of shape (batch_size, 1)

    Returns:
      state: array of shape (batch_size, parameter_size)
    """
    return jnp.concatenate(state_elements, axis=-1)


class OptimalCategoricalPredictor(OptimalPredictor):
  """Optimal bayesian predictor for Categorical distributions.

  State is (alpha_1, ..., alpha_n), parameters of a Dirichlet(n) distribution
  (conjugate prior).
  The outputs are the parameters for a Dirichlet(n) distribution.
  """

  def update_state(
      self,
      rng: chex.PRNGKey,
      state: chex.Array,
      x: chex.Array,
  ) -> chex.Array:
    return state + x

  def output_from_state(
      self,
      rng: chex.PRNGKey,
      state: chex.Array,
  ) -> chex.Array:
    parameters = state / jnp.sum(state, axis=-1, keepdims=True)
    return jnp.log(parameters)
