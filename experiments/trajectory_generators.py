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

"""Trajectory generators used to train our RNNs to predict parameters.

These are basically our supervised datasets, generating data and loss functions.
"""

import abc
import contextlib
import functools
import math
import random
from typing import Any

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from nonstationary_mbml import base_constants
from nonstationary_mbml.experiments import distributions


def sample_distribution_parameters(
    rng: chex.PRNGKey,
    parameter_distribution: distributions.Distribution,
    parameter_distribution_params: tuple[float, ...],
    batch_size: int,
) -> chex.Array:
  """Creates a sample of parameters for the generation distribution.

  Args:
    rng: the PRNG key.
    parameter_distribution: the distribution from which parameters are sampled.
    parameter_distribution_params:  parameter tuples for the
      parameter_distribution.
    batch_size: the size of the batch that will be sampled from the generation
      distribution.

  Returns:
    an array of shape (batch_size, parameter_size), where parameter_size is the
      sum of the `feature_size`s of all the parameter distributions.
  """
  rng, key = jrandom.split(rng)

  params = parameter_distribution_params
  params = jnp.expand_dims(jnp.array(params), axis=0)
  # params now has shape (1, pd.parameter_size)
  params = jnp.repeat(params, batch_size, axis=0)
  # params now has shape (batch_size, pd.parameter_size)

  # add the sampled parameters to the output
  return parameter_distribution.sample(
      key, params, (batch_size, parameter_distribution.feature_size))


class TrajectoryGenerator(base_constants.DataGenerator):
  """Abstract class for trajectory generators."""

  def __init__(
      self,
      gen_distribution: distributions.Distribution,
      parameter_distribution: distributions.Distribution,
      parameter_distribution_params: tuple[float, ...],
  ):
    self.gen_distribution = gen_distribution
    self.parameter_size = gen_distribution.parameter_size

    if parameter_distribution.feature_size != self.parameter_size:
      raise ValueError(
          f"Incorrect number of parameter distibutions for "
          f"{gen_distribution.__class__}. "
          f"Expected {self.parameter_size}, "
          f"got {parameter_distribution.feature_size} output parameters.")

    if len(
        parameter_distribution_params) != parameter_distribution.parameter_size:
      raise ValueError(
          "Not enough parameters supplied for"
          f" {parameter_distribution.__class__}. Expected tuple of length"
          f" {parameter_distribution.parameter_size}, got"
          f" {parameter_distribution_params}."
      )

    self.parameter_distribution = parameter_distribution
    self.parameter_distribution_params = parameter_distribution_params

  @abc.abstractmethod
  def sample(
      self,
      rng: chex.PRNGKey,
      batch_size: int,
      seq_length: int,
  ) -> tuple[chex.Array, chex.Array]:
    """Samples a batch with randomly sampled true parameters.

    In the following, parameter_size is the number of parameters needed by the
    generative distribution (2 for a gaussian, 2 for uniform, etc.),
    feature_size is the number of output the generative distribution produces (1
    most of the time, n for categorical or dirichlet).
    This function is basically a BatchApply wrapper around parameters_for_gen.

    Args:
      rng: the random key to use in the random generation algorithm
      batch_size: the number of sequences to return
      seq_length: the length of the sequences to return

    Returns:
      batch: the batch of data, of shape (batch_size, seq_length, feature_size)
      parameters: the parameters used to sample this batch,
        of shape (batch_size, seq_length, parameter_size)
    """


class StaticTrajectoryGenerator(TrajectoryGenerator):
  """TG where the distribution parameters remain constant within a trajectory."""

  @functools.partial(jax.jit, static_argnums=(0, 2, 3))
  def sample(
      self,
      rng: chex.PRNGKey,
      batch_size: int,
      seq_length: int,
  ) -> tuple[chex.Array, chex.Array]:
    params_rng, gen_rng = jrandom.split(rng)

    parameters = sample_distribution_parameters(
        params_rng, self.parameter_distribution,
        self.parameter_distribution_params, batch_size)

    sample_shape = (batch_size, seq_length, self.gen_distribution.feature_size)
    batch = self.gen_distribution.sample(gen_rng, parameters, sample_shape)

    # The same parameters are used at each timestep, so we copy them along T:
    parameters_for_output = jnp.stack([parameters] * seq_length, axis=1)
    return batch, parameters_for_output


class DynamicTrajectoryGenerator(TrajectoryGenerator, abc.ABC):
  """TG where distribution parameters change within a trajectory."""

  @functools.partial(jax.jit, static_argnums=(0, 2, 3))
  def sample(
      self,
      rng: chex.PRNGKey,
      batch_size: int,
      seq_length: int,
  ) -> tuple[chex.Array, chex.Array]:
    params_rng, gen_rng = jrandom.split(rng, 2)
    parameters = self._parameters_for_gen(params_rng, batch_size, seq_length)
    partial_gen = functools.partial(
        self.gen_distribution.sample,
        rng=gen_rng,
        shape=(batch_size * seq_length, self.gen_distribution.feature_size))
    batch = hk.BatchApply(partial_gen)(parameters=parameters)
    return batch, parameters

  def _initial_state(self, rng: chex.PRNGKey, batch_size: int) -> chex.Array:
    init_parameters = sample_distribution_parameters(
        rng, self.parameter_distribution, self.parameter_distribution_params,
        batch_size)
    return init_parameters

  def _parameters_for_gen(
      self,
      rng: chex.PRNGKey,
      batch_size: int,
      seq_length: int,
  ) -> chex.Array:
    """Parameters used to generate trajectory.

    Args:
      rng: the random key to use in the random generation algorithm
      batch_size: the number of sequences to return
      seq_length: the length of the sequences to return

    Returns:
      parameters: the parameters used to generate a trajectory,
        of shape (batch_size, seq_length, parameter_size)
    """
    init_state = self._initial_state(rng, batch_size)
    keys = jrandom.split(rng, seq_length)

    def scan_update_params(state, t):
      return self.update_params(keys[t], state, t)

    _, params = jax.lax.scan(scan_update_params, init_state,
                             jnp.arange(seq_length))
    return params.swapaxes(0, 1)

  @abc.abstractmethod
  def update_params(
      self,
      rng: chex.PRNGKey,
      state_t: Any,
      t: int,
  ) -> tuple[Any, chex.Array]:
    """Updates params at each timestep t.

    Args:
      rng: random key
      state_t: the state of the generator at time t, used to produce the new
        parameters
      t: time

    Returns:
      new_state: the state at time t+1
      new_params: a (batch_size, parameter_size) tensor, parameters for time t+1
    """


class RegularShiftTrajectoryGenerator(DynamicTrajectoryGenerator):
  """Dynamic TG, samples new parameters every shift_period timesteps."""

  def __init__(
      self,
      shift_period: int,
      gen_distribution: distributions.Distribution,
      parameter_distribution: distributions.Distribution,
      parameter_distribution_params: tuple[float, ...],
  ):
    super().__init__(gen_distribution, parameter_distribution,
                     parameter_distribution_params)
    self._shift_period = shift_period

  def update_params(
      self,
      rng: chex.PRNGKey,
      state_t: Any,
      t: int,
  ) -> tuple[Any, chex.Array]:
    """Function passed to scan. State only contains last parameters here."""
    params = state_t
    batch_size = params.shape[0]
    samples = sample_distribution_parameters(rng, self.parameter_distribution,
                                             self.parameter_distribution_params,
                                             batch_size)
    new_params = jnp.where(t % self._shift_period == 0, samples, params)
    return new_params, new_params


class RandomShiftNoMemoryTrajectoryGenerator(DynamicTrajectoryGenerator):
  """Dynamic TG, samples new parameters every shift_period timesteps.

  Shift_period is now a random variable, which is sampled once at the beginning
  of the trajectory, and then resampled when the last sampled shift_period time
  is reached.
  """

  def __init__(
      self,
      shift_distribution: distributions.Distribution,
      shift_parameters: Any,
      gen_distribution: distributions.Distribution,
      parameter_distribution: distributions.Distribution,
      parameter_distribution_params: tuple[float, ...],
  ):
    super().__init__(gen_distribution, parameter_distribution,
                     parameter_distribution_params)
    self._shift_distribution = shift_distribution
    self._shift_parameters = shift_parameters

  def _sample_delta_shift_time(self, rng: chex.PRNGKey,
                               batch_size: int) -> chex.Array:
    delta_shift_time = sample_distribution_parameters(rng,
                                                      self._shift_distribution,
                                                      self._shift_parameters,
                                                      batch_size)
    delta_shift_time = jnp.clip(delta_shift_time, a_min=0)
    delta_shift_time = jnp.round(delta_shift_time)
    return delta_shift_time

  def _initial_state(self, rng: chex.PRNGKey, batch_size: int) -> chex.Array:
    init_params = super()._initial_state(rng, batch_size)
    initial_time = 0
    first_delta = self._sample_delta_shift_time(rng, batch_size)
    next_shift_time = initial_time + first_delta
    return init_params, next_shift_time

  def update_params(
      self,
      rng: chex.PRNGKey,
      state_t: Any,
      t: int,
  ) -> tuple[Any, chex.Array]:
    params, next_shift_time = state_t
    batch_size = params.shape[0]
    rng, rng2 = jrandom.split(rng)
    samples = sample_distribution_parameters(rng, self.parameter_distribution,
                                             self.parameter_distribution_params,
                                             batch_size)
    new_params = jnp.where(t == next_shift_time, samples, params)
    next_shift_time += jnp.where(
        t == next_shift_time, self._sample_delta_shift_time(rng2, batch_size),
        0)
    return (new_params, next_shift_time), new_params


def sample_ptw_switch_points(min_value: int, max_value: int) -> list[int]:
  """Returns switch points sampled from the PTW prior."""

  switch_points = list()

  if max_value <= min_value:
    return switch_points

  mean_value = (max_value + min_value) // 2

  if random.random() < 0.5:
    switch_points += sample_ptw_switch_points(min_value, mean_value)
    switch_points.append(mean_value)
    switch_points += sample_ptw_switch_points(mean_value + 1, max_value)

  return switch_points


def fixed_ptw_switch_points(seq_length: int) -> list[int]:
  """Returns switch points sampled from the PTW prior."""
  next_power_2 = 2**(math.ceil(math.log2(seq_length)))
  switch_points = sample_ptw_switch_points(1, next_power_2)
  switch_points = filter(lambda x: x <= seq_length - 1, switch_points)
  return list(sorted(switch_points))


class PTWRandomShiftTrajectoryGenerator(TrajectoryGenerator):
  """Dynamic TG using PTW prior."""

  def __init__(
      self,
      gen_distribution: distributions.Distribution,
      parameter_distribution: distributions.Distribution,
      parameter_distribution_params: tuple[float, ...],
  ):
    super().__init__(
        gen_distribution,
        parameter_distribution,
        parameter_distribution_params,
    )
    self._sample_data = jax.jit(
        self.gen_distribution.sample, static_argnums=(2,))
    self._sample_parameters = jax.jit(
        sample_distribution_parameters, static_argnums=(1, 2, 3))

  def sample(
      self,
      rng: chex.PRNGKey,
      batch_size: int,
      seq_length: int,
  ) -> tuple[chex.Array, chex.Array]:
    """Returns trajectories with switch points following the PTW prior."""
    switch_points = fixed_ptw_switch_points(seq_length)
    switch_points.append(seq_length)

    batch = []
    all_parameters = []
    rng_seq = hk.PRNGSequence(rng)
    last_switch = 0
    for switch_point in switch_points:
      length = switch_point - last_switch
      if length == 0:
        continue
      last_switch = switch_point
      parameters = self._sample_parameters(
          next(rng_seq), self.parameter_distribution,
          self.parameter_distribution_params, batch_size)
      all_parameters.append(jnp.stack([parameters] * length, axis=1))
      batch.append(
          self._sample_data(
              rng=next(rng_seq),
              shape=(batch_size, length, self.gen_distribution.feature_size),
              parameters=parameters))
    batch = jnp.concatenate(batch, axis=1)
    return batch, jnp.concatenate(all_parameters, axis=1)


@contextlib.contextmanager
def local_seed(seed: int):
  """Context manager to set local seed."""
  state = np.random.get_state()
  np.random.seed(seed)
  try:
    yield
  finally:
    np.random.set_state(state)


def iid_ptw_change_point(batch_size: int, seq_length: int) -> np.ndarray:
  """Generates IID switch points sampled as indicators from the PTW prior."""

  next_power_2 = 2**(math.ceil(math.log2(seq_length)))

  # Initialises an array to store the change points as boolean masks.
  change_points = np.zeros((batch_size, next_power_2), dtype=bool)

  def insert(idx: np.ndarray, left: np.ndarray, right: np.ndarray) -> None:
    """Splits the selected trajectories given their left and right ends."""

    # Returns if there is no trajectories to split.
    if idx.shape[0] == 0:
      return

    # For debugging:
    # assert (left <= right).all() and (left>=0).all() and (right<=T).all()
    assert (len(idx) == len(left) == len(right))

    mid = (left + right) // 2

    # Splits with prob. 0.5 if the segment length is nonzero.
    split = np.random.random(*idx.shape) < 0.5
    split = split & (left < right)

    # Updates the indices and endpoints, keeping those that had a split.
    idx = idx[split]
    left = left[split]
    right = right[split]
    mid = mid[split]

    # Sets a change point for trajectories that are split.
    change_points[idx, mid] = True

    # Recursively splits the left and right halves.
    insert(idx, left, mid)
    insert(idx, mid + 1, right)

  # Sets base condition.
  all_idx = np.arange(batch_size)
  left = np.ones(batch_size, np.uint64)
  right = np.ones(batch_size, np.uint64) * next_power_2

  insert(all_idx, left, right)
  change_points = change_points[:, :seq_length]
  return change_points


class IIDPTWRandomShiftTrajectoryGenerator(TrajectoryGenerator):
  """Draws iid trajectories from PTW prior."""

  def sample(
      self,
      rng: chex.PRNGKey,
      batch_size: int,
      seq_length: int,
  ) -> tuple[chex.Array, chex.Array]:
    """Returns iid trajectories with switch points following the PTW prior."""

    rng_seq = hk.PRNGSequence(rng)
    with local_seed(next(rng_seq)[0].item()):
      change_points = iid_ptw_change_point(batch_size, seq_length)

    num_switch = change_points.sum(-1)
    max_num_switch = num_switch.max()
    segments = change_points.cumsum(-1)
    all_parameters = np.zeros(change_points.shape +
                              (self.parameter_distribution.feature_size,))

    # Loops over all segments.
    # As the number of segments over the entire batch decays exponentially,this
    # loop is not likely to be large and grows logarithmically with the number
    # of time steps.
    for i in range(0, max_num_switch + 1):

      # A mask for the i'th segment over all trajectories.
      seg_idx = segments == i

      # Draws one sample for this segment and all trajectories.
      param_samples = sample_distribution_parameters(
          next(rng_seq), self.parameter_distribution,
          self.parameter_distribution_params, batch_size)

      # Expand parameters to sequence length. This is wasteful but is fast.
      param_samples = jnp.repeat(param_samples[:, None, :], seq_length, axis=1)

      # Sets the parameters using the mask.
      all_parameters[seg_idx] = param_samples[seg_idx]

    all_parameters = all_parameters.reshape(-1, all_parameters.shape[-1])
    batch = self.gen_distribution.sample(
        rng=next(rng_seq),
        shape=(batch_size * seq_length, self.gen_distribution.feature_size),
        parameters=all_parameters)

    batch = batch.reshape((batch_size, seq_length, -1))
    all_parameters = all_parameters.reshape((batch_size, seq_length, -1))
    return batch, all_parameters


@jax.jit
def _sample_categorical_batch(key: chex.PRNGKey, all_parameters: chex.Array):
  """Jittable function to sample one-hot categorical variables."""
  num_outcome = all_parameters.shape[-1]
  batch = jrandom.categorical(key, logits=jnp.log(all_parameters))
  batch = jnp.eye(num_outcome)[batch]
  return batch


class IIDPTWRandomShiftCategoricalTrajectoryGenerator(TrajectoryGenerator):
  """Draws iid categorical trajectories from PTW prior.

  This is faster than the general implementation above.
  """

  def sample(
      self,
      rng: chex.PRNGKey,
      batch_size: int,
      seq_length: int,
  ) -> tuple[chex.Array, chex.Array]:
    """Returns iid categorical trajectories with PTW switch points."""

    rng_seq = hk.PRNGSequence(rng)
    num_outcome = self.parameter_distribution.parameter_size
    assert num_outcome == self.parameter_distribution.feature_size

    with local_seed(next(rng_seq)[0].item()):
      change_points = iid_ptw_change_point(batch_size, seq_length)

      num_switch = change_points.sum(-1)
      max_num_switch = num_switch.max()
      segs = change_points.cumsum(-1)
      all_parameters = np.zeros(change_points.shape + (num_outcome,))

      for i in range(0, max_num_switch + 1):

        seg_idx = segs == i

        # Draws samples using numpy directly is fast.
        param_samples = np.random.dirichlet(
            np.ones(num_outcome) * 0.5, size=batch_size)
        param_samples = np.repeat(param_samples[:, None, :], seq_length, axis=1)
        all_parameters[seg_idx] = param_samples[seg_idx]

    # This is faster than np.random.multinomial.
    all_parameters = jnp.asarray(all_parameters)
    batch = _sample_categorical_batch(next(rng_seq), all_parameters)
    return batch, all_parameters


class LINTrajectoryGenerator(DynamicTrajectoryGenerator):
  """Draws IID trajectories from the linear model defined by Willems 1996."""

  def _initial_state(
      self,
      rng: chex.PRNGKey,
      batch_size: int,
  ) -> tuple[int, chex.Array]:
    init_parameters = super()._initial_state(rng, batch_size)
    return (jnp.ones((batch_size,)), init_parameters)

  def update_params(
      self,
      rng: chex.PRNGKey,
      state_t: Any,
      t: int,
  ) -> tuple[Any, chex.Array]:
    # Time steps are 1-indexed, but `t` starts at 0.
    t += 1
    # The state consists of the indices of the current intervals, denoted as
    # `t_c` in Willems 1996, and `params`, the parameters of the interval.
    t_c, params = state_t
    batch_size = params.shape[0]

    samples = sample_distribution_parameters(
        rng=rng,
        parameter_distribution=self.parameter_distribution,
        parameter_distribution_params=self.parameter_distribution_params,
        batch_size=batch_size,
    )
    new_params = jnp.where(jnp.expand_dims(t == t_c, 1), samples, params)

    # The probability of observing a switch point decays harmonically and
    # depends on the time `t` and the beginning of the current interval `t_c`.
    is_switch_point = jrandom.bernoulli(rng, 0.5 / (t - t_c + 1))
    # If we observe a switch point, we set `t_c` to the next time step to
    # indicate the start of a new interval.
    t_c = jnp.where(is_switch_point, t + 1, t_c)

    return (t_c, new_params), new_params
