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

"""LiveAndDie predictor classes."""

import copy
from typing import Any

import chex
import haiku as hk
import numpy as np

from nonstationary_mbml import predictors


@chex.dataclass
class LADRecords:
  """The records of a linear LAD estimator state.

  Attributes:
    timestep: The records' timestep (shared across records).
    reset_timestep: The records' reset timesteps, of shape `(batch_size, time)`.
    counts: The records' counts, of shape `(batch_size, time, 2)`.
    log_prob: The records' log probability, of shape `(batch_size, time)`.
  """

  timestep: int
  reset_timestep: chex.Array
  counts: chex.Array
  log_prob: chex.Array


@chex.dataclass
class LADState:
  """The state of the linear LAD estimator.

  Attributes:
    log_prob: The state's log probabilities, of shape `(batch_size)`.
    records: The state's records.
    timestep: The state's global timestep.
  """
  log_prob: chex.Array
  records: LADRecords
  timestep: int


class LADPredictor(predictors.Predictor):
  """Linear Live and Die estimator predictor.

  WARNING:
    LAD outputs a prediction before seeing the first token, which is
    inconsistent with our predictor interface. Thus, we omit the first
    prediction and append a dummy output at the end.
  """

  def initial_state(
      self,
      params: Any,
      rng: chex.PRNGKey,
      batch_size: int,
  ) -> Any:
    """Returns the initial LADState."""
    del params, rng
    return LADState(
        log_prob=np.zeros((batch_size,), dtype=np.float32),
        records=LADRecords(
            timestep=0,
            reset_timestep=np.zeros((batch_size, 0), dtype=np.uint8),
            counts=np.zeros((batch_size, 0, 2), dtype=np.uint8),
            log_prob=np.zeros((batch_size, 0), dtype=np.float32),
        ),
        timestep=0,
    )

  def init_params(
      self,
      rng: chex.PRNGKey,
      batch_init: chex.Array,
      state_init: chex.Array,
  ) -> hk.Params:
    return dict()

  def unroll(
      self,
      params: Any,
      rng: chex.PRNGKey,
      batch: chex.Array,
      init_state: Any,
  ) -> chex.Array:
    del params, rng

    def scan_update_output(
        state: LADState,
        x: chex.Array,
    ) -> tuple[chex.Array, chex.Array]:
      pred = self.output_from_state(state)
      new_state = self.update(state, x.argmax(-1))
      return new_state, pred

    # Change to time-major layout since we unroll over the leading dimension.
    batch = batch.swapaxes(0, 1)
    state = copy.deepcopy(init_state)
    predictions = list()

    for x in batch:
      state, pred = scan_update_output(state, x)
      predictions.append(pred)

    predictions = np.stack(predictions, axis=0)

    # LAD outputs a prediction before seeing the first token, which is
    # inconsistent with our predictor interface. Thus, we omit the first
    # prediction and append a dummy output at the end.
    predictions = np.concatenate(
        [predictions[1:],
         np.full_like(predictions[:1], np.nan)], axis=0)

    predictions = predictions.swapaxes(0, 1)

    return predictions

  def output_from_state(self, state: LADState) -> chex.Array:
    """Returns the log probability of the next symbol being 0 or 1."""
    log_prob = self.log_prob(state, np.ones_like(state.log_prob, dtype=int))
    return np.stack([np.log(1.0 - np.exp(log_prob)), log_prob], axis=-1)

  def compute_log_marginal(self, state: LADState) -> LADState:
    """Returns the state updated with new log marginal probability."""
    chex.assert_axis_dimension_gt(
        state.records.counts, 1, 0, exception_type=ValueError
    )
    state.log_prob = np.logaddexp.reduce(state.records.log_prob, axis=1)
    return state

  def transition_probability(
      self,
      new_timestep: int,
      new_reset_timestep: chex.Array,
      old_timestep: int,
      old_reset_timestep: chex.Array,
      state_timestep: int,
  ) -> chex.Array:
    """Returns the transition probability, shape `(batch_size, time_steps)`."""
    n = 0.5

    if np.any(new_timestep == new_reset_timestep):
      assert new_timestep == state_timestep
    else:
      n += old_timestep - old_reset_timestep

    d = old_timestep - old_reset_timestep + 1.0

    return np.log(n / d)

  def update(self, state: LADState, symbol: chex.Array) -> LADState:
    """Returns the updated state after processing `symbol`.

    Args:
      state: The current state of the linear LAD estimator.
      symbol: The next symbol in the sequence, of shape `(batch_size,)`
    """
    state.timestep += 1
    state = self.compute_state_probs_linear(state, symbol)
    state = self.compute_log_marginal(state)
    return state

  def log_prob(self, state: LADState, symbol: chex.Array) -> chex.Array:
    """Returns the log probability of `symbol` being the next symbol.

    Args:
      state: The current state of the linear LAD estimator.
      symbol: The next symbol in the sequence, of shape `(batch_size,)`.
    """
    new_state = copy.deepcopy(state)
    self.update(new_state, symbol)
    log_prob = new_state.log_prob - state.log_prob
    assert np.all(log_prob <= 0)
    return log_prob

  def compute_state_probs_linear(
      self,
      state: LADState,
      symbol: chex.Array,
  ) -> LADState:
    """Returns the updated state by after computing the coding probabilities.

    Args:
      state: The current state of the linear LAD estimator.
      symbol: The next symbol in the sequence, of shape `(batch_size,)`.
    """
    batch_size = state.log_prob.shape[0]
    new_records = LADRecords(
        timestep=state.timestep,
        reset_timestep=np.tile(
            np.arange(1, state.timestep + 1), (batch_size, 1)
        ),
        counts=np.zeros((batch_size, state.timestep, 2), dtype=np.uint8),
        log_prob=np.zeros((batch_size, state.timestep), dtype=np.float32),
    )

    time_steps = state.records.log_prob.shape[1]
    time_range = range(time_steps)
    batch_range = range(batch_size)

    if 0 < time_steps:
      idx = state.records.reset_timestep - 1
      indices = np.stack((np.tile(batch_range, (time_steps, 1)).T, idx), axis=1)
      chex.assert_trees_all_equal(
          new_records.reset_timestep[indices[:, 0], indices[:, 1]],
          state.records.reset_timestep,
          exception_type=ValueError,
      )

      n_grid = np.dstack(np.meshgrid(batch_range, time_range))
      symbol_indices = np.tile(symbol, (1, time_steps, 1))
      n_indices = np.concatenate((n_grid, symbol_indices.transpose(1, 2, 0)),
                                 axis=-1).transpose(1, 2, 0)

      n = (
          state.records.counts[
              n_indices[:, 0], n_indices[:, 1], n_indices[:, 2]
          ]
          + 0.5
      )
      d = (
          new_records.timestep
          - new_records.reset_timestep[indices[:, 0], indices[:, 1]]
          + 1.0
      )
      r = np.log(n / d)

      rec_indices = np.concatenate((indices, symbol_indices.transpose(2, 0, 1)),
                                   axis=1)

      trans_prob = self.transition_probability(
          new_timestep=new_records.timestep,
          new_reset_timestep=new_records.reset_timestep[
              indices[:, 0], indices[:, 1]
          ],
          old_timestep=state.records.timestep,
          old_reset_timestep=state.records.reset_timestep,
          state_timestep=state.timestep,
      )
      new_records.log_prob[indices[:, 0], indices[:, 1]] = (
          state.records.log_prob + trans_prob + r
      )
      new_records.counts[indices[:, 0], indices[:, 1]] = state.records.counts
      new_records.counts[
          rec_indices[:, 0], rec_indices[:, 1], rec_indices[:, 2]
      ] += 1

    # Now handle the (x, x) state.
    idx = state.timestep - 1
    new_records.counts[batch_range, idx, symbol] += 1

    if time_steps == 0:
      new_records.log_prob[:, idx] = np.log(0.5)
    else:
      new_records.log_prob[:, idx] = np.logaddexp.reduce(
          state.records.log_prob
          + self.transition_probability(
              new_timestep=new_records.timestep,
              new_reset_timestep=new_records.reset_timestep[:, idx],
              old_timestep=state.records.timestep,
              old_reset_timestep=state.records.reset_timestep,
              state_timestep=state.timestep,
          )
          + np.log(0.5),
          axis=1,
      )

    state.records = new_records

    return state
