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

"""PTW predictor classes."""

import abc
import copy
from typing import Any, Sequence

import chex
import numpy as np

from nonstationary_mbml import predictors


def _leading_zeros(x):
  """Compute the number of leading zeros of x in binary."""

  if x == 0:
    return 32
  n = 0
  if x <= 0x0000FFFF:
    n = n + 16
    x = x << 16
  if x <= 0x00FFFFFF:
    n = n + 8
    x = x << 8
  if x <= 0x0FFFFFFF:
    n = n + 4
    x = x << 4
  if x <= 0x3FFFFFFF:
    n = n + 2
    x = x << 2
  if x <= 0x7FFFFFFF:
    n = n + 1
  return n


def _mscb(d, t):
  """Most significant change bit between t-1 and t-2 in d-length binary."""
  if t == 1:
    return 0
  assert t <= 2**d
  return d - (32 - _leading_zeros((t - 1) ^ (t - 2)))


# given log(x) and log(y), compute log(x+y). uses the following identity:
#   log(x + y) = log(x) + log(1 + y/x) = log(x) + log(1+exp(log(y)-log(x)))
def array_log_add(x: chex.Array, y: chex.Array) -> chex.Array:
  idx = x > y
  x[idx], y[idx] = y[idx], x[idx]

  rval = y - x
  idx2 = rval < 100.0

  rval[idx2] = np.log1p(np.exp(rval[idx2]))
  rval += x
  return rval


class KT:
  """KT class."""

  def __init__(self):
    self.counts = [0, 0]
    self.marg = 0.0

  def _prob(self, x: chex.Array):
    num = self.counts[x.argmax()] + 0.5
    den = self.counts[0] + self.counts[1] + 1
    return num / den

  def update(self, x: chex.Array) -> None:
    assert len(x) == 2
    assert x.sum() == 1
    self.marg += np.log(self._prob(x))
    self.counts[x.argmax()] += 1


class ArrayKT:
  """ArrayKT class."""

  def __init__(self, batch_size: int, depth: int) -> None:
    self.depth = depth
    self.batch_size = batch_size
    self.counts = np.zeros((batch_size, depth, 2), dtype=np.int64)
    self.marg = np.zeros((batch_size, depth))

  def _prob(self, x: chex.Array, d: int) -> chex.Array:
    batch_size = x.shape[0]
    num = self.counts[range(batch_size), d, x.argmax(-1)] + 0.5
    den = self.counts[:, d].sum(-1) + 1
    return num / den

  def update(self, x: chex.Array, d: int) -> None:
    self.marg[:, d] += np.log(self._prob(x, d))
    self.counts[:, d] += x

  def reset(self, drange: Sequence[int]) -> None:
    self.counts[:, drange] = 0
    self.marg[:, drange] = 0.0


@chex.dataclass
class PTWState:
  b: chex.Array
  w: chex.Array
  kt: ArrayKT
  t: int


class PTWPredictor(predictors.Predictor, abc.ABC):
  """Partition tree weighting predictor.

  WARNING:
    PTW outputs a prediction before seeing the first token, which is
    inconsistent with our predictor interface. Thus, we omit the first
    prediction and append a dummy output at the end.

  Attributes:
    d: depth
  """

  def __init__(self, depth: int) -> None:
    self.d = depth

  def init_params(self, *args, **kwargs):
    pass

  def initial_state(self, params: chex.Array, rng: chex.Array,
                    batch_size: int) -> PTWState:
    return PTWState(
        b=np.zeros((batch_size, self.d + 1)),
        w=np.zeros((batch_size, self.d + 1)),
        kt=ArrayKT(batch_size, self.d + 1),
        t=0,
    )

  def update_state(self, rng: chex.PRNGKey, state: chex.Array,
                   x: chex.Array) -> chex.Array:
    d = self.d

    t = state.t  # pytype: disable=attribute-error  # numpy-scalars
    i = _mscb(d, t + 1)

    state.b[:, i] = state.w[:, i + 1]  # pytype: disable=attribute-error  # numpy-scalars
    # Doing the reset
    state.b[:, i + 1:d + 1] = 0  # pytype: disable=attribute-error  # numpy-scalars
    state.w[:, i + 1:d + 1] = 0  # pytype: disable=attribute-error  # numpy-scalars

    state.kt.reset(range(i + 1, d + 1))  # pytype: disable=attribute-error  # numpy-scalars

    state.kt.update(x, d)  # pytype: disable=attribute-error  # numpy-scalars
    state.w[:, d] = state.kt.marg[:, d]  # pytype: disable=attribute-error  # numpy-scalars

    for j in range(d - 1, -1, -1):
      state.kt.update(x, j)  # pytype: disable=attribute-error  # numpy-scalars
      lhs = np.log(0.5) + state.kt.marg[:, j]  # pytype: disable=attribute-error  # numpy-scalars
      rhs = np.log(0.5) + state.w[:, j + 1] + state.b[:, j]  # pytype: disable=attribute-error  # numpy-scalars
      wi = array_log_add(lhs, rhs)
      state.w[:, j] = wi  # pytype: disable=attribute-error  # numpy-scalars

    state.t = state.t + 1  # pytype: disable=attribute-error  # numpy-scalars

    return state

  def output_from_state(self, rng: chex.PRNGKey,
                        state: chex.Array) -> chex.Array:

    wx = state.w[:, 0]  # pytype: disable=attribute-error  # numpy-scalars
    cp_state = copy.deepcopy(state)

    batch_size = wx.shape[0]
    ones = np.repeat(np.asarray([[1, 0]]), batch_size, axis=0)
    cp_state = self.update_state(rng, cp_state, ones)

    output = cp_state.w[:, 0] - wx  # pytype: disable=attribute-error  # numpy-scalars
    output = np.stack([output, np.log(1 - np.exp(output))], axis=-1)

    return output

  def unroll(
      self,
      params: Any,
      rng: chex.PRNGKey,
      batch: chex.Array,
      init_state: chex.Array,
  ) -> chex.Array:
    # Params are not used in this predictor.
    del params

    def scan_update_output(
        state: chex.Array,
        x: chex.Array,
    ) -> tuple[chex.Array, chex.Array]:
      pred = self.output_from_state(rng, state)
      new_state = self.update_state(rng, state, x)
      return new_state, pred

    batch = batch.astype(np.int64)
    # Change to time-major layout since lax.scan unrolls over leading dimension.
    batch = batch.swapaxes(0, 1)
    state = copy.deepcopy(init_state)
    predictions = []
    for x in batch:
      state, pred = scan_update_output(state, x)
      predictions.append(pred)
    predictions = np.stack(predictions, axis=0)

    # PTW outputs a prediction before seeing the first token, which is
    # inconsistent with our predictor interface. Thus, we omit the first
    # prediction and append a dummy output at the end.
    predictions = np.concatenate(
        [predictions[1:],
         np.full_like(predictions[:1], np.nan)], axis=0)

    predictions = predictions.swapaxes(0, 1)

    return predictions
