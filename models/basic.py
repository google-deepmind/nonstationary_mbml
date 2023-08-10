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

"""Basic neural network models.

This file contains builders for different simple models:
- MLP
- CNN
- RNN/LSTM
"""

from typing import Any, Callable, Mapping, Optional, Sequence

import chex
import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp


def make_cnn(
    channel_widths: Sequence[int],
    kernel_shape: tuple[int, int],
    output_size: int,
    mlp_widths: Optional[Sequence[int]] = None,
) -> Callable[[chex.Array], chex.Array]:
  """Returns a CNN model with extra MLP layers."""

  def model(x: chex.Array) -> chex.Array:
    for channels in channel_widths:
      x = hk.Conv2D(channels, kernel_shape=kernel_shape)(x)
      x = jnn.relu(x)
    x = hk.Flatten()(x)

    if mlp_widths is not None:
      for width in mlp_widths:
        x = hk.Linear(width)(x)
        x = jnn.relu(x)

    return hk.Linear(output_size)(x)

  return model


def make_mlp(hidden_layers_sizes: Sequence[int],
             output_size: int) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """Returns an MLP model."""

  def mlp(x):
    flattened_in = hk.Flatten()(x)
    layer_sizes = tuple(hidden_layers_sizes) + (output_size,)
    return hk.nets.MLP(layer_sizes)(flattened_in)

  return mlp


def make_rnn(
    output_size: int,
    rnn_core: type[hk.RNNCore],
    return_all_outputs: bool = False,
    return_all_states: bool = False,
    input_window: int = 1,
    **rnn_kwargs: Mapping[str, Any],
) -> Callable[[jnp.ndarray], Any]:
  """Returns an RNN model, not haiku transformed.

  Only the last output in the sequence is returned. A linear layer is added to
  match the required output_size.

  Args:
    output_size: The output size of the model.
    rnn_core: The haiku RNN core to use. LSTM by default.
    return_all_outputs: Whether to return the whole sequence of outputs of the
      RNN, or just the last one.
    return_all_states: Whether to return all the intermediary RNN states.
    input_window: The number of tokens that are fed at once to the RNN.
    **rnn_kwargs: Kwargs to be passed to the RNN core.
  """

  def rnn_model(
      x: jnp.ndarray, initial_state: Optional[Any] = None
  ) -> jnp.ndarray:
    core = rnn_core(**rnn_kwargs)
    if initial_state is None:
      initial_state = core.initial_state(x.shape[0])

    batch_size, seq_length, embed_size = x.shape
    if seq_length % input_window != 0:
      x = jnp.pad(x, ((0, 0), (0, input_window - seq_length % input_window),
                      (0, 0)))
    new_seq_length = x.shape[1]
    x = jnp.reshape(
        x,
        (batch_size, new_seq_length // input_window, input_window, embed_size))
    x = hk.Flatten(preserve_dims=2)(x)

    output, all_states = hk.dynamic_unroll(
        core, x, initial_state, time_major=False, return_all_states=True)
    output = jnp.reshape(output, (batch_size, new_seq_length, output.shape[-1]))

    if not return_all_outputs:
      output = output[:, -1, :]  # (batch, time, alphabet_dim)
    output = jnn.relu(output)
    output = hk.Linear(output_size)(output)

    if not return_all_states:
      return output
    else:
      return output, all_states  # pytype: disable=bad-return-type  # jax-ndarray

  return rnn_model
