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

"""Factories to initialise predictors based on different neural architectures.

To create a new predictor you need to register a function that returns a
predictor based on a configuration and output size. The predictor factory should
be uniquely identified by its name.
"""

import functools
from typing import Any, Callable, Optional, Sequence, Type

import einops
import haiku as hk
import jax
import jax.numpy as jnp

from nonstationary_mbml import predictors
from nonstationary_mbml.models import basic
from nonstationary_mbml.models import positional_encodings as pos_encs_lib
from nonstationary_mbml.models import transformer


_Config = dict[str, Any]

# A function that can be used to create a Predictor.
_PredictorFactory = Callable[[int, _Config], predictors.Predictor]

# Maps names to the correct agent factory.
PREDICTOR_FACTORIES: dict[str, _PredictorFactory] = {}


def _register_predictor_factory(
    name: str,
) -> Callable[[_PredictorFactory], _PredictorFactory]:
  """Decorator for registering a function as a factory using the `name` id."""
  if name.lower() != name:
    raise ValueError(
        'Please use lower-case names to register the predictor factories.'
    )

  def wrap(fn: _PredictorFactory) -> _PredictorFactory:
    PREDICTOR_FACTORIES[name] = fn
    return fn

  return wrap


class MLPWrappedRNN(hk.RNNCore):
  """A wrapper for RNNs to add MLP layers."""

  def __init__(
      self,
      core: Type[hk.RNNCore],
      before_mlp_layers: Sequence[int] = (),
      after_mlp_layers: Sequence[int] = (),
      **core_kwargs
  ):
    super().__init__()
    self._core = core(**core_kwargs)
    self._before_mlp = hk.nets.MLP(before_mlp_layers)
    self._after_mlp = hk.nets.MLP(after_mlp_layers)

  def __call__(self, inputs: Any, prev_state: Any) -> tuple[Any, Any]:
    before_mlp_output = self._before_mlp(inputs)
    core_output, next_state = self._core(before_mlp_output, prev_state)
    after_mlp_output = self._after_mlp(core_output)
    return after_mlp_output, next_state

  def initial_state(self, batch_size: Optional[int]) -> Any:
    return self._core.initial_state(batch_size)


class SlidingWindowTransformer:
  """A Transformer model that can handle large histories using a sliding window.
  """

  def __init__(self, output_size: int, context_length: int,
               architecture_config: _Config):
    self._transformer = transformer.make_transformer_encoder(
        output_size=output_size,
        return_all_outputs=True,
        causal_masking=True,
        **architecture_config,
    )
    self._context_length = context_length
    self._output_size = output_size

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    batch_size, history_len, num_features = x.shape[:3]
    history_batch_size = history_len // self._context_length
    x_batched_history = einops.rearrange(
        x,
        'b (h c) f -> b h c f',
        b=batch_size,
        h=history_batch_size,
        c=self._context_length,
        f=num_features,
    )
    out = jax.vmap(self._transformer, in_axes=1, out_axes=1)(x_batched_history)
    return einops.rearrange(
        out,
        'b h c o -> b (h c) o',
        b=batch_size,
        h=history_batch_size,
        c=self._context_length,
        o=self._output_size,
    )


def _make_rnn_predictor(
    output_size: int,
    architecture_config: _Config,
    rnn_core: Type[hk.RNNCore],
) -> predictors.Predictor:
  """Returns an RNN predictor based on config."""
  unroll_factory = basic.make_rnn(
      output_size=output_size,
      rnn_core=rnn_core,
      return_all_outputs=True,
      return_all_states=True,
      input_window=1,
      **architecture_config,
  )

  def initial_state_factory(batch_size: int):
    return rnn_core(**architecture_config).initial_state(batch_size)

  return predictors.RNNPredictor(unroll_factory, initial_state_factory)


_register_predictor_factory('rnn')(
    functools.partial(_make_rnn_predictor, rnn_core=MLPWrappedRNN)
)


@_register_predictor_factory('transformer')
def _make_transformer_predictor(
    output_size: int, architecture_config: _Config
) -> predictors.Predictor:
  """Returns Transformer predictor based on config."""
  positional_encodings_params = {}
  if 'positional_encodings_params' in architecture_config:
    positional_encodings_params = architecture_config[
        'positional_encodings_params'
    ]
  architecture_config['positional_encodings_params'] = (
      pos_encs_lib.POS_ENC_PARAMS_TABLE[
          architecture_config['positional_encodings']
      ](**positional_encodings_params)
  )
  architecture_config['positional_encodings'] = pos_encs_lib.POS_ENC_TABLE[
      architecture_config['positional_encodings']
  ]
  predictor = transformer.make_transformer_encoder(
      output_size=output_size,
      return_all_outputs=True,
      causal_masking=True,
      **architecture_config,
  )
  return predictors.InContextPredictor(predictor)


@_register_predictor_factory('sliding_window_transformer')
def _make_sliding_window_transformer_predictor(
    output_size: int, architecture_config: _Config
) -> predictors.Predictor:
  """Returns Transformer predictor based on config."""
  positional_encodings_params = {}
  if 'positional_encodings_params' in architecture_config:
    positional_encodings_params = architecture_config[
        'positional_encodings_params'
    ]
  architecture_config['positional_encodings_params'] = (
      pos_encs_lib.POS_ENC_PARAMS_TABLE[
          architecture_config['positional_encodings']
      ](**positional_encodings_params)
  )
  architecture_config['positional_encodings'] = pos_encs_lib.POS_ENC_TABLE[
      architecture_config['positional_encodings']
  ]

  context_len = architecture_config['context_length']
  model_kwargs = {
      k: v for k, v in architecture_config.items() if k != 'context_length'
  }
  predictor = SlidingWindowTransformer(output_size, context_len, model_kwargs)
  return predictors.InContextPredictor(predictor)
