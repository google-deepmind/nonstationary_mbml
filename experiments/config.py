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

"""Base config for meta_learning experiments."""

import dataclasses
import math
from typing import Any, Optional, Sequence

from nonstationary_mbml import base_config as config_lib
from nonstationary_mbml.models import stack_rnn


@dataclasses.dataclass
class DataConfig(config_lib.DataConfig):
  """Config for the data distribution.

  Please distinguish between 'kwargs' which are values passed to the objects
  at initialization, and 'params' which are passed at sample time.
  """
  # The iid distribution is used to sample the sequence values.
  # For instance, a coin toss can be modeled with a Bernoulli distribution,
  # which is equivalent to a categorical distribution of dimension 2.
  # Thus, the values would be:
  #   - iid_distribution = `categorical`
  #   - iid_distribution_kwargs.size = 2
  iid_distribution: str = ''
  iid_distribution_kwargs: dict[str,
                                Any] = dataclasses.field(default_factory=dict)

  # The parameter distribution is used to sample the parameters of the above
  # iid distribution. For instance, a coin toss can be modeled with a Bernoulli
  # distribution that requires a single parameter `p`, which could be sample
  # from a beta distribution (the conjugate prior). This is equivalent to using
  # a categorical distribution of dimension 2 with a Dirichlet prior of
  # dimension 2.
  # Thus, the values would be:
  #   - parameter_distribution = 'dirichlet'
  #   - parameter_distributions_kwargs = {'size': 2}
  #   - parameter_distributions_params = (1., 1.)
  parameter_distribution: str = ''
  parameter_distribution_kwargs: dict[str, Any] = dataclasses.field(
      default_factory=dict)
  parameter_distribution_params: tuple[float, ...] = dataclasses.field(
      default_factory=tuple)

  # The trajector generator is used to generate the sequences.
  # See constants.py for all choices.
  trajectory_generator: str = 'static'
  trajectory_generator_kwargs: dict[str, Any] = dataclasses.field(
      default_factory=dict)


@dataclasses.dataclass
class EvalConfig(config_lib.EvalConfig):
  """Config for the evaluator."""
  # Sequence length used for evaluation, None means same as training.
  seq_length: Optional[int] = None
  # Chunk size to use at evaluation time. None means use the sequence length.
  chunk_length: Optional[int] = None
  # See constants.py for possible optimal predictors.
  optimal_predictors: Optional[Sequence[str]] = None
  optimal_predictors_kwargs: dict[str, dict[str, Any]] = dataclasses.field(
      default_factory=dict
  )
  data: Optional[DataConfig] = None  # Which data distrib to use for evaluation.


@dataclasses.dataclass
class ExperimentConfig(config_lib.ExperimentConfig):
  """Needed inheritance to avoid typing error."""
  name: str = '[MBML Nonstationary Distributions]'
  eval: EvalConfig = dataclasses.field(default_factory=EvalConfig)
  data: DataConfig = dataclasses.field(default_factory=DataConfig)


@dataclasses.dataclass
class ExperimentSweep(config_lib.ExperimentSweep):
  """Needed inheritance to avoid typing error."""
  base_config: ExperimentConfig = ExperimentConfig()


def post_process_config(config: ExperimentConfig) -> None:
  """Processes a config at launch time, in place."""
  # Setting the stack size for the stack-RNN.
  if config.model.model_type == 'rnn':
    if config.model.architecture_kwargs['core'] == stack_rnn.StackRNNCore:
      if config.model.architecture_kwargs['stack_size'] is None:
        config.model.architecture_kwargs['stack_size'] = config.train.seq_length

  # Setting the context size for the Transformer.
  if config.model.model_type == 'sliding_window_transformer':
    if config.model.architecture_kwargs['context_length'] is None:
      if config.train.gradient_chunk_length is None:
        config.model.architecture_kwargs[
            'context_length'] = config.train.seq_length
      else:
        config.model.architecture_kwargs[
            'context_length'] = config.train.gradient_chunk_length

  # Setting the eval data in case it was set to None.
  if config.eval.data is None:
    config.eval.data = config.data

  # Setting the eval length in case it was set to None.
  if config.eval.seq_length is None:
    config.eval.seq_length = config.train.seq_length

  if 'ptw' in config.eval.optimal_predictors:
    if config.eval.optimal_predictors_kwargs['ptw']['depth'] is None:
      config.eval.optimal_predictors_kwargs['ptw']['depth'] = math.ceil(
          math.log2(config.eval.seq_length)
      )
