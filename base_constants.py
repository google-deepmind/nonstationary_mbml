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

"""Constants for online_learning."""

import abc
from typing import Any

import chex
from typing_extensions import Protocol

from nonstationary_mbml import base_config as config_lib
from nonstationary_mbml import predictors


class DataGenerator(abc.ABC):
  """Abstract data generation class."""

  @abc.abstractmethod
  def sample(
      self,
      rng: chex.PRNGKey,
      batch_size: int,
      seq_length: int,
  ) -> tuple[chex.Array, chex.Array]:
    """Samples a batch of data.

    Args:
      rng: The random key to use in the random generation algorithm.
      batch_size: The number of sequences to return.
      seq_length: The length of the sequences to return.

    Returns:
      batch: The batch of data, of shape (batch_size, seq_length, feature_size).
      parameters: The parameters used to sample this batch. Can just be the
        random seed if not applicable.
    """


class DataGeneratorBuilder(Protocol):

  def __call__(self, config: config_lib.DataConfig) -> DataGenerator:
    """Returns a data generator from a config."""


class Evaluator(abc.ABC):
  """Abstract evaluator class."""

  @abc.abstractmethod
  def step(
      self, predictor_params: Any, predictor_state: Any, rng: chex.PRNGKey
  ) -> dict[str, Any]:
    """Evaluates the predictor and returns a log dict."""


class EvaluatorBuilder(Protocol):

  def __call__(
      self, predictor: predictors.Predictor, eval_config: config_lib.EvalConfig
  ) -> Evaluator:
    """Returns an evaluator from a training predictor, a loss_fn and a config.

    Args:
      predictor: The predictor being trained. Most likely a neural network.
        Parameters will be passed in the main loop, not when building the
        evaluator.
      eval_config: The evaluation config. Depends on the experiment being run.
    """
