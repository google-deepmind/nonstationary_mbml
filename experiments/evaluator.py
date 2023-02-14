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

"""Evaluation for the meta learning experiments."""

import copy
import math
from typing import Any, Optional

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tree

from nonstationary_mbml import agents
from nonstationary_mbml import base_constants
from nonstationary_mbml.experiments import config as config_lib
from nonstationary_mbml.experiments import constants as meta_learning_constants
from nonstationary_mbml.experiments import trajectory_generators as tg


@jax.jit
def _compute_true_cross_entropy(logits: chex.Array,
                                distribution_params: chex.Array) -> chex.Array:
  logits = logits[:, :-1]
  distribution_params = distribution_params[:, 1:]
  return optax.softmax_cross_entropy(logits, distribution_params)


class MetaLearningEvaluator(base_constants.Evaluator):
  """Evaluator for meta learning."""

  def __init__(
      self,
      agent: agents.Agent,
      data_generator: tg.TrajectoryGenerator,
      batch_size: int,
      seq_length: int,
      optimal_agents: Optional[dict[str, agents.Agent]] = None,
      chunk_length: Optional[int] = None,
  ) -> None:
    self._agent = agent
    self._optimal_agents = optimal_agents
    self._data_generator = data_generator
    self._batch_size = batch_size
    self._seq_length = seq_length
    self._chunk_length = chunk_length

    @jax.jit
    def _dist_entropy(distribution_params: chex.Array) -> chex.Array:
      dist_entropy = hk.BatchApply(data_generator.gen_distribution.entropy)(
          distribution_params)
      return jnp.squeeze(dist_entropy, axis=-1)

    self._dist_entropy = _dist_entropy

    if optimal_agents is not None:
      self._optimal_agents_init_state = dict()

      for agent_name, optimal_agent in self._optimal_agents.items():
        self._optimal_agents_init_state[
            agent_name] = optimal_agent.initial_state(
                None, None, batch_size=batch_size)

  def step(self, agent_params: Any, agent_state: Any,
           rng: chex.PRNGKey) -> dict[str, Any]:
    """Evaluates the agent and returns a log dict."""
    rngs = hk.PRNGSequence(rng)
    data_batch, distribution_params = self._data_generator.sample(
        rng, self._batch_size, self._seq_length)
    if self._chunk_length is None:
      logits, _ = self._agent.unroll(agent_params, next(rngs), data_batch,
                                     agent_state)
    else:
      final_logits = []
      agent_state = copy.deepcopy(agent_state)
      for i in range(math.ceil(self._seq_length / self._chunk_length)):
        data_chunk = data_batch[:, i * self._chunk_length:(i + 1) *
                                self._chunk_length]
        logits, states = self._agent.unroll(agent_params, next(rngs),
                                            data_chunk, agent_state)
        if states is not None:
          agent_state = tree.map_structure(lambda x: x[:, -1], states)
        else:
          agent_state = None
        final_logits.append(logits)
      logits = np.concatenate(final_logits, axis=1)
    true_entropy = self._dist_entropy(distribution_params[:, 1:])
    instantaneous_regret = _compute_true_cross_entropy(
        logits, distribution_params) - true_entropy
    mean_regret = jnp.mean(instantaneous_regret)
    cumulative_regret = jnp.mean(jnp.sum(instantaneous_regret, axis=1))

    if self._optimal_agents is not None:
      optimal_logits = dict()
      optimal_cumulative_regret = dict()
      optimal_instantaneous_regret = dict()

      for agent_name, optimal_agent in self._optimal_agents.items():
        init_state = copy.deepcopy(self._optimal_agents_init_state[agent_name])
        optimal_logits[agent_name] = optimal_agent.unroll(
            params=None,
            rng=next(rngs),
            batch=data_batch,
            init_state=init_state)
        optimal_instantaneous_regret[agent_name] = _compute_true_cross_entropy(
            optimal_logits[agent_name], distribution_params) - true_entropy
        optimal_cumulative_regret[agent_name] = jnp.mean(
            jnp.sum(optimal_instantaneous_regret[agent_name], axis=1))

    log_dict = {}
    log_dict['logits'] = logits
    log_dict['mean_regret'] = mean_regret
    log_dict['cumulative_regret'] = cumulative_regret

    if self._optimal_agents is not None:
      for agent_name, optimal_agent in self._optimal_agents.items():
        log_dict[f'optimal_cumulative_regret/{agent_name}'] = (
            optimal_cumulative_regret[agent_name]
        )
        log_dict[f'cumulative_regret_above_optimal/{agent_name}'] = (
            cumulative_regret - optimal_cumulative_regret[agent_name]
        )

    return log_dict


# The following function follows the protocol base_constants.EvaluatorBuilder.
def build_evaluator(agent: agents.Agent,
                    config: config_lib.EvalConfig) -> MetaLearningEvaluator:
  """Returns an evaluator from a meta_learning eval config."""
  if config.optimal_agents is not None:
    optimal_agents = dict()
    for optimal_agent in config.optimal_agents:
      optimal_agents[optimal_agent] = meta_learning_constants.OPTIMAL_AGENTS[
          optimal_agent](**config.optimal_agents_kwargs[optimal_agent])
  else:
    optimal_agents = None
  data_generator = meta_learning_constants.build_data_generator(config.data)
  return MetaLearningEvaluator(
      agent=agent,
      data_generator=data_generator,
      batch_size=config.batch_size,
      seq_length=config.seq_length,
      optimal_agents=optimal_agents,
      chunk_length=config.chunk_length,
  )
