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

"""Implements the training loop that performs parameter updates."""

import copy
import functools
import math
import random
from typing import Any, Callable, Optional

import chex
import haiku as hk
import jax
from jax import numpy as jnp
from jax import random as jrandom
import numpy as np
import optax
import tqdm
import tree

from nonstationary_mbml import base_config as config_lib
from nonstationary_mbml import base_constants
from nonstationary_mbml import predictor_factories
from nonstationary_mbml import predictors


def _make_whole_loss_fn(
    predictor: predictors.Predictor,
) -> Callable[[hk.Params, chex.PRNGKey, chex.Array, Any], tuple[float, Any]]:
  """Returns the loss function for update_parameters_whole_sequence."""

  def loss_fn(params: hk.Params, rng: chex.PRNGKey, inputs: chex.Array,
              init_state: Any) -> tuple[float, Any]:
    """Returns the loss for the model and the last state.

    Args:
      params: The parameters of the model, usually a neural network.
      rng: The random seed used to unroll the model (dropout for instance).
      inputs: The input array, of shape (B, T, F). B batch dimension, T time
        dimension, F feature dimension.
      init_state: The initial state of the model. Can be anything, but usually
        will be an ArrayTree (like LSTM state).
    """
    output, states = predictor.unroll(params, rng, inputs, init_state)
    last_state = _get_last_state(states)
    predictions = output[:, :-1]
    targets = inputs[:, 1:]
    losses = optax.softmax_cross_entropy(predictions, targets)
    return jnp.mean(losses), last_state
  return loss_fn


@functools.partial(jax.jit, static_argnames=('grad_fn', 'optimizer'))
def update_parameters_whole_sequence(
    params: hk.Params,
    rng: chex.PRNGKey,
    batch: chex.Array,
    grad_fn: Callable[[hk.Params, chex.PRNGKey, chex.Array, chex.Array],
                      tuple[tuple[chex.Array, chex.Array], chex.ArrayTree]],
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    init_state: Any,
) -> tuple[dict[str, Any], hk.Params, optax.OptState]:
  """Returns updated params and extra logs (like loss, last state etc).

  Backpropagation is done on the whole sequence. The whole function is jitted.

  Args:
    params: The current parameters of the network.
    rng: The random seed, used for dropout for instance.
    batch: The data batch.
    grad_fn: A gradient function, which takes some parameters, a random seed,
      the data to compute the gradient on, and an initial state for the model.
      It returns the gradient of the parameters for this batch of data, and
      extra values.
    optimizer: An optax optimizer.
    opt_state: The optimizer state.
    init_state: The initial state of the network (for an RNN for instance). Can
      be None.
  """
  (loss, last_state), grad = grad_fn(params, rng, batch, init_state)
  updates, new_opt_state = optimizer.update(grad, opt_state)
  new_params = optax.apply_updates(params, updates)

  log_dict = {
      'loss': loss,
      'last_state': last_state,
      'grad_norm_unclipped': optax.global_norm(grad),
  }

  return log_dict, new_params, new_opt_state


@functools.partial(jax.jit, static_argnames=('optimizer'))
def _compute_updates_from_chunks(
    params: hk.Params,
    losses: list[float],
    grads: list[chex.ArrayTree],
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
):
  """Returns updates from the list of gradients of the chunks."""
  # Compute the mean of losses across chunks.
  loss = jnp.mean(jnp.array(losses))
  # Compute the mean of gradients across chunks.
  avg_grads_fn = lambda *g: functools.reduce(jax.lax.add, g) / len(g)
  grad = jax.tree_util.tree_map(avg_grads_fn, *grads)

  # Classical update of parameters with the mean of gradients.
  updates, new_opt_state = optimizer.update(grad, opt_state)
  new_params = optax.apply_updates(params, updates)
  return loss, grad, new_params, new_opt_state


def _make_chunks_loss_fn(
    predictor: predictors.Predictor,
) -> Callable[
    [hk.Params, chex.PRNGKey, chex.Array, chex.Array, Any, bool],
    tuple[float, Any],
]:
  """Returns the loss function for update_parameters_in_chunks."""

  def loss_fn(params, rng, inputs, targets, init_state, last_chunk: bool):
    """Returns the loss for the model and the last state.

    Args:
      params: The parameters of the model, usually a neural network.
      rng: The random seed used to unroll the model (dropout for instance).
      inputs: The input array, of shape (B, T, F). B batch dimension, T time
        dimension, F feature dimension.
      targets: The targets array, also of shape (B, T, F).
      init_state: The initial state of the model. Can be anything, but usually
        will be an ArrayTree (like LSTM state).
      last_chunk: Whether the loss is computed for the last chunk or not.
    """
    output, states = predictor.unroll(params, rng, inputs, init_state)
    last_state = _get_last_state(states)
    if last_chunk:
      output = output[:, :-1]
    losses = optax.softmax_cross_entropy(output, targets)
    return jnp.mean(losses), last_state

  return loss_fn


def update_parameters_in_chunks(
    params: hk.Params, rng: chex.PRNGKey, batch: chex.Array, grad_fn: Callable[
        [hk.Params, chex.PRNGKey, chex.Array, chex.Array, chex.Array, bool],
        tuple[tuple[chex.Array, chex.Array],
              chex.ArrayTree]], optimizer: optax.GradientTransformation,
    opt_state: optax.OptState, init_state: Any,
    chunk_length: int) -> tuple[dict[str, Any], hk.Params, optax.OptState]:
  """Returns updated params and extra logs (like loss, last state etc).

  Backpropagation is done on chunks of the sequence, then averaged. The whole
  function itself is not jitted, due to memory issues with long sequences.
  Only the gradient computation of the chunks and the averaging is jitted.

  Args:
    params: The current parameters of the network.
    rng: The random seed, used for dropout for instance.
    batch: The data batch.
    grad_fn: A gradient function, which takes some parameters, a random seed,
      the data to compute the gradient on, and an initial state for the model.
      It returns the gradient of the parameters for this batch of data, and
      extra values.
    optimizer: An optax optimizer.
    opt_state: The optimizer state.
    init_state: The initial state of the network (for an RNN for instance). Can
      be None.
    chunk_length: Size of the chunks to consider. If lower than 1 or larger than
      seq_length, the passed value is clipped to this range.
  """
  seq_length = batch.shape[1]

  rngs = hk.PRNGSequence(rng)
  losses, grads = [], []
  init_state = copy.deepcopy(init_state)
  n_chunks = math.ceil(seq_length / chunk_length)
  for i in range(n_chunks):
    inputs = batch[:, i * chunk_length:(i + 1) * chunk_length]
    targets = batch[:, i * chunk_length + 1:(i + 1) * chunk_length + 1]
    last_chunk = (i == n_chunks - 1)
    (loss, last_state), grad = grad_fn(params, next(rngs), inputs, targets,
                                       init_state, last_chunk)
    # Update the initial state for the next batch with the last state.
    init_state = last_state
    losses.append(loss)
    grads.append(grad)

  # Compute updates. This part is jitted.
  loss, grad, new_params, new_opt_state = _compute_updates_from_chunks(
      params, losses, grads, optimizer, opt_state)

  log_dict = {
      'loss': loss,
      'last_state': last_state,
      'grad_norm_unclipped': optax.global_norm(grad),
  }

  return log_dict, new_params, new_opt_state


def _get_last_state(states: chex.ArrayTree) -> Optional[chex.ArrayTree]:
  """Returns the last state from an array tree of states."""
  if states is not None:
    return tree.map_structure(lambda x: x[:, -1], states)
  return None


def train(config: config_lib.ExperimentConfig,
          build_data_generator: base_constants.DataGeneratorBuilder,
          build_evaluator: base_constants.EvaluatorBuilder,
          use_tqdm: bool = False) -> None:
  """Trains a model.

  Nothing is returned.

  We choose to pass the data generator and evaluator to the train function
  directly rather than via the config, as that would mean adding new fields
  to the main constants file for each folder under experiments/. With this
  design, one can reuse the training loop in train.py in any other folder,
  without needing to change the constants file in this folder.

  Args:
    config: An experiment config, containing the hyperparameters.
    build_data_generator: A function to build a data generator.
    build_evaluator: A function to build an evaluator.
    use_tqdm: Whether to use a progress bar during training.
  """
  random.seed(config.seed)
  np.random.seed(config.seed)
  rng_seq = hk.PRNGSequence(config.seed)

  data_generator = build_data_generator(config.data)
  if config.train.seq_length_fixed:
    sample_batch = functools.partial(
        data_generator.sample,
        batch_size=config.train.batch_size,
        seq_length=config.train.seq_length,
    )
  else:
    sample_batch = functools.partial(
        data_generator.sample, batch_size=config.train.batch_size
    )
  frames_per_batch = config.train.batch_size * config.train.seq_length

  if config.train.seq_length_fixed:
    dummy_input, _ = sample_batch(rng=jrandom.PRNGKey(0))
  else:
    dummy_input, _ = sample_batch(
        rng=jrandom.PRNGKey(0), seq_length=config.train.seq_length
    )
  # No need to use the full batch size for params/config initialization.
  # Spares time and memory to only use batch_size = 1.
  dummy_input = dummy_input[:1]

  predictor = predictor_factories.PREDICTOR_FACTORIES[
      config.model.model_type.lower()
  ](
      dummy_input.shape[-1],
      config.model.architecture_kwargs,
  )

  evaluator = build_evaluator(predictor, config.eval)

  if config.train.gradient_chunk_length is None:
    loss_fn = _make_whole_loss_fn(predictor)
    update_parameters = update_parameters_whole_sequence
  else:
    loss_fn = _make_chunks_loss_fn(predictor)
    chunk_length = np.clip(config.train.gradient_chunk_length, 1,
                           config.train.seq_length)
    update_parameters = functools.partial(
        update_parameters_in_chunks, chunk_length=chunk_length)

  # Optimizer setup.
  optimizer = optax.adam(config.train.learning_rate)
  max_grad_norm = config.train.clip_grad_norm
  if max_grad_norm > 0:
    # First clip, *then* pass to optimizer.
    optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm), optimizer)

  # Update parameters setup.
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  if config.train.gradient_chunk_length is not None:
    # We jit the grad function here because the whole update_parameters function
    # itself is not jitted.
    grad_fn = jax.jit(grad_fn, static_argnames=('last_chunk',))

  model_init_rng = jrandom.PRNGKey(config.train.model_init_seed)
  dummy_hidden_state = predictor.initial_state(
      None, model_init_rng, config.train.batch_size
  )
  params = predictor.init_params(
      model_init_rng, dummy_input, dummy_hidden_state
  )
  opt_state = optimizer.init(params)

  predictor_init_state = predictor.initial_state(  # pytype: disable=wrong-arg-types
      params, None, config.train.batch_size
  )
  predictor_eval_init_state = predictor.initial_state(  # pytype: disable=wrong-arg-types
      params, None, config.eval.batch_size
  )

  range_fn = tqdm.trange if use_tqdm else range

  for step in range_fn(config.schedule.training_steps + 1):
    if config.train.seq_length_fixed:
      data_batch, _ = sample_batch(rng=next(rng_seq))
    else:
      log_length = random.randint(
          0, math.floor(math.log2(config.train.seq_length)))
      data_batch, _ = sample_batch(
          rng=next(rng_seq), seq_length=2**log_length)

    train_log_dict, params, opt_state = update_parameters(
        params=params,
        rng=next(rng_seq),
        batch=data_batch,
        grad_fn=grad_fn,
        optimizer=optimizer,
        opt_state=opt_state,
        init_state=predictor_init_state,
    )
    if not config.train.reset_predictor_init_state:
      predictor_init_state = train_log_dict['last_state']

    if (
        0 < config.logger.log_frequency
        and step % config.logger.log_frequency == 0
    ):
      eval_log_dict = evaluator.step(
          predictor_params=params,
          predictor_state=predictor_eval_init_state,
          rng=next(rng_seq),
      )

      train_log_dict = jax.device_get(train_log_dict)
      eval_log_dict = jax.device_get(eval_log_dict)
      log_dict = {**train_log_dict, **eval_log_dict}  # Merge the dicts.
      del log_dict['last_state']  # We don't want to log the state.
      del log_dict['logits']  # We don't want to log the logits.
      log_dict['num_frames'] = step * frames_per_batch
      log_dict['step'] = step
      print(log_dict)
