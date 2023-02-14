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

"""Base config for experiments, and base sweep object.

Please don't touch this file! Change the parameters in your own sweep.
"""

import dataclasses
from typing import Any, Iterator, Mapping, Optional


@dataclasses.dataclass
class TrainConfig:
  model_init_seed: int = 1
  learning_rate: float = 1e-4
  seq_length: int = 20
  seq_length_fixed: bool = True
  batch_size: int = 128
  clip_grad_norm: float = 1.
  reset_predictor_init_state: bool = True
  gradient_chunk_length: Optional[int] = None


@dataclasses.dataclass
class ModelConfig:
  model_type: str = 'rnn'
  architecture_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ScheduleConfig:
  training_steps: int = 100000  # Number of gradient steps.
  num_saves: int = 30  # Total number of parameters to save.
  first_save: int = 0  # Num steps before first saving.
  saving_schedule_type: str = 'log'  # Either "linear" or "log".
  ckpt_frequency: int = 2000  # Frequency of checkpointing, in gradient steps.


@dataclasses.dataclass
class LoggerConfig:
  log_frequency: int = 250  # Frequency of logging, in gradient steps.
  log_remotely: bool = True  # Whether to add data to Bigtable.


@dataclasses.dataclass
class DataConfig:
  """Config for the data distribution.

  This class may be inherited and enhanced by experiments in the 'experiments'
  folder.
  """


@dataclasses.dataclass
class EvalConfig:
  """Config for the evaluator.

  This class may be inherited and enhanced by experiments in the 'experiments'
  folder.
  """
  batch_size: int = 128  # Batch size used for evaluation.


@dataclasses.dataclass
class ExperimentConfig:
  """Config for supervised learning experiments."""
  name: str = 'Supervised online learning'
  seed: int = 1
  data: DataConfig = dataclasses.field(default_factory=DataConfig)
  eval: EvalConfig = dataclasses.field(default_factory=EvalConfig)
  model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
  train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
  schedule: ScheduleConfig = dataclasses.field(default_factory=ScheduleConfig)
  logger: LoggerConfig = dataclasses.field(default_factory=LoggerConfig)


@dataclasses.dataclass
class ExperimentSweep:
  """A sweep to be passed to the experiment launcher, with parameter sweeps."""
  general_sweep: Iterator[dict[str, Any]]
  specific_sweeps: Mapping[str, Mapping[str, Iterator[Mapping[str, Any]]]]
  base_config: ExperimentConfig = ExperimentConfig()
