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

"""Script for launching locally."""

from absl import app

from nonstationary_mbml import base_config as config_lib
from nonstationary_mbml import train
from nonstationary_mbml.experiments import config as meta_learning_config_lib
from nonstationary_mbml.experiments import constants
from nonstationary_mbml.experiments import evaluator


def main(unused_argv) -> None:
  config = config_lib.ExperimentConfig()
  config.data = meta_learning_config_lib.DataConfig()
  config.data.iid_distribution = 'categorical'
  config.data.iid_distribution_kwargs['size'] = 2
  config.data.parameter_distribution = 'dirichlet'
  config.data.parameter_distribution_params = (0.5, 0.5)
  config.data.parameter_distribution_kwargs['size'] = 2

  config.eval = meta_learning_config_lib.EvalConfig()
  config.eval.seq_length = 20
  config.eval.data = config.data

  config.train.seq_length = 20

  config.model.model_type = 'sliding_window_transformer'
  config.model.architecture_kwargs = {
      'context_length': 10,
      'positional_encodings': 'ALIBI'
  }

  train.train(config, constants.build_data_generator, evaluator.build_evaluator)  # pytype: disable=wrong-arg-types


if __name__ == '__main__':
  app.run(main)
