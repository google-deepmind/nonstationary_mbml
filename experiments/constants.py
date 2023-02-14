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

from nonstationary_mbml.experiments import config as config_lib
from nonstationary_mbml.experiments import distributions
from nonstationary_mbml.experiments import live_and_die_agents
from nonstationary_mbml.experiments import ptw_agents
from nonstationary_mbml.experiments import trajectory_generators

DISTRIBUTIONS = {
    'beta': distributions.BetaDistribution,
    'gamma': distributions.GammaDistribution,
    'exponential': distributions.ExponentialDistribution,
    'constant': distributions.ConstantDistribution,
    'dirichlet': distributions.DirichletDistribution,
    'categorical': distributions.CategoricalDistribution,
    'uniform': distributions.UniformDistribution,
}

TRAJECTORY_GENERATORS = {
    'static':
        trajectory_generators.StaticTrajectoryGenerator,
    'regular_shift':
        trajectory_generators.RegularShiftTrajectoryGenerator,
    'random_shift':
        trajectory_generators.RandomShiftNoMemoryTrajectoryGenerator,
    'ptw':
        trajectory_generators.PTWRandomShiftTrajectoryGenerator,
    'iid_ptw_cat':
        trajectory_generators.IIDPTWRandomShiftCategoricalTrajectoryGenerator,
    'lin':
        trajectory_generators.LINTrajectoryGenerator,
}

OPTIMAL_AGENTS = {
    'ptw': ptw_agents.PTWAgent,
    'lin': live_and_die_agents.LADAgent,
}


# The following function follows the protocol constants.DataGeneratorBuilder.
def build_data_generator(
    config: config_lib.DataConfig) -> trajectory_generators.TrajectoryGenerator:
  """Returns a data generator from a meta_learning data config."""
  iid_distribution = DISTRIBUTIONS[config.iid_distribution](
      **config.iid_distribution_kwargs)
  return TRAJECTORY_GENERATORS[config.trajectory_generator](
      gen_distribution=iid_distribution,
      parameter_distribution=DISTRIBUTIONS[config.parameter_distribution](
          **config.parameter_distribution_kwargs),
      parameter_distribution_params=config.parameter_distribution_params,
      **config.trajectory_generator_kwargs)
