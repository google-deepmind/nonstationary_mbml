# Memory-Based Meta-Learning on Non-Stationary Distributions

<p align="center">
  <img src="https://raw.githubusercontent.com/deepmind/nonstationary_mbml/master/overview.svg" alt="Overview figure"/>
</p>

This repository provides an implementation of the paper [Memory-Based Meta-Learning on Non-Stationary Distributions](https://arxiv.org/abs/2302.03067).

> Memory-based meta-learning is a technique for approximating Bayes-optimal predictors.
> Under fairly general conditions, minimizing sequential prediction error, measured by the log loss, leads to implicit meta-learning.
> The goal of this work is to investigate how far this interpretation can be realized by current sequence prediction models and training regimes.
> The focus is on piecewise stationary sources with unobserved switching-points, which arguably capture an important characteristic of natural language and action-observation sequences in partially observable environments.
> We show that various types of memory-based neural models, including Transformers, LSTMs, and RNNs, can learn to accurately approximate known Bayes-optimal algorithms and behave as if performing Bayesian inference over the latent switching-points and the latent parameters governing the data distribution within each segment.

It is based on [JAX](https://jax.readthedocs.io) and [Haiku](https://dm-haiku.readthedocs.io) and contains all code, datasets, and models necessary to reproduce the paper's results.


## Content

```
.
├── experiments
|   ├── config.py                   - Experiment configurations
|   ├── constants.py                - Experiment constants
|   ├── distributions.py            - Probability distributions
|   ├── evaluator.py                - Evaluation loop
|   └── live_and_die_predictors.py  - LAD (Willems, 1996)
|   └── local_launch.py             - Local launch script
|   └── ptw_predictors.py           - PTW (Veness et al., 2013)
|   └── trajectory_generators.py    - Trajectory generators
|
├── models
|   ├── basic.py                    - CNNs, MLPs, RNNs
|   └── positional_encodings.py     - ALiBi (Press et al., 2022), relative (Dai et al., 2019), sin/cos (Vaswani et al., 2017)
|   ├── stack_rnn.py                - Stack-RNN (Joulin & Mikolov, 2015)
|   └── transformer.py              - Transformer (Vaswani et al., 2017)
|
├── README.md
├── predictor_factories.py          - Factories to initialize predictors
├── predictors.py                   - Predictor interface
├── base_config.py                  - Base configurations
├── base_constants.py               - Base constants
├── requirements.txt                - Dependencies
└── train.py                        - Training loop
```


## Installation

```
pip install -r requirements.txt
```


## Usage

```
python experiments/local_launch.py
```

The experiment configurations can be adjusted in `base_config.py` and
`experiments/config.py`.


## Citing this work

```bibtex
@article{genewein2023memory,
  author    = {Tim Genewein and
               Gr{\'{e}}goire Del{\'{e}}tang and
               Anian Ruoss and
               Li Kevin Wenliang and
               Elliot Catt and
               Vincent Dutordoir and
               Jordi Grau-Moya and
               Laurent Orseau and
               Marcus Hutter and
               Joel Veness},
  title     = {Memory-Based Meta-Learning on Non-Stationary Distributions},
  journal   = {CoRR},
  volume    = {abs/2302.03067},
  year      = {2023},
}
```


## License and disclaimer

Copyright 2023 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
