# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A MCTS actor."""

from typing import Tuple
from copy import deepcopy

import acme
import dm_env
import numpy as np
import sonnet as snt
import tensorflow as tf
from acme import adders
from acme import specs
from acme.agents.tf.mcts import models
from acme.tf import variable_utils as tf2_variable_utils
from scipy import special

from acme.agents.tf.mcts import search
from acme.agents.tf.mcts import types


class MCTSActor(acme.Actor):
    """Executes a policy- and value-network guided MCTS search."""

    _prev_timestep: dm_env.TimeStep

    def __init__(
            self,
            environment_spec: specs.EnvironmentSpec,
            model: models.Model,
            network: snt.Module,
            discount: float,
            num_simulations: int,
            search_policy: str = 'puct',
            ucb_scaling: float = 1.0,
            adder: adders.Adder = None,
            variable_client: tf2_variable_utils.VariableClient = None,
    ):

        # Internalize components: model, network, data sink and variable source.
        self._model = model
        self._network = tf.function(network)
        self._variable_client = variable_client
        self._adder = adder

        # Internalize hyper-parameters.
        self._num_actions = environment_spec.actions.num_values
        self._num_simulations = num_simulations
        self._discount = discount
        if search_policy == 'puct':
            self._search_policy = (lambda node: search.puct(node, ucb_scaling))
        elif search_policy == 'bfs':
            self._search_policy = search.bfs
        else:
            ValueError(f'Invalid search_policy in MCTS: {search_policy}')
        # We need to save the policy so as to add it to replay on the next step.
        self._probs = np.ones(shape=(self._num_actions,), dtype=np.float32) / self._num_actions
        # We save the target value calculated according to Moerland et al. here
        self._Vhat = np.zeros(shape=(1,), dtype=np.float32)

    def _forward(self, observation: types.Observation) -> Tuple[types.Probs, types.Value]:
        """Performs a forward pass of the policy-value network."""
        if isinstance(observation, dict):
            action_mask = observation['action_mask']
            observation = observation['obs']
        else:
            action_mask = None
        logits, value = self._network(tf.expand_dims(observation, axis=0))

        # Convert to numpy & take softmax.
        logits = logits.numpy().squeeze(axis=0)
        if action_mask is not None:
            logits[~action_mask] = -np.inf
        value = value.numpy().item()
        probs = special.softmax(logits)
        return probs, value

    def select_action(self, observation: types.Observation) -> types.Action:
        """Computes the agent's policy via MCTS."""
        obs = observation['obs'] if isinstance(observation, dict) else observation
        if self._model.needs_reset:
            self._model.reset_to_original_env()

        if self._model.needs_reset:
            self._model.reset(observation)

        # Compute a fresh MCTS plan.
        root = search.mcts(
            observation,
            model=self._model,
            search_policy=self._search_policy,
            evaluation=self._forward,
            num_simulations=self._num_simulations,
            num_actions=self._num_actions,
            discount=self._discount,
        )

        # The agent's policy is softmax w.r.t. the *visit counts* as in AlphaZero.
        probs = search.visit_count_policy(root)
        actions = root.valid_actions()
        action = np.int32(np.random.choice(actions, p=probs))

        # Save the policy probs so that we can add them to replay in `observe()`.
        self._probs = np.zeros(self._num_actions, dtype=np.float32)
        self._probs[actions] = probs.astype(np.float32)

        self._Vhat = np.zeros(shape=(1,), dtype=np.float32)
        Vhat = np.inner(probs, root.children_values)  # Target value for learning according to Moerland et al.
        self._Vhat[0] = Vhat
        return action

    def update(self):
        """Fetches the latest variables from the variable source, if needed."""
        if self._variable_client:
            self._variable_client.update()

    def observe_first(self, timestep: dm_env.TimeStep):
        self._prev_timestep = timestep
        if self._adder:
            self._adder.add_first(timestep)

    def observe(self, action: types.Action, next_timestep: dm_env.TimeStep):
        """Updates the agent's internal model and adds the transition to replay."""
        self._model.update(self._prev_timestep, action, next_timestep)

        self._prev_timestep = next_timestep

        if self._adder:
            self._adder.add(action, next_timestep, extras={'pi': self._probs, 'Vhat': self._Vhat})
