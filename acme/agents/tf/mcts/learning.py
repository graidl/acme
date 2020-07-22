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

"""A MCTS "AlphaZero-style" learner."""

from typing import List

import acme
import numpy as np
import sonnet as snt
import tensorflow as tf
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
from acme.tf import savers as tf2_savers


class AZLearner(acme.Learner):
    """AlphaZero-style learning."""

    def __init__(
            self,
            network: snt.Module,
            optimizer: snt.Optimizer,
            dataset: tf.data.Dataset,
            discount: float,
            logger: loggers.Logger = None,
            counter: counting.Counter = None,
            checkpoint: bool = False,
            snapshot: bool = False,
            directory: str = '~/acme/',
            td_learning: bool = True,
    ):
        # Logger and counter for tracking statistics / writing out to terminal.
        self._counter = counting.Counter(counter, 'learner')
        self._logger = logger or loggers.TerminalLogger('learner', time_delta=30.)

        # Internalize components.
        # TODO(b/155086959): Fix type stubs and remove.
        self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types
        self._optimizer = optimizer
        self._network = network
        self._variables = network.trainable_variables
        self._discount = np.float32(discount)
        self._td_learning = td_learning
        # Create a checkpointer and snapshotter objects.
        self._checkpointer = None
        self._snapshotter = None

        if checkpoint:
            self._checkpointer = tf2_savers.Checkpointer(
                objects_to_save={
                    'counter': self._counter,
                    'network': self._network,
                    'optimizer': self._optimizer,
                },
                directory=directory,
                add_uid=False,
                time_delta_minutes=5.0
            )
            self._snapshotter = tf2_savers.Snapshotter(
                objects_to_save={
                    'network': self._network,
                },
                directory=directory,
                time_delta_minutes=15.0
            )

    @tf.function
    def _step(self) -> tf.Tensor:
        """Do a step of SGD on the loss."""

        inputs = next(self._iterator)
        o_t, _, r_t, d_t, o_tp1, extras = inputs.data

        if isinstance(o_t, dict):
            o_t = o_t['obs']
            o_tp1 = o_tp1['obs']

        pi_t = extras['pi']

        with tf.GradientTape() as tape:
            # Forward the network on the two states in the transition.
            logits, value = self._network(o_t)
            if self._td_learning:
                _, target_value = self._network(o_tp1)
                target_value = tf.stop_gradient(target_value)
                # target_value = extras['Vhat']  # Moerland et al. target value
                # Value loss is simply on-policy TD learning.
                value_loss = tf.square(r_t + self._discount * d_t * target_value - value)
            else:
                # learn from total reward up to episode end
                value_loss = tf.square(r_t - value)

            # Policy loss distills MCTS policy into the policy network.
            policy_loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=pi_t)

            # Compute gradients.
            loss = tf.reduce_mean(value_loss + policy_loss)
            gradients = tape.gradient(loss, self._network.trainable_variables)

        self._optimizer.apply(gradients, self._network.trainable_variables)

        return loss

    def step(self):
        """Does a step of SGD and logs the results."""
        loss = self._step()
        if self._checkpointer is not None:
            self._checkpointer.save()
        if self._snapshotter is not None:
            self._snapshotter.save()
        self._logger.write({'loss': loss})

    def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
        """Exposes the variables for actors to update from."""
        return tf2_utils.to_numpy(self._variables)

    def save_checkpoint_and_snapshot(self):
        """If checkpointer/snapshotter used, do forced save."""
        if self._checkpointer is not None:
            self._checkpointer.save(force=True)
        if self._snapshotter is not None:
            self._snapshotter.save(force=True)
