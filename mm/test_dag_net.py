from unittest import TestCase

import numpy as np
import torch
from torch import optim
from torch import nn

from .dag_net import DagNet
from mm.data import Node


class DagTestBase:
    def setUp(self):
        raise NotImplementedError

    def _train(self):
        losses = []
        for i in range(self.train_features.shape[0]):
            self.optimizer.zero_grad()
            hiddens = {0: self.dagnet.initHidden()}
            preds, _ = self.dagnet(self.nodes, self.train_features[i], hiddens)
            loss = self.loss_fn(preds, self.train_targets[i])
            losses.append(loss.item())
            loss.backward()
            # Does the gradient update
            self.optimizer.step()
        return np.mean(losses)

    def _validate(self):
        losses = []
        for i in range(self.train_features.shape[0]):
            hiddens = {0: self.dagnet.initHidden()}
            preds, _ = self.dagnet(self.nodes, self.train_features[i], hiddens)
            loss = self.loss_fn(preds, self.train_targets[i])
            losses.append(loss.item())
        return np.mean(losses)

    def test(self):
        print('Training', self.__class__.__name__)
        pre_val_loss = self._validate()

        for i in range(self.epochs):
            loss = self._train()
            if i % 10 == 0 or i == self.epochs - 1:
                print(i, loss)

        val_loss = self._validate()
        val_dec = max(round((1 - val_loss / pre_val_loss) * 100, 2), 0)
        print('Validation loss declined by {}%'.format(val_dec))
        self.assertGreater(val_dec, 99)


class TestDagSimple(TestCase, DagTestBase):
    """Test if DagNet can solve a simple sequential classification problem."""

    @staticmethod
    def _compute_target(features, f_weights):
        """Compute a non-linear function of the features."""
        z = features * f_weights
        a = np.where(z > 0.5 * np.sum(z) / np.size(z), z, 0)
        return np.linalg.norm(a, 1)

    def setUp(self):
        self.epochs = 600
        self.n_batches = 30
        self.n_val_batch = 5
        self.batch_size = 32
        self.lr = 0.001
        self.l2_reg = 0
        self.n_features = 8
        self.hidden_size = 160

        # DAG represents a sequence of games where 4 teams all play each other.
        self.nodes = [
            Node(f1=0,  f2=1,  h_in1=0, h_in2=0, h_out1=1, h_out2=2),
            Node(f1=2,  f2=3,  h_in1=0, h_in2=0, h_out1=3, h_out2=4),
            Node(f1=4,  f2=5,  h_in1=1, h_in2=3, h_out1=1, h_out2=3),
            Node(f1=6,  f2=7,  h_in1=2, h_in2=4, h_out1=2, h_out2=4),
            Node(f1=8,  f2=9,  h_in1=1, h_in2=4, h_out1=1, h_out2=4),
            Node(f1=10, f2=11, h_in1=2, h_in2=3, h_out1=2, h_out2=3)
        ]
        self.n_nodes = len(self.nodes)

        # Create feature vectors drawn from a normal distribution
        rand = np.random.RandomState(100)
        f_shape = (self.n_batches,
                   2 * self.n_nodes,
                   self.batch_size,
                   self.n_features)
        features = rand.randint(-5, 6, size=f_shape) / 10
        f_weights = rand.normal(0, 1, size=(self.n_features,))
        # Classification target is whether the current the sum of the L2 norms
        # of the current vector and all the dependent vectors in the first
        # sequence are greater than those in the second sequence.
        targets = np.zeros((self.n_batches, self.batch_size, self.n_nodes),
                           dtype=np.long)
        for i in range(self.n_batches):
            for j, node in enumerate(self.nodes):
                sum1, sum2 = 0, 0
                for k in range(self.batch_size):
                    sum1 += self._compute_target(features[i, node.f1, k],
                                                 f_weights)
                    sum2 += self._compute_target(features[i, node.f2, k],
                                                 f_weights)
                    if sum1 > sum2:
                        # Second dim must be batch size for NLLLoss
                        targets[i, k, j] = 1
        ratio = np.mean(targets)
        self.assertTrue(0.4 < ratio < 0.6)

        # Train/validation split
        features = features.astype(np.float32)
        train_features = features[self.n_val_batch:]
        train_targets = targets[self.n_val_batch:]
        val_features = features[:self.n_val_batch]
        val_targets = targets[:self.n_val_batch]

        self.train_features = torch.from_numpy(train_features)
        self.train_targets = torch.from_numpy(train_targets)
        self.val_features = torch.from_numpy(val_features)
        self.val_targets = torch.from_numpy(val_targets)

        self.dagnet = DagNet(self.n_features, self.hidden_size,
                             self.batch_size)
        self.optimizer = optim.Adam(self.dagnet.parameters(), lr=self.lr,
                                    weight_decay=self.l2_reg)
        self.loss_fn = nn.NLLLoss()


class TestDag(TestCase, DagTestBase):
    """Test whether SeqNet can predict results in a simulated tourney."""
    def setUp(self):
        sd = 1
        nf = 0.3
        self.epochs = 100
        self.n_batches = 30
        self.n_val_batch = 5
        self.batch_size = 32
        self.lr = 0.001
        self.l2_reg = 0
        self.n_features = 8
        self.hidden_size = 160

        # DAG represents a sequence of games where 4 teams all play each other.
        self.nodes = [
            Node(f1=0,  f2=1,  h_in1=0, h_in2=0, h_out1=1, h_out2=2),
            Node(f1=2,  f2=3,  h_in1=0, h_in2=0, h_out1=3, h_out2=4),
            Node(f1=4,  f2=5,  h_in1=1, h_in2=3, h_out1=1, h_out2=3),
            Node(f1=6,  f2=7,  h_in1=2, h_in2=4, h_out1=2, h_out2=4),
            Node(f1=8,  f2=9,  h_in1=1, h_in2=4, h_out1=1, h_out2=4),
            Node(f1=10, f2=11, h_in1=2, h_in2=3, h_out1=2, h_out2=3)
        ]
        self.n_nodes = len(self.nodes)
        self.n_teams = 4
        self.assertEqual(self.n_nodes, self.n_teams * (self.n_teams - 1) / 2)

        rand = np.random.RandomState(200)
        f_shape = (self.n_batches,
                   2 * self.n_nodes,
                   self.batch_size,
                   self.n_features)
        features = np.zeros(shape=f_shape)
        for i in range(self.n_batches):
            for j in range(self.batch_size):
                # Graph-specific features
                g_features = rand.normal(0, sd,
                                         size=(self.n_teams, self.n_features))
                for k, node in enumerate(self.nodes):
                    loc1 = rand.uniform(-sd * nf, sd * nf)
                    noise1 = rand.normal(loc1, sd * nf,
                                         size=(self.n_features,))
                    loc2 = rand.uniform(-sd * nf, sd * nf)
                    noise2 = rand.normal(loc2, sd * nf,
                                         size=(self.n_features,))
                    # Hiddens are 1-indexed.
                    h1 = node.h_out1 - 1
                    h2 = node.h_out2 - 1
                    features[i, node.f1, j] = g_features[h1] + noise1
                    features[i, node.f2, j] = g_features[h2] + noise2

        targets = np.zeros((self.n_batches, self.batch_size, self.n_nodes))
        prev_changes = 0
        for i in range(self.n_batches):
            for j in range(self.batch_size):
                results = {t: {} for t in range(1, self.n_teams + 1)}
                for k, node in enumerate(self.nodes):
                    # Two team indices. Index 0 is zero vector
                    t1 = node.h_out1
                    t2 = node.h_out2
                    prev_diffs = set(results[t1].keys()) & set(results[t2].keys())
                    prev1 = [results[t1][tp] for tp in prev_diffs]
                    prev2 = [results[t2][tp] for tp in prev_diffs]
                    curr1 = np.linalg.norm(features[i, node.f1, j])
                    curr2 = np.linalg.norm(features[i, node.f2, j])
                    n1 = curr1 + sum(prev1)
                    n2 = curr2 + sum(prev2)
                    diff = n1 - n2
                    if diff * (curr1 - curr2) < 0:
                        prev_changes += 1
                    if diff >= 0:
                        targets[i, j, k] = 1
                    results[t1][t2] = n1 - n2
                    results[t2][t1] = n2 - n1
        print('prev changes', prev_changes)
        print('changes', 0.5 * prev_changes / (self.n_batches * self.batch_size))
        print('targets', np.mean(targets))

        # Train/validation split
        features = features.astype(np.float32)
        targets = targets.astype(np.long)
        train_features = features[self.n_val_batch:]
        train_targets = targets[self.n_val_batch:]
        val_features = features[:self.n_val_batch]
        val_targets = targets[:self.n_val_batch]

        self.train_features = torch.from_numpy(train_features)
        self.train_targets = torch.from_numpy(train_targets)
        self.val_features = torch.from_numpy(val_features)
        self.val_targets = torch.from_numpy(val_targets)

        self.dagnet = DagNet(n_features=self.n_features,
                             hidden_size=self.hidden_size,
                             batch_size=self.batch_size)
        self.optimizer = optim.Adam(self.dagnet.parameters(), lr=0.01,
                                    weight_decay=self.l2_reg)
        self.loss_fn = nn.NLLLoss()

