"""Implements an SeqNet that can process directed acyclic graphs.

Some code adapted from: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html  # noqa 501

"""
from typing import Dict, Hashable, List, NamedTuple

import torch
from torch import nn
import torch.nn.functional as F


from mm.data import Node


class StepResult(NamedTuple):
    # Prediction from hidden states only
    pred_h: float
    # Prediction from input features + hidden state
    pred_c: float
    # Hidden states
    hidden1: torch.Tensor
    hidden2: torch.Tensor


class DagNet(nn.Module):
    """Binary classifier for nodes of regular DAGs with in/out degree 2."""
    def __init__(self, n_features, hidden_size, batch_size, dropout,
                 n_classes=2):

        super().__init__()

        # Inputs to hidden states
        self.i2h = nn.Linear(2 * n_features, 2 * hidden_size)
        self.h2h = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.dropout_h = nn.Dropout(p=dropout)
        # Hidden to output
        self.h2o = nn.Linear(2 * hidden_size, n_classes)

        self.batch_size = batch_size
        self.n_classes = n_classes

        self.n_features = n_features
        self.hidden_size = hidden_size

    def step(self, input1, hidden1, input2, hidden2) -> StepResult:
        """Apply the RNN cell at a node.

        Args:
            input1: (batch_size, n_features)
            hidden1: (batch_size, hidden_size)
            input2: (batch_size, n_features)
            hidden2: (batch_size, hidden_size)

        """
        combined_i = torch.cat((input1, input2), -1)
        combined_h = torch.cat((hidden1, hidden2), -1)

        hidden = self.h2h(self.dropout_h(combined_h))
        combined = F.leaky_relu(self.i2h(combined_i) + hidden)

        pred_h = F.log_softmax(self.h2o(F.leaky_relu(hidden)), dim=-1)
        pred_c = F.log_softmax(self.h2o(combined), dim=-1)

        hidden1 = combined[..., :self.hidden_size]
        hidden2 = combined[..., self.hidden_size:]

        return StepResult(pred_h=pred_h, pred_c=pred_c,
                          hidden1=hidden1, hidden2=hidden2)

    def forward(self, nodes: List[Node], features: torch.Tensor,
                hiddens: Dict[Hashable, torch.Tensor]):
        """

        Args:
            nodes: List of nodes in topologically sorted order describing input
                features and hidden states and output hidden states. Index 0 is
                reserved for the initial hidden state.
            features: (2 * n_nodes, batch_size, n_features) tensor
            hiddens: Dict holding hidden states tensors of shape (n_features,)
                at indices determined by the input nodes. Will be modified in
                place.

        Returns:
            preds_h: Predictions from hidden states only.
                (batch_size, n_classes, seq_len)
            preds_c: Predictions from hidden states + input features.
                (batch_size, n_classes, seq_len)
            hiddens: Updated hiddens.

        """
        preds_h = torch.zeros(self.batch_size, self.n_classes, len(nodes),
                              dtype=torch.float)
        preds_c = torch.zeros(self.batch_size, self.n_classes, len(nodes),
                              dtype=torch.float)
        for i, node in enumerate(nodes):
            step_res = self.step(features[node.features[0]],
                                 hiddens[node.h_in[0]],
                                 features[node.features[1]],
                                 hiddens[node.h_in[1]])
            preds_h[:, :, i] = step_res.pred_h
            preds_c[:, :, i] = step_res.pred_c
            hiddens[node.h_out[0]] = step_res.hidden1
            hiddens[node.h_out[1]] = step_res.hidden2
        return preds_h, preds_c, hiddens

    def initHidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)
