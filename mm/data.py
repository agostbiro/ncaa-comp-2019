from pathlib import Path
from typing import List, NamedTuple, Hashable

import networkx as nx
import numpy as np
import pandas as pd
import torch


class Node(NamedTuple):
    """Contains input and output indexes for each node in the DAG."""
    # Feature indices
    features: List[int]
    # Input hidden states keys
    h_in: List[Hashable]
    # Output hidden state keys
    h_out: List[Hashable]

    target: int
    h_target_mask: int

    # Whether the node is a tourney game and if so whether it's in the first
    # round. Used for validation.
    is_tourney: bool
    is_first_round: bool


class DagData(NamedTuple):
    """Data for training DagNet."""
    nodes: List[Node]
    # Input features
    features: torch.Tensor
    # Mask for input features on validation. Removes tourney games.
    validation_mask: torch.Tensor

    # Binary classification targets for each node
    targets: torch.Tensor
    # Hidden prediction loss mask for leaf nodes
    h_targets_mask: torch.Tensor


def process_game(games, graph, game_idx, reverse_order=False):
    nd = {
        'features': [],
        'h_in': [],
        'h_out': [],
        'h_target_mask': 1,
        'is_tourney': games.loc[game_idx].Tourney == 1,
        'is_first_round': False
    }

    in_edges = list(graph.in_edges(game_idx, data=True))
    if reverse_order:
        in_edges = reversed(in_edges)

    for i, edge in enumerate(in_edges):
        prev_game_idx, _, data = edge
        # Input feature from previous game_idx
        nd['features'].append(prev_game_idx)
        nd['h_in'].append(data['team_id'])
        nd['h_out'].append(data['team_id'])

        # If first team won, target is 0, if second team won target is 1.
        if games.loc[game_idx].WTeamID == data['team_id']:
            nd['target'] = i

        # No hidden states for leaf nodes
        if graph.in_degree(prev_game_idx) < 2:
            nd['h_target_mask'] = 0

        if nd['is_tourney'] and games.loc[prev_game_idx].Tourney == 0:
            nd['is_first_round'] = True

    return Node(**nd)


def load_graph(edges: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    for _, row in edges.iterrows():
        graph.add_edge(row['PrevGame'], row['NextGame'], team_id=row['TeamID'])
    return graph


def load_data(graph_dir: Path, season: int, batch_size: 1,
              regular=True, tourney=True) -> DagData:
    """Load training data for a season."""
    # Batching is not implemented
    assert batch_size == 1, batch_size

    # TODO allow requesting only regular or tourney data
    games = pd.read_csv(graph_dir / ('{}_games.csv'.format(season)))
    edges = pd.read_csv(graph_dir / ('{}_edges.csv'.format(season)))

    # TODO normalize?
    # Unnamed: 0 is the index which is redundant when converting to numpy
    features = games.drop(columns=['Unnamed: 0', 'Season']).values
    features = features.astype(np.float32)

    val_mask = np.ones(features.shape, dtype=np.float32)
    # Mask out tourney games
    val_mask[features[:, 0] == 1] = 0

    graph = load_graph(edges)

    # TODO make sure to process each node with input A and B as AB and BA.
    targets = []
    h_targets_mask = []
    nodes = []
    in_degrees = {}
    rand = np.random.RandomState(season)
    for game in nx.topological_sort(graph):
        if not regular and games.loc[game].Tourney == 0:
            continue
        if not tourney and games.loc[game].Tourney == 1:
            continue

        in_deg = graph.in_degree(game)
        in_degrees[game] = in_deg
        # Max in degree of 2 is assumed
        assert in_deg < 3, game
        # Since we rely on data from previous games, we can not make
        # predictions for games where a team hasn't played before.
        if in_deg < 2:
            continue

        reverse_order = bool(rand.randint(0, 2))
        node = process_game(games, graph, game, reverse_order=reverse_order)
        nodes.append(node)
        targets.append(node.target)
        h_targets_mask.append(node.h_target_mask)

        #node = process_game(games, graph, game, reverse_order=True)
        #nodes.append(node)
        #targets.append(node.target)
        #h_targets_mask.append(node.h_target_mask)

    print(np.mean(targets))
    # Add batch size 1
    targets_t = torch.from_numpy(np.array(targets)[np.newaxis, :])
    h_targets_mask_t = torch.from_numpy(
        np.array(h_targets_mask, dtype=np.float32)[np.newaxis, :])
    features_t = torch.from_numpy(features[:, np.newaxis, :])
    val_mask_t = torch.from_numpy(val_mask[:, np.newaxis, :])

    return DagData(nodes=nodes, targets=targets_t, features=features_t,
                   validation_mask=val_mask_t, h_targets_mask=h_targets_mask_t)
