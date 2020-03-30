from pathlib import Path

import networkx as nx
import pandas as pd

from .train import load_model


def build_graph(data_dir: Path, graphs_dir: Path, val_season: int) \
        -> nx.DiGraph:
    slots = pd.read_csv(data_dir / 'NCAATourneySlots.csv')
    slots = slots[slots.Season == val_season]
    seeds = pd.read_csv(data_dir / 'NCAATourneySeeds.csv')
    seeds = seeds[seeds.Season == val_season]

    games = pd.read_csv(graphs_dir / ('{}_games.csv'.format(val_season)))

    contenders = {}


def eval(data_dir: Path, graphs_dir: Path, params_path: Path, cp_path: Path,
         out_dir: Path):
    dagnet, optimizer, checkpoint, params = load_model(params_path, cp_path)
    dagnet.eval()
    val_season = params['val_season']
