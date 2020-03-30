from collections import defaultdict
import datetime
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim

from .dag_net import DagNet
from .data import load_data


def load_model(params_path: Path, cp_path: Path):
    params = pd.read_csv(params_path)
    checkpoint = torch.load(cp_path)
    dagnet = DagNet(n_features=params['n_features'],
                    hidden_size=params['hidden_size'],
                    batch_size=params['batch_size'],
                    dropout=params['dropout'])
    dagnet.load_state_dict(checkpoint['model_state'])
    optimizer = optim.Adam(dagnet.parameters(),
                           lr=params['lr'],
                           weight_decay=params['l2_reg'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])

    return dagnet, optimizer, checkpoint, params


def save_model(out_dir: Path, train_start, model, optimizer,
               val_regular_hiddens, c_loss, h_loss, val_loss, epoch):
    path = out_dir / ('{}_checkpoint_{}.cp'.format(train_start, epoch))
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'val_regular_hiddens': dict(val_regular_hiddens),
        'c_loss': c_loss,
        'h_loss': h_loss,
        'val_loss': val_loss,
        'epoch': epoch
    }, path)


def get_timestamp():
    t = datetime.datetime.utcnow().isoformat()
    return re.sub('\.[0-9]+$', '', t.replace(':', '-'))


def validate(val_data, hiddens, dagnet):
    """Run validation on tourney games for a season.

    Hidden states computed during the regular season.
    """
    dagnet.eval()
    # Mask out tourney game box scores
    masked_features = val_data.features * val_data.validation_mask
    preds_h, preds_c, hiddens = dagnet(val_data.nodes, masked_features,
                                       hiddens)

    h_loss = F.nll_loss(preds_h, val_data.targets, reduction='none').detach().numpy()
    c_loss = F.nll_loss(preds_c, val_data.targets, reduction='none').detach().numpy()

    losses = []
    for i, node in enumerate(val_data.nodes):
        assert node.is_tourney
        # If first round, we have box scores from prev games
        if node.is_first_round:
            losses.append(c_loss[0, i])
        # If later round, we can only rely on the hidden states
        else:
            losses.append(h_loss[0, i])

    return np.mean(losses)


def train(graph_dir: Path, out_dir: Path, val_season: int,
          first_season: int, last_season: int, lr: float, epochs: int,
          dropout: float):
    batch_size = 1
    l2_reg = 0.001
    hidden_size = 32
    train_start = get_timestamp()
    args = locals()
    assert first_season <= val_season <= last_season

    seasons = {}
    val_data = None
    for s in range(first_season, last_season + 1):
        if s == val_season:
            val_data = load_data(graph_dir, s, batch_size, regular=False)
            seasons[s] = load_data(graph_dir, s, batch_size, tourney=False)
        else:
            seasons[s] = load_data(graph_dir, s, batch_size)

    n_features = seasons[first_season].features.size()[-1]

    args['n_features'] = n_features
    saved_args = pd.Series(args)
    print(saved_args)
    saved_args.to_csv(out_dir / '{}_params.csv'.format(train_start),
                      header=False)

    dagnet = DagNet(n_features=n_features, hidden_size=hidden_size,
                    batch_size=batch_size, dropout=dropout)
    optimizer = optim.Adam(dagnet.parameters(), lr=lr,
                           weight_decay=l2_reg)

    res_df = pd.DataFrame(columns=['Epoch', 'Train_h_loss', 'Train_c_loss',
                                   'Val_loss', 'Time'])
    res_df.set_index('Epoch')
    print(res_df.columns.values)
    np.set_printoptions(precision=6, suppress=True)

    rand = np.random.RandomState(0)
    val_regular_hiddens = None
    c_loss_m = None
    h_loss_m = None
    val_loss = None
    season_keys = list(seasons.keys())
    for i in range(epochs):
        dagnet.train()
        t0 = time.perf_counter()
        h_losses = []
        c_losses = []
        rand.shuffle(season_keys)
        for season in season_keys:
            s_data = seasons[season]
            optimizer.zero_grad()

            hiddens = defaultdict(lambda: dagnet.initHidden())
            preds_h, preds_c, hiddens = dagnet(s_data.nodes, s_data.features,
                                               hiddens)
            # Save hidden states from the regular season for validation on the
            # tourney.
            if season == val_season:
                val_regular_hiddens = hiddens

            unmasked_h_loss = F.nll_loss(preds_h, s_data.targets,
                                         reduction='none')
            masked_h_loss = unmasked_h_loss * s_data.h_targets_mask
            h_loss = masked_h_loss.sum() / s_data.h_targets_mask.sum()
            c_loss = F.nll_loss(preds_c, s_data.targets)
            loss = h_loss + c_loss
            loss.backward()
            optimizer.step()

            h_losses.append(h_loss.item())
            c_losses.append(c_loss.item())

        val_loss = validate(val_data, val_regular_hiddens, dagnet)
        td = time.perf_counter() - t0
        c_loss_m = np.mean(c_losses)
        h_loss_m = np.mean(h_losses)
        row = [i, c_loss_m, h_loss_m, val_loss, round(td)]
        res_df.loc[i] = row
        print(res_df.loc[i].values)

    save_model(out_dir=out_dir, train_start=train_start, model=dagnet,
               optimizer=optimizer, val_regular_hiddens=val_regular_hiddens,
               c_loss=c_loss_m, h_loss=h_loss_m, val_loss=val_loss,
               epoch=max(0, epochs - 1))

    res_df.to_csv(out_dir / ('{}_results.csv'.format(train_start)))

