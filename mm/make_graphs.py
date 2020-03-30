from argparse import ArgumentParser
from pathlib import Path
from typing import Dict


import pandas as pd


LOC_MAP = {
    'A': -1,
    'N': 0,
    'H': 1
}


def load_detailed_results(datadir: Path) -> Dict[int, pd.DataFrame]:
    reg = pd.read_csv(datadir / 'RegularSeasonDetailedResults.csv')
    tour = pd.read_csv(datadir / 'NCAATourneyDetailedResults.csv')

    # Add flag indicating whether game was played in a tourney.
    reg.insert(1, 'Tourney', 0)
    tour.insert(1, 'Tourney', 1)

    # Replace Away, Neutral and Home court with numbers.
    reg.WLoc.replace(to_replace=LOC_MAP, inplace=True)
    tour.WLoc.replace(to_replace=LOC_MAP, inplace=True)

    # Create a data frame per season of the regular and tourney games.
    res = {}
    for s in reg.Season.unique():
        res[s] = pd.concat([reg[reg.Season == s], tour[tour.Season == s]],
                           ignore_index=True)
    return res


def make_edges(season_res: pd.DataFrame) -> pd.DataFrame:
    # Map team id to last game
    last_game = {}
    edges = []

    for i, row in season_res.iterrows():
        wt = row['WTeamID']
        lt = row['LTeamID']
        if wt in last_game:
            edges.append(dict(PrevGame=last_game[wt],
                              NextGame=i,
                              TeamID=wt))
        if lt in last_game:
            edges.append(dict(PrevGame=last_game[lt],
                              NextGame=i,
                              TeamID=lt))

        last_game[wt] = i
        last_game[lt] = i

    return pd.DataFrame(edges, columns=['PrevGame', 'NextGame', 'TeamID'])


if __name__ == '__main__':
    parser = ArgumentParser(description='Generate graphs from games for each '
                                        'season and save them to a directory')
    parser.add_argument('out_dir', type=Path, help='Result location. Will be '
                                               'created if it doesn\'t exist')
    parser.add_argument('--in_dir', type=Path, default='./data/kaggle',
                        help='Kaggle data directory location')
    args = parser.parse_args()

    args.out_dir.mkdir(exist_ok=True)

    res = load_detailed_results(args.in_dir)
    for season, games in res.items():
        edges = make_edges(res[season])
        edges.to_csv(args.out_dir / '{}_edges.csv'.format(season))
        games.to_csv(args.out_dir / '{}_games.csv'.format(season))







