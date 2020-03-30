from argparse import ArgumentParser
from pathlib import Path

from .train import train

parser = ArgumentParser(description='Train DagNet')
parser.add_argument('--val_season', type=int, default=2018,
                    help='The tourney from this season will be used as '
                         'validation')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout prob for hidden layers')
parser.add_argument('--first_season', type=int, default=2003,
                    help='Last season to include')
parser.add_argument('--last_season', type=int, default=2018,
                    help='Last season to include')
parser.add_argument('--in_dir', type=Path, default='./data/graphs/mens',
                    help='Directory with graphs')
parser.add_argument('--out_dir', type=Path, default='.',
                    help='Directory with graphs')
args = parser.parse_args()

train(graph_dir=args.in_dir,
      out_dir=args.out_dir,
      val_season=args.val_season,
      first_season=args.first_season,
      last_season=args.last_season,
      lr=args.lr,
      epochs=args.epochs,
      dropout=args.dropout)
