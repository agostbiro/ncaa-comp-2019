from argparse import ArgumentParser
from pathlib import Path

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
import pandas as pd


def digraph_connected_components(G):
    seen_nodes = set()
    for node in nx.topological_sort(G):
        if node not in seen_nodes:
            sub_g = nx.DiGraph()
            sub_g.add_node(node)
            for source, target in nx.dfs_edges(G, node):
                if target not in seen_nodes:
                    sub_g.add_edge(source, target)
                    seen_nodes.add(source)
                    seen_nodes.add(target)
            yield sub_g


if __name__ == '__main__':
    parser = ArgumentParser(description='Generate graphs from games for each '
                                        'season and save them to a directory')
    parser.add_argument('season', type=int, help='Season to display')
    parser.add_argument('--max_nodes', type=int, default=500,
                        help='Maximum number of nodes to display')
    parser.add_argument('--in_dir', type=Path, default='./data/graphs',
                        help='Graph data directory')
    parser.add_argument('--out_dir', type=Path, default='.',
                        help='Directory to save the results to')
    args = parser.parse_args()

    edges = pd.read_csv(args.in_dir / '{}_edges.csv'.format(args.season))

    G = nx.DiGraph()

    for i, row in edges.iterrows():
        G.add_edge(row['PrevGame'], row['NextGame'])

    conn_comps = list(digraph_connected_components(G))
    n_nodes = sum(nx.number_of_nodes(g) for g in conn_comps)
    assert n_nodes == nx.number_of_nodes(G), (n_nodes, nx.number_of_nodes(G))

    for i, g in enumerate(conn_comps):
        A = to_agraph(g)
        A.layout('dot')
        fn = '{}_graph_{}.svg'.format(args.season, i)
        A.draw(str(args.out_dir / fn))


