import os
import pickle
import pandas as pd
import scipy
from tqdm.auto import tqdm
import numpy as np
import networkx as nx

from src.data.dataset import DataincDataset
from src.helpers.path_helper import *


ID_ATTRIBUTES = {'isin': 'ISIN', 'cusip': 'CUSIP', 'valor': 'VALOR', 'sedol': 'SEDOL'}
ID_ATTRIBUTES_REAL = {'isin': 'active_isin', 'cusip': 'active_cusip', 'valor': 'active_valor', 'sedol': 'active_sedol'}


def load_syn_master_mapping(dataset_name):

    if 'synthetic_companies' in dataset_name:
        master_df = pd.read_csv('data/raw/synthetic_data/seed_0/companies_master_mapping_seed_0.csv')

    elif 'synthetic_securities' in dataset_name:
        master_df = pd.read_csv('data/raw/synthetic_data/seed_0/securities_master_mapping_seed_0.csv')

    return master_df


def add_id_to_records(records_df, raw_df,) -> pd.DataFrame:
    # Assign to each (test) record its corresponding id value from the raw data

    records_df = records_df.astype({'external_id': 'str', 'data_source_id': 'str'})
    raw_df = raw_df.astype({'external_id': 'str', 'data_source_id': 'str'})
    return records_df.merge(raw_df[['id', 'external_id', 'data_source_id']], on=['external_id', 'data_source_id'], how='left')

def generate_matches_graph(pairwise_matches_preds: pd.DataFrame, threshold: float = 0.999) -> nx.Graph:

    positive_matches_df = pairwise_matches_preds[pairwise_matches_preds['prob'] > threshold]

    matches_graph = nx.from_pandas_edgelist(positive_matches_df, 'lid', 'rid', ['prob', 'match_type'])

    return matches_graph

def generate_transitive_matches_graph(matches_graph, add_transitive_edges = True, results_path = None, subgraph_size_threshold = 100):

    subgraphs = list(nx.connected_components(matches_graph))

    transitive_matches = []

    big_subgraph_sizes = [len(c) for c in subgraphs if len(c) > subgraph_size_threshold]

    if len(big_subgraph_sizes) > 0:
        # We save the size of big subgraphs
        big_subgraph_sizes_df = pd.DataFrame(big_subgraph_sizes, columns=['size'])
        big_subgraph_sizes_df.to_csv(os.path.join(results_path, 'big_subgraph_sizes.csv'), index=False)

    for subgraph_idx, c in enumerate(subgraphs):
        subgraph_nodes = list(c)

        if len(subgraph_nodes) > 1000:
            # If the subgraph is too big, we dont add its transitive edges to the matches_graph due to memory constraints, instead we will count
            # the transitive edges later on get_scores_matching.py
            continue

        # Add edges between all nodes of the subgraph to the matches_graph
        for lid in subgraph_nodes:
            for rid in subgraph_nodes:
                if lid != rid:
                    if not matches_graph.has_edge(lid, rid) and not matches_graph.has_edge(rid, lid) and add_transitive_edges:
                        matches_graph.add_edge(lid, rid, prob=0, match_type='transitive_match')
                    else:
                        transitive_matches.append((lid, rid, 'transitive_match'))

    return matches_graph, transitive_matches


