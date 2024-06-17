import os
import sys
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp

from scipy.sparse import coo_matrix
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.helpers.path_helper import *




def get_scores_args():
    parser = argparse.ArgumentParser(description='Calculate the scores of a records matching experiment')

    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--experiment_names_list', action='append', required=True)
    parser.add_argument('--ground_truth_path', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=0.999, required=False)
    parser.add_argument('--non_positional_ids', action='store_true', required=False)
    args = parser.parse_args()

    return args


def get_scores(matching_folder_path, ground_truth_file_path, dataset_name, threshold, non_positional_ids = False):
    """
    Computes all the scores of a given matching, including pairwise scores and pre/post graph cleanup entity group matching scores.
    """
    scores_dict = {}

    # Load the ground truth

    ground_truth = pd.read_csv(ground_truth_file_path)
    # Filter out the negative pairs of the ground truth
    ground_truth = filter_ground_truth_pairs_df(ground_truth)

    # Load the pairwise predictions, the subgraph size list (if applicable), the pre-graph cleanup transitive matches and all the post-graph cleanup matches

    pairwise_matches_preds = pd.read_csv(os.path.join(matching_folder_path, 'pairwise_matches_preds.csv'), header=0)

    # Check for the big subgraphs size list
    if os.path.isfile(os.path.join(matching_folder_path, 'big_subgraph_sizes.csv')):
        big_subgraph_sizes_df= pd.read_csv(os.path.join(matching_folder_path, 'big_subgraph_sizes.csv'), header=0)
        big_subgraph_sizes = big_subgraph_sizes_df['size'].values.tolist()
    else:
        big_subgraph_sizes = []

    pre_cleanup_transitive_matches = pd.read_csv(os.path.join(matching_folder_path, 'pre_cleanup_transitive_matches.csv'), header=0)
    post_graph_cleanup_matches = pd.read_csv(os.path.join(matching_folder_path, 'post_graph_cleanup_matches.csv'), header=0)


    # Preprocess the pairwise predictions

    pairwise_matches_preds = filter_pairs_df(pairwise_matches_preds, threshold = threshold)

    # If we have non-positional ids, we need to trasnform them to positional ids (otherwise the sparse matrices will be too big)

    if non_positional_ids:
        ground_truth = transform_ids_to_positional_ids(ground_truth, dataset_name)
        pairwise_matches_preds = transform_ids_to_positional_ids(pairwise_matches_preds, dataset_name)
        pre_cleanup_transitive_matches = transform_ids_to_positional_ids(pre_cleanup_transitive_matches, dataset_name)
        post_graph_cleanup_matches = transform_ids_to_positional_ids(post_graph_cleanup_matches, dataset_name)

    # Find the max lid/rid 

    max_id = get_max_id(ground_truth, pairwise_matches_preds, pre_cleanup_transitive_matches, post_graph_cleanup_matches)

    # Construct the sparse coo-matrices of pairs and labels for the ground truth and the predictions
 
    ground_truth_sparse = construct_sparse_matrix(ground_truth, max_id)
    pairwise_preds = construct_sparse_matrix(pairwise_matches_preds, max_id)
    pre_cleanup_transitive_matches_sparse = construct_sparse_matrix(pre_cleanup_transitive_matches, max_id)
    post_graph_cleanup_matches = construct_sparse_matrix(post_graph_cleanup_matches, max_id)

    # Build the sparse matrix for the pre cleanup entity group matching score

    pre_clean_up_matches = pairwise_preds + pre_cleanup_transitive_matches_sparse

    # For each matches matrix, get true positives, false positives and false negatives and calculate the pairwise scores

    for matches_matrix in [pairwise_preds, pre_clean_up_matches, post_graph_cleanup_matches]:

        true_positives, false_positives = get_true_and_false_positives(ground_truth_sparse=ground_truth_sparse,
                                                                       predictions_sparse=matches_matrix)

        false_negatives = get_false_negatives(ground_truth_sparse=ground_truth_sparse,
                                              predictions_sparse=matches_matrix)

        precision, recall, f1_score = get_pairwise_scores(true_positives=true_positives,
                                                        false_positives=false_positives,
                                                        false_negatives=false_negatives,
                                                        big_subgraph_sizes=big_subgraph_sizes,
                                                        pairwise_pre_cleanup_scores=(matches_matrix is pre_clean_up_matches))


        if matches_matrix is pairwise_preds:
            scores_dict['pairwise_preds'] = {'precision': precision, 'recall': recall, 'f1_score': f1_score, 'true_positives': len(true_positives), 'false_positives': len(false_positives), 'false_negatives': len(false_negatives)}
        elif matches_matrix is pre_clean_up_matches:
            scores_dict['pre_cleanup_matches'] = {'precision': precision, 'recall': recall, 'f1_score': f1_score, 'true_positives': len(true_positives), 'false_positives': len(false_positives), 'false_negatives': len(false_negatives)}
        elif matches_matrix is post_graph_cleanup_matches:
            scores_dict['post_graph_cleanup_matches'] = {'precision': precision, 'recall': recall, 'f1_score': f1_score, 'true_positives': len(true_positives), 'false_positives': len(false_positives), 'false_negatives': len(false_negatives)}

    # Calculate graph metrics for the pre and post graph cleanup matches

    scores_dict['subgraph_purity'] = {'pre-cleanup': None, 'post-cleanup': None}
    scores_dict['number_of_subgraphs'] = {'pre-cleanup': None, 'post-cleanup': None}

    for matches_matrix in [pre_clean_up_matches, post_graph_cleanup_matches]:

        # Generate the matches graph from all the positive predictions
        matches_graph = nx.from_scipy_sparse_matrix(matches_matrix, create_using=nx.Graph())
        subgraph_purity = get_subgraph_purity(matches_graph, ground_truth_sparse)
        number_of_subgraphs = nx.number_connected_components(matches_graph)

        if matches_matrix is pre_clean_up_matches:
            scores_dict['subgraph_purity']['pre-cleanup'] = subgraph_purity
            scores_dict['number_of_subgraphs']['pre-cleanup'] = number_of_subgraphs
        elif matches_matrix is post_graph_cleanup_matches:
            scores_dict['subgraph_purity']['post-cleanup'] = subgraph_purity
            scores_dict['number_of_subgraphs']['post-cleanup'] = number_of_subgraphs

    return scores_dict

def get_max_id(ground_truth, pairwise_matches_preds, pre_cleanup_transitive_matches, post_graph_cleanup_matches):
    # Get the maximum id from the ground truth and all sets of matches
    max_id = max(ground_truth['lid'].max(), ground_truth['rid'].max(), 
                 pairwise_matches_preds['lid'].max(), pairwise_matches_preds['rid'].max(), 
                 pre_cleanup_transitive_matches['lid'].max(), pre_cleanup_transitive_matches['rid'].max(), 
                 post_graph_cleanup_matches['lid'].max(), post_graph_cleanup_matches['rid'].max())
    
    return max_id + 1


def filter_pairs_df(pairs_df, threshold):

    # Filter out the pairs under the probability threshold
    pairs_df = pairs_df[pairs_df['prob'] >= threshold]

    return pairs_df

def filter_ground_truth_pairs_df(pairs_df):
    # Filter out the negative pairs of the ground truth
    pairs_df = pairs_df[pairs_df['label'] == 1]

    return pairs_df

def transform_ids_to_positional_ids(df, dataset_name):
    # First load the test entity data file of the dataset
    test_entity_data_path = os.path.join('data', 'processed', dataset_name, 'test_entity_data.csv')
    test_entity_data = pd.read_csv(test_entity_data_path)
    test_entity_data['index'] = test_entity_data.index

    # Substitute lids and rids with the positional ids in the test entity data

    df = df.merge(test_entity_data[['index', 'id']], left_on='lid', right_on='id', how='left')
    df = df.rename(columns={'index': 'lid_pos'})
    df = df.drop(columns=['id'])

    df = df.merge(test_entity_data[['index', 'id']], left_on='rid', right_on='id', how='left')
    df = df.rename(columns={'index': 'rid_pos'})
    df = df.drop(columns=['id'])

    # Rename the positional ids to lids and rids
    df = df.drop(columns=['lid', 'rid'])
    df = df.rename(columns={'lid_pos': 'lid', 'rid_pos': 'rid'})

    return df 

def construct_sparse_matrix(df, max_id):
    sparse = coo_matrix(
        (
            np.ones(df.shape[0]),
            (df['lid'].values, df['rid'].values)
        ),
        shape=(max_id, max_id))
    
    # Make the matrix symmetric and all 1s
    sparse = sparse + sparse.T
    sparse.data = np.ones(sparse.data.shape[0])

    # Keep only the upper triangular part of the matrix (we do this to remove the duplicated pairs (i.e (A,B) and (B,A) pairs as well as the diagonal)
    sparse = sp.triu(sparse, k=1)

    return sparse



def get_true_and_false_positives(ground_truth_sparse, predictions_sparse):
    """
    Calculate True and False positives of the model predictions, to do this we iterate over the nonzero values of
    predictions_sparse.

    :param: ground_truth_sparse: The complete set of positive ground truth pairs in sparse format.
    :param: predictions_sparse: The complete set of positive pairs predicted by the model in sparse format.
    :return: true_positives, false_positives
    """
    ground_truth_sparse_csr = ground_truth_sparse.tocsr()
    rows, cols = predictions_sparse.nonzero()
    true_positives = set()
    false_positives = set()
    for row, col in zip(rows, cols):
        if ground_truth_sparse_csr[row, col] == 1 or ground_truth_sparse_csr[col,row] == 1: # We check both (A,B) and (B,A) in the ground truth
            true_positives.add((row, col))
        else:
            false_positives.add((row, col))

    return true_positives, false_positives

def get_false_negatives(ground_truth_sparse, predictions_sparse):
    """
    Calculate False Negatives of the model predictions, to do this we iterate over the nonzero values of
    ground_truth_sparse.

    :param: ground_truth_sparse: The complete set of positive ground truth pairs in sparse format.
    :param: all_predictions_sparse: The complete set of positive pairs predicted by the model in sparse format.
    :return: false_negatives

    """
    all_predictions_csr = predictions_sparse.tocsr()
    rows,cols = ground_truth_sparse.nonzero()
    false_negatives = set()
    for row,col in zip(rows,cols):
        if all_predictions_csr[row,col] == 0 and all_predictions_csr[col, row] == 0: # We check both (A,B) and (B,A) in the predictions
            false_negatives.add((row,col))

    return false_negatives

def get_pairwise_scores(true_positives, false_positives, false_negatives, big_subgraph_sizes = [], pairwise_pre_cleanup_scores = False):

    if pairwise_pre_cleanup_scores:
        false_positives = len(false_positives) + int(sum([subgraph_size * (subgraph_size - 1) / 2 for subgraph_size in big_subgraph_sizes]))
    else:
        false_positives = len(false_positives)

    precision = round(100 * len(true_positives) / (len(true_positives) + false_positives), 2)
    recall = round(100 * len(true_positives) / (len(true_positives) + len(false_negatives)), 2)
    f1_score = round((2 * precision * recall) /(precision + recall), 2)

    return precision, recall, f1_score


def get_subgraph_purity(matches_graph, ground_truth_sparse):

    subgraphs = list(nx.connected_components(matches_graph))

    # For each subgraph, calculate the % of true edges (including transitive edges)

    subgraph_purities = []
    ground_truth_sparse_csr = ground_truth_sparse.tocsr()

    for subgraph_idx, c in tqdm(enumerate(subgraphs), total=len(subgraphs), desc='Calculating subgraph purity'):
        # Get the edges of the subgraph from all_predictions_sparse
        subgraph_nodes = list(c)
        if len(subgraph_nodes) == 1:
            continue # Skip subgraphs with only one node
        purity = 0
        # Check how many of the edges of the subgraph are in the ground truth
        true_edges = ground_truth_sparse_csr[subgraph_nodes, :][:, subgraph_nodes].sum()
        purity += true_edges
        # Calculate the total number of edges in the complete subgraph
        number_of_edges = len(subgraph_nodes) * (len(subgraph_nodes) - 1) / 2
        purity = purity / number_of_edges
        # We keep track of the subgraph purity and its size
        subgraph_purities.append((purity, len(subgraph_nodes)))

    # To calculate overall subgraph purity, we weight each subgraph purity by its size and divide by the total number
    # of nodes

    subgraph_purity = sum([purity * size for purity, size in subgraph_purities]) / sum([size for purity, size in
                                                                                        subgraph_purities])
    return subgraph_purity

if __name__ == '__main__':
    exp_args = get_scores_args()


    all_scores_dict = {}        
    all_scores_dict['pairwise_preds'] = {'precision': [], 'recall': [], 'f1_score': [], 'true_positives': [], 'false_positives': [], 'false_negatives': []}
    all_scores_dict['pre_cleanup_matches'] = {'precision': [], 'recall': [], 'f1_score': [], 'true_positives': [], 'false_positives': [], 'false_negatives': []}
    all_scores_dict['post_graph_cleanup_matches'] = {'precision': [], 'recall': [], 'f1_score': [], 'true_positives': [], 'false_positives': [], 'false_negatives': []}
    all_scores_dict['subgraph_purity'] = {'pre-cleanup': [], 'post-cleanup': []}
    all_scores_dict['number_of_subgraphs'] = {'pre-cleanup': [], 'post-cleanup': []}

    for experiment_name in exp_args.experiment_names_list:
        matching_folder_path = dataset_results_folder_path__with_subfolders(subfolder_list=[exp_args.dataset_name, experiment_name])

        scores_dict = get_scores(matching_folder_path, exp_args.ground_truth_path, exp_args.dataset_name, exp_args.threshold, exp_args.non_positional_ids)

        all_scores_dict['pairwise_preds']['precision'].append(scores_dict['pairwise_preds']['precision'])
        all_scores_dict['pairwise_preds']['recall'].append(scores_dict['pairwise_preds']['recall'])
        all_scores_dict['pairwise_preds']['f1_score'].append(scores_dict['pairwise_preds']['f1_score'])
        all_scores_dict['pairwise_preds']['true_positives'].append(scores_dict['pairwise_preds']['true_positives'])
        all_scores_dict['pairwise_preds']['false_positives'].append(scores_dict['pairwise_preds']['false_positives'])
        all_scores_dict['pairwise_preds']['false_negatives'].append(scores_dict['pairwise_preds']['false_negatives'])

        all_scores_dict['pre_cleanup_matches']['precision'].append(scores_dict['pre_cleanup_matches']['precision'])
        all_scores_dict['pre_cleanup_matches']['recall'].append(scores_dict['pre_cleanup_matches']['recall'])
        all_scores_dict['pre_cleanup_matches']['f1_score'].append(scores_dict['pre_cleanup_matches']['f1_score'])

        all_scores_dict['post_graph_cleanup_matches']['precision'].append(scores_dict['post_graph_cleanup_matches']['precision'])
        all_scores_dict['post_graph_cleanup_matches']['recall'].append(scores_dict['post_graph_cleanup_matches']['recall'])
        all_scores_dict['post_graph_cleanup_matches']['f1_score'].append(scores_dict['post_graph_cleanup_matches']['f1_score'])

        all_scores_dict['subgraph_purity']['pre-cleanup'].append(scores_dict['subgraph_purity']['pre-cleanup'])
        all_scores_dict['subgraph_purity']['post-cleanup'].append(scores_dict['subgraph_purity']['post-cleanup'])

        all_scores_dict['number_of_subgraphs']['pre-cleanup'].append(scores_dict['number_of_subgraphs']['pre-cleanup'])
        all_scores_dict['number_of_subgraphs']['post-cleanup'].append(scores_dict['number_of_subgraphs']['post-cleanup'])


    # Print the true positives, false positives and false negatives for the pairwise predictions and the mean and std of all scores

    print('Scores for experiments: ' + str(exp_args.experiment_names_list) + ' on dataset: ' + str(exp_args.dataset_name))
    print('-' * 80)
    print('PAIRWISE PREDS')
    print('-' * 80)
    print('True positives: ' + str(np.mean(all_scores_dict['pairwise_preds']['true_positives'])) + ' +/- ' + str(np.std(all_scores_dict['pairwise_preds']['true_positives'])))
    print('False positives: ' + str(np.mean(all_scores_dict['pairwise_preds']['false_positives'])) + ' +/- ' + str(np.std(all_scores_dict['pairwise_preds']['false_positives'])))
    print('False negatives: ' + str(np.mean(all_scores_dict['pairwise_preds']['false_negatives'])) + ' +/- ' + str(np.std(all_scores_dict['pairwise_preds']['false_negatives'])))
    print('Precision: ' + str(np.mean(all_scores_dict['pairwise_preds']['precision'])) + ' +/- ' + str(np.std(all_scores_dict['pairwise_preds']['precision'])))
    print('Recall: ' + str(np.mean(all_scores_dict['pairwise_preds']['recall'])) + ' +/- ' + str(np.std(all_scores_dict['pairwise_preds']['recall'])))
    print('F1 score: ' + str(np.mean(all_scores_dict['pairwise_preds']['f1_score'])) + ' +/- ' + str(np.std(all_scores_dict['pairwise_preds']['f1_score'])))

    print('-' * 80)
    print('PRE CLEANUP MATCHES (PAIRWISE + TRANSITIVE)')
    print('-' * 80)
    print('Precision: ' + str(np.mean(all_scores_dict['pre_cleanup_matches']['precision'])) + ' +/- ' + str(np.std(all_scores_dict['pre_cleanup_matches']['precision'])))
    print('Recall: ' + str(np.mean(all_scores_dict['pre_cleanup_matches']['recall'])) + ' +/- ' + str(np.std(all_scores_dict['pre_cleanup_matches']['recall'])))
    print('F1 score: ' + str(np.mean(all_scores_dict['pre_cleanup_matches']['f1_score'])) + ' +/- ' + str(np.std(all_scores_dict['pre_cleanup_matches']['f1_score'])))

    print('-' * 80)
    print('POST GRAPH CLEANUP MATCHES')
    print('-' * 80)
    print('Precision: ' + str(np.mean(all_scores_dict['post_graph_cleanup_matches']['precision'])) + ' +/- ' + str(np.std(all_scores_dict['post_graph_cleanup_matches']['precision'])))
    print('Recall: ' + str(np.mean(all_scores_dict['post_graph_cleanup_matches']['recall'])) + ' +/- ' + str(np.std(all_scores_dict['post_graph_cleanup_matches']['recall'])))
    print('F1 score: ' + str(np.mean(all_scores_dict['post_graph_cleanup_matches']['f1_score'])) + ' +/- ' + str(np.std(all_scores_dict['post_graph_cleanup_matches']['f1_score'])))

    print('-' * 80)
    print('SUBGRAPH PURITY')
    print('-' * 80)
    print('Subgraph purity pre-cleanup: ' + str(np.mean(all_scores_dict['subgraph_purity']['pre-cleanup'])) + ' +/- ' + str(np.std(all_scores_dict['subgraph_purity']['pre-cleanup'])))
    print('Subgraph purity post-cleanup: ' + str(np.mean(all_scores_dict['subgraph_purity']['post-cleanup'])) + ' +/- ' + str(np.std(all_scores_dict['subgraph_purity']['post-cleanup'])))

    print('-' * 80)
    print('NUMBER OF SUBGRAPHS')
    print('-' * 80)
    print('Number of subgraphs pre-cleanup: ' + str(np.mean(all_scores_dict['number_of_subgraphs']['pre-cleanup'])) + ' +/- ' + str(np.std(all_scores_dict['number_of_subgraphs']['pre-cleanup'])))
    print('Number of subgraphs post-cleanup: ' + str(np.mean(all_scores_dict['number_of_subgraphs']['post-cleanup'])) + ' +/- ' + str(np.std(all_scores_dict['number_of_subgraphs']['post-cleanup'])))