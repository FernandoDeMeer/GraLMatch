import os
import sys
sys.path.append(os.getcwd())
import json
import argparse

from src.matching.matcher import *
from scripts.get_scores_matching import *

import pandas as pd


ENTITY_DATA_PATHS = {
    'finance/synthetic_companies': 'data/processed/synthetic_companies/seed_44/entity_data.csv',
    'finance/synthetic_securities': 'data/processed/synthetic_securities/seed_44/entity_data.csv',
    'wdc_products': 'data/processed/wdc/test_entity_data.csv'
}


CANDIDATE_ID_PATHS = {
    'finance/synthetic_companies': 'data/processed/synthetic_companies/full_test_candidates.csv',
    'finance/synthetic_securities': 'data/processed/synthetic_securities/full_test_candidates.csv',
    'wdc_products': 'data/processed/wdc/full_test_candidates.csv'
}


GROUND_TRUTH_PATH = {
    'finance/synthetic_companies': 'data/processed/synthetic_companies/seed_44/test__pre_split__given_matches.csv',
    'finance/synthetic_securities': 'data/processed/synthetic_securities/seed_44/test__pre_split__given_matches.csv',
    'wdc_products': 'data/processed/wdc/seed_2/test__pre_split__given_matches.csv'
}

def construct_ground_truth_matrix(ground_truth_df, dataset):
    ground_truth_sparse = coo_matrix(
        (
            np.ones(ground_truth_df.shape[0]),
            (ground_truth_df['lid'].values, ground_truth_df['rid'].values)
        ),
        shape=(dataset.shape[0], dataset.shape[0]))
    
    # Make the matrix symmetric and all 1s
    ground_truth_sparse = ground_truth_sparse + ground_truth_sparse.T
    ground_truth_sparse.data = np.ones(ground_truth_sparse.data.shape[0])

    # Keep only the upper triangular part of the matrix (we do this to remove the duplicated pairs (i.e (A,B) and (B,A) pairs as well as the diagonal)
    ground_truth_sparse = sp.triu(ground_truth_sparse, k=1)

    return ground_truth_sparse

def construct_predictions_matrix(predictions_df, dataset):

    predictions_sparse = coo_matrix((np.ones(predictions_df.shape[0]), (predictions_df['lid'].values,
                                                    predictions_df['rid'].values)),
                shape=(dataset.shape[0], dataset.shape[0]))

    # Make the matrix symmetric and all 1s
    predictions_sparse = predictions_sparse + predictions_sparse.T
    predictions_sparse.data = np.ones(predictions_sparse.data.shape[0])

    # Keep only the upper triangular part of the matrix (we do this to remove the duplicated pairs (i.e (A,B) and (B,A) pairs as well as the diagonal)
    predictions_sparse = sp.triu(predictions_sparse, k=1)

    return predictions_sparse

def preprocess_pairs_df(pairs_df, test_entity_data, threshold=None):

    # Remove all duplicated pairs i.e remove (rid, lid) if (lid, rid) is also in the df
    cols = ['lid', 'rid']
    pairs_df[cols] = np.sort(pairs_df[cols].values, axis=1)
    if 'prob' in pairs_df.columns:
        pairs_df = pairs_df.sort_values(by=cols + ['prob'])
        if threshold is not None:
            pairs_df = pairs_df[pairs_df['prob'] >= threshold]
    
    pairs_df = pairs_df.drop_duplicates(subset=cols, keep='last')

    # Remove all pairs where lid == rid
    pairs_df = pairs_df[pairs_df['lid'] != pairs_df['rid']]

    # All the lid and rid values are w.r.t the raw dataset, for the purpose of the scores we need to convert them to the corresponding index in the test_entity_data

    # Add the index column to the test_entity_data
    test_entity_data['index'] = np.arange(test_entity_data.shape[0])
    pairs_df = pairs_df.merge(test_entity_data[['id', 'index']], left_on='lid', right_on='id', how='left')
    pairs_df = pairs_df.rename(columns={'index': 'lid_index'})
    pairs_df = pairs_df.drop(columns=['id', 'lid'])
    pairs_df = pairs_df.merge(test_entity_data[['id', 'index']], left_on='rid', right_on='id', how='left')
    pairs_df = pairs_df.rename(columns={'index': 'rid_index'})
    pairs_df = pairs_df.drop(columns=['id', 'rid'])

    # rename the lid_index and rid_index columns to lid and rid

    pairs_df = pairs_df.rename(columns={'lid_index': 'lid', 'rid_index': 'rid'})

    return pairs_df

# entity_df: entity_df
# pairwise_matches_preds: df_ids
# ground_truth: df_ids
#
def get_score(entity_df, pairwise_matches_preds, results_path, ground_truth, threshold):
    """
    Computes all the scores of a given matching, including pairwise scores and pre/post graph cleanup entity group matching scores.
    """
    scores_dict = {}

    # Load the test entity data and the ground truth

    # test_entity_data = pd.read_csv(os.path.join(dataset_processed_folder_path(dataset_name), 'test_entity_data.csv'), index_col=0)
    # ground_truth = pd.read_csv(ground_truth_file_path, index_col=0)

    # Load the pairwise predictions, the subgraph size list (if applicable), the pre-graph cleanup transitive matches and all the post-graph cleanup matches

    # pairwise_matches_preds = pd.read_csv(os.path.join(matching_folder_path, 'pairwise_matches_preds.csv'), header=0)

    # Check for the big subgraphs size list
    if os.path.isfile(os.path.join(results_path, 'big_subgraph_sizes.csv')):
        big_subgraph_sizes_df= pd.read_csv(os.path.join(results_path, 'big_subgraph_sizes.csv'), header=0)
        big_subgraph_sizes = big_subgraph_sizes_df['size'].values.tolist()
    else:
        big_subgraph_sizes = []

    pre_cleanup_transitive_matches = pd.read_csv(os.path.join(results_path, 'pre_cleanup_transitive_matches.csv'), header=0)
    post_graph_cleanup_matches = pd.read_csv(os.path.join(results_path, 'post_graph_cleanup_matches.csv'), header=0)

    # Preprocess the ground truth and the predictions to remove duplicated pairs (i.e (A,B) and (B,A) pairs) and convert the ids to the corresponding index in the test_entity_data

    ground_truth = preprocess_pairs_df(ground_truth, entity_df)  
    pairwise_matches_preds = preprocess_pairs_df(pairwise_matches_preds, entity_df, threshold=threshold)
    pre_cleanup_transitive_matches = preprocess_pairs_df(pre_cleanup_transitive_matches, entity_df)
    post_graph_cleanup_matches = preprocess_pairs_df(post_graph_cleanup_matches, entity_df)

    # Construct the sparse coo-matrices of pairs and labels for the ground truth and the predictions

    ground_truth_sparse = construct_ground_truth_matrix(ground_truth, dataset=entity_df)

    pairwise_preds = construct_predictions_matrix(pairwise_matches_preds, dataset=entity_df)
    pre_cleanup_transitive_matches_sparse = construct_predictions_matrix(pre_cleanup_transitive_matches, dataset=entity_df)
    post_graph_cleanup_matches = construct_predictions_matrix(post_graph_cleanup_matches, dataset=entity_df)

    # Build the sparse matrix for the pre cleanup entity group matching score

    pre_clean_up_matches = pairwise_preds + pre_cleanup_transitive_matches_sparse

    # For each matches matrix, get true positives, false positives and false negatives and calculate the pairwise scores

    for matches_matrix in [pairwise_preds, pre_clean_up_matches, post_graph_cleanup_matches]:

        true_positives, false_positives = get_true_and_false_positives(ground_truth_sparse=ground_truth_sparse,
                                                                       predictions_sparse=matches_matrix)

        false_negatives = get_false_negatives(ground_truth_sparse=ground_truth_sparse,
                                              predictions_sparse=matches_matrix)

        accuracy, recall, f1_score = get_pairwise_scores(true_positives=true_positives,
                                                        false_positives=false_positives,
                                                        false_negatives=false_negatives,
                                                        big_subgraph_sizes=big_subgraph_sizes,
                                                        pairwise_pre_cleanup_scores=(matches_matrix is pre_clean_up_matches))
        
        if matches_matrix is pairwise_preds:
            scores_dict['pairwise_preds'] = {'accuracy': accuracy, 'recall': recall, 'f1_score': f1_score}
        elif matches_matrix is pre_clean_up_matches:
            scores_dict['pre_cleanup_matches'] = {'accuracy': accuracy, 'recall': recall, 'f1_score': f1_score}
        elif matches_matrix is post_graph_cleanup_matches:
            scores_dict['post_graph_cleanup_matches'] = {'accuracy': accuracy, 'recall': recall, 'f1_score': f1_score}

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



def load_config(args):
    configs = json.load(open('../em_ditto/configs.json'))
    configs = {conf['name'] : conf for conf in configs}
    config = configs[args.task]
    return config


def main(args):
    config = load_config(args)
    df_proba = pd.read_csv(args.results_file_path + '/test_candidate_probabilities.csv')
    df_entity = pd.read_csv(ENTITY_DATA_PATHS[args.task])
    df_ids = pd.read_csv(CANDIDATE_ID_PATHS[args.task])
    df_ids['prob'] = df_proba['probabilities']
    df_ground_truth = pd.read_csv(GROUND_TRUTH_PATH[args.task])
    df_ground_truth = df_ground_truth[df_ground_truth['lid'] != df_ground_truth['rid']]
    threshold = df_proba.loc[0, 'threshold'],
    print(f"Using Threshold={threshold}")

    if 'synthetic' in args.task:
        num_data_sources = 5
    else:
        if 'data_source_id' in df_entity.columns:
            num_data_sources = df_entity['data_source_id'].nunique()
        else:
            num_data_sources = 8


    if args.task == 'finance/real_securities':
        matcher_class = SecurityMatcher
    elif args.task == 'finance/real_companies':
        matcher_class = RealCompanyMatcher
    elif args.task == 'finance/synthetic_securities':
        matcher_class = SynSecurityMatcher
    elif args.task == 'finance/synthetic_companies':
        matcher_class = SynCompanyMatcher
    elif args.task == 'wdc_products':
        matcher_class = WDCMatcher
    else:
        raise ValueError(f'The task {args.task} does not have an associated matcher class.')

    matcher = matcher_class(model=None, processed_folder_path=args.processed_folder_path, results_path=args.results_file_path)
    matches_graph = matcher.pre_cleanup(df_ids, threshold=df_proba.loc[0, 'threshold'], num_of_datasources=num_data_sources)
    matcher.graph_cleanup(matches_graph, num_of_datasources=num_data_sources)
    
    df_ids = df_ids[['lid', 'rid']]
    score_dict = get_score(df_entity, df_ids, matcher.results_path, df_ground_truth, threshold=threshold)

    

    print()
    print(f"Pairwise Matching Performance")
    print(f"\tPrecision: {score_dict['pairwise_preds']['accuracy']:.2f}\tRecall: {score_dict['pairwise_preds']['recall']:.2f}\tF1-Score: {score_dict['pairwise_preds']['f1_score']:.2f}")
    print()
    print(f"Pre Graph Cleanup")
    print(f"\tPrecision: {score_dict['pre_cleanup_matches']['accuracy']:.2f}\tRecall: {score_dict['pre_cleanup_matches']['recall']:.2f}\tF1-Score: {score_dict['pre_cleanup_matches']['f1_score']:.2f}\tCluster Purity: {score_dict['subgraph_purity']['pre-cleanup']:.2f}")
    print()
    print(f"Post Graph Cleanup")
    print(f"\tPrecision: {score_dict['post_graph_cleanup_matches']['accuracy']:.2f}\tRecall: {score_dict['post_graph_cleanup_matches']['recall']:.2f}\tF1-Score: {score_dict['post_graph_cleanup_matches']['f1_score']:.2f}\tCluster Purity: {score_dict['subgraph_purity']['post-cleanup']:.2f}")
    print()



    def sort_drop(df):
        cols = ['lid', 'rid']
        df[cols] = np.sort(df[cols].values, axis=1)
        df = df.drop_duplicates()
        df = df[df['lid'] != df['rid']]
        return df

    gt = pd.read_csv(GROUND_TRUTH_PATH[args.task])
    gt = sort_drop(gt)

    c = pd.read_csv(CANDIDATE_ID_PATHS[args.task])[['lid', 'rid']]
    c = sort_drop(c)

    overlap = c.merge(gt, left_on=['lid', 'rid'], right_on=['lid', 'rid'])
    blocking_recall = overlap.shape[0] / gt.shape[0]
    print(f"Blocking has a recall of {100*blocking_recall:.2f} %")
    print()

    import code; code.interact(local=dict(globals(), **locals()))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="finance/synthetic_securities")
    parser.add_argument('--results_file_path', type=str, default='../em_ditto/checkpoints/synthetic_securities128/finance/synthetic_securities')
    parser.add_argument('--processed_folder_path', type=str, default='data/processed/synthetic_securities')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
