import os
import pickle
from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

from scripts.synthetic_data.create_train_val_test_pairs import \
    get_train_val_test_split
from src.data import full_data_utils
from src.data.dataset import DataincDataset
from src.helpers.path_helper import *
from src.models.config import Config
from src.models.pytorch_model import PyTorchModel


class Matcher(ABC):

    def __init__(self, model: PyTorchModel, processed_folder_path: str = None, results_path: str = None):
        self.model = model
        if processed_folder_path:
            self.processed_folder_path = processed_folder_path
        else:
            self.processed_folder_path = dataset_processed_folder_path(dataset_name=self.model.dataset.name)

        if results_path:
            self.results_path = results_path
        else:
            self.results_path = dataset_results_folder_path__with_subfolders(subfolder_list=[self.model.dataset.name, self.model.args.experiment_name])

    def test_records_from_positive_matches(self, test_entity_data: pd.DataFrame):
        """
        Filter the given test_entity_data down to the records that are actually
        part of the positive matches in the test set, i.e. removing all the records
        that are only part of it because they are one half of a negative match.
        """
        test_df = self.model.dataset.test_df
        test_df = test_df[test_df['label'] == 1]

        # all the unique values of 'lid' and 'rid' in test_df
        test_ids = set()
        test_ids.update(list(test_df['lid']))
        test_ids.update(list(test_df['rid']))

        test_records = test_entity_data[test_entity_data['id'].isin(test_ids)]
        return test_records

    @abstractmethod
    def blocking(self, test_entity_data: pd.DataFrame) -> pd.DataFrame:
        """
        takes the full test_entity_data and creates candidate pairs from it
        """

        raise NotImplementedError("Should be implemented in the respective subclasses.")

    def get_test_entity_data(self) -> pd.DataFrame:
        """
        basic implementation to get test_entity_data
        can be overwritten if needed by the respective subclass
        """
        # Check if the test records have already been previously saved
        test_folder_path = os.path.join(self.processed_folder_path, 'test_entity_data.csv')
        if file_exists_or_create(test_folder_path):
            return pd.read_csv(test_folder_path)

        test_id_df = self.model.dataset.test_df

        full_entity_data = self.model.dataset.get_entity_data()

        test_ids = set()
        test_ids.update(list(test_id_df['lid']))
        test_ids.update(list(test_id_df['rid']))

        test_entity_data = full_entity_data[full_entity_data['id'].isin(test_ids)]
        # Save the test_entity_data
        test_entity_data.to_csv(os.path.join(self.processed_folder_path, 'test_entity_data.csv'), index=False)

        return test_entity_data

    def save_test_candidates(self, candidates_df: pd.DataFrame):
        """
        saves a blocked candidates_df with cols (lid, rid, label, match_type) in the processed path of the ds
        """
        # Delete the rows with lid == rid
        candidates_df = candidates_df[candidates_df['lid'] != candidates_df['rid']]
        # Delete duplicates
        candidates_df = candidates_df.drop_duplicates(subset=['lid', 'rid'])
        # Save the test candidates
        candidates_df.to_csv(os.path.join(self.processed_folder_path, 'full_test_candidates.csv'), index=False)

    def pre_cleanup(self, pairwise_matches_preds: pd.DataFrame, threshold: float = 0.999, num_of_datasources: int = 5):
        """
        Pre cleanup step before running the graph cleanup. In the base class, we just return the matches_graph.
        In subclasses we addtionally break up large subgraphs that make the graph cleanup run too long.
        """
        matches_graph = full_data_utils.generate_matches_graph(pairwise_matches_preds, threshold=threshold)

        _, transitive_matches = full_data_utils.generate_transitive_matches_graph(matches_graph,
                                                                                  add_transitive_edges=False,
                                                                                  results_path=self.results_path,
                                                                                  subgraph_size_threshold=100)

        # Save the pre-cleanup transitive matches
        transitive_matches_df = pd.DataFrame(transitive_matches, columns=['lid', 'rid', 'match_type'])
        transitive_matches_df.to_csv(os.path.join(self.results_path, 'pre_cleanup_transitive_matches.csv'), index=False)

        return matches_graph



    def graph_cleanup(self, matches_graph, num_of_datasources=5):
        """
        Clean up the matches graph by breaking up large subgraphs and removing edges with high betweenness centrality
        :param num_of_datasources: The number of data sources in the dataset
        :return: files of the cleaned up matches graph and the deleted edges
        """

        print('=' * 50)
        print('Starting graph cleanup')
        print('=' * 50)
        deleted_edges_dict = {'lid': [], 'rid': [], 'prob': [], 'match_type': []}

        ###############################################################################################################

        # 1st Cleanup: Break up subgraphs bigger than 5*num_of_datasources via minimum edge cuts.

        ###############################################################################################################

        print('=' * 50)
        print('Starting minimum edge cut cleanup')
        print('=' * 50)

        # While there are subgraphs with more than 5*number_of_datasources nodes, we break them up via minimum edge cuts
        subgraphs = list(nx.connected_components(matches_graph))

        while any([len(c) > 5 * num_of_datasources for c in subgraphs]):
            largest_subgraph = max(subgraphs, key=len)
            print('Largest subgraph size: {}, Number of subgraphs: {}'.format(len(largest_subgraph),
                                                                              len(subgraphs)))
            # Gather all subgraphs with the maximum size
            largest_subgraphs = [c for c in subgraphs if len(c) == len(largest_subgraph)]

            # Clean up the largest subgraphs via minimum edge cuts
            for subgraph_idx, c in enumerate(largest_subgraphs):

                matches_graph, deleted_edges_dict = self.minimum_edge_cut_clean_up(c, matches_graph, deleted_edges_dict)

            subgraphs = list(nx.connected_components(matches_graph))

        ###############################################################################################################

        # 2nd Cleanup: Remove the edges with the highest betweenness centrality in each subgraph with more than
        # num_of_datasources nodes.

        ###############################################################################################################
        print('=' * 50)
        print('Starting betweenness centrality cleanup')
        print('=' * 50)

        subgraphs = list(nx.connected_components(matches_graph))

        while any([len(c) > num_of_datasources for c in subgraphs]):
            # Get the subgraphs with more than num_of_datasources nodes
            large_subgraphs = [c for c in subgraphs if len(c) > num_of_datasources]
            print('Number of subgraphs with more than {} nodes: {}, Number of subgraphs: {}'.format(num_of_datasources,
                                                                                                    len(large_subgraphs),
                                                                                                    len(subgraphs)))

            # Clean up the large subgraphs via removing the edge with the highest betweenness centrality
            for subgraph_idx, c in enumerate(large_subgraphs):
                # Compute the betweenness centrality of all edges of the subgraph
                subgraph = matches_graph.subgraph(large_subgraphs[subgraph_idx])
                betweenness_centrality = nx.edge_betweenness_centrality(subgraph)
                # Get the edge with the highest betweenness centrality
                max_betweenness_edge = max(betweenness_centrality, key=betweenness_centrality.get)
                # Record the deleted edge on the deleted_edges_dict
                deleted_edges_dict['lid'].append(max_betweenness_edge[0])
                deleted_edges_dict['rid'].append(max_betweenness_edge[1])
                deleted_edges_dict['prob'].append(matches_graph[max_betweenness_edge[0]][max_betweenness_edge[1]]['prob'])
                deleted_edges_dict['match_type'].append(matches_graph[max_betweenness_edge[0]][max_betweenness_edge[1]]['match_type'])
                # Remove the edge with the highest betweenness centrality from the matches_graph
                matches_graph.remove_edge(max_betweenness_edge[0], max_betweenness_edge[1])

            subgraphs = list(nx.connected_components(matches_graph))

        ###############################################################################################################

        # Final Graph Step: Add all the transitive edges of each subgraph to the matches_graph

        ###############################################################################################################

        matches_graph, _ = full_data_utils.generate_transitive_matches_graph(matches_graph, True)

        # Save the edges of the post graph cleanup matches_graph, with their match_type

        matches_graph_df = pd.DataFrame(matches_graph.edges(data=True), columns=['lid', 'rid', 'match_type'])
        matches_graph_df.to_csv(os.path.join(self.results_path, 'post_graph_cleanup_matches.csv'), index=False)

        # Save the deleted edges

        deleted_edges_df = pd.DataFrame(deleted_edges_dict)
        deleted_edges_df.to_csv(os.path.join(self.results_path, 'graph_cleanup_deleted_edges.csv'), index=False)

        print('Finished graph cleanup')

    ###########################################################################

    # Utils

    ###########################################################################

    def minimum_edge_cut_clean_up(self, subgraph, matches_graph, deleted_edges_dict):

        # Get the minimum edge cut of the subgraph
        subgraph = matches_graph.subgraph(subgraph)
        min_edge_cut = nx.minimum_edge_cut(subgraph)

        # Save the deleted edges with their lid, rid, prob attributes
        deleted_edges_lids = [lid for lid, rid in min_edge_cut]
        deleted_edges_rids = [rid for lid, rid in min_edge_cut]
        deleted_edges_probs = [matches_graph[lid][rid]['prob'] for lid, rid in min_edge_cut]
        deleted_edges_match_types = [matches_graph[lid][rid]['match_type'] for lid, rid in min_edge_cut]
        deleted_edges_dict['lid'].extend(deleted_edges_lids)
        deleted_edges_dict['rid'].extend(deleted_edges_rids)
        deleted_edges_dict['prob'].extend(deleted_edges_probs)
        deleted_edges_dict['match_type'].extend(deleted_edges_match_types)

        # Remove the cut edges from the graph
        matches_graph.remove_edges_from(min_edge_cut)

        return matches_graph, deleted_edges_dict


    def run_matching(self, args):
        """
        Runs the whole matching pipeline:
        - A) blocking
        - B) pairwise matching
        - C) graph cleanup
        """

        test_entity_data = self.get_test_entity_data()

        # A) get candidates using the blocking function
        if args.company_matching:
            candidate_df = self.blocking(test_entity_data, args.company_matching)
        else:
            candidate_df = self.blocking(test_entity_data)
        # drop match_type for now to use the testing function
        candidate_idx_df = candidate_df.drop(columns=['match_type'])

        # inject the candidates into the test_data_loader, not the best way, but quick for now
        self.model.test_data_loader.dataset.idx_df = candidate_idx_df

        # B) run pairwise matching

        # First check if the pairwise_matches_preds have already been previously saved
        pairwise_matches_preds_path = os.path.join(self.results_path, 'pairwise_matches_preds.csv')

        if file_exists_or_create(pairwise_matches_preds_path):
            self.pairwise_matches_preds = pd.read_csv(pairwise_matches_preds_path)
        else:
            # Run the pairwise matching
            self.model.test(epoch=args.epoch) 
            self.pairwise_matches_preds = self.load_and_save_pairwise_matches_preds(args, candidate_df)

        # C) perform graph_cleanup
        if 'data_source_id' in test_entity_data.columns:
            num_ds = test_entity_data['data_source_id'].nunique()
        else:
            # Get the raw_df
            raw_df = self.model.dataset.get_raw_df()
            if 'data_source_id' in raw_df.columns:
                num_ds = raw_df['data_source_id'].nunique()
            else:
                # If the dataset doesn't have a set number of data sources, we set num_ds arbitrarily to 5.
                num_ds = 5
        
        #Check if threshold is set in args
        if hasattr(args, 'threshold'):
            self.matches_graph = self.pre_cleanup(self.pairwise_matches_preds, threshold=args.threshold, num_of_datasources=num_ds)
        else:
            self.matches_graph = self.pre_cleanup(self.pairwise_matches_preds, threshold=0.999, num_of_datasources=num_ds)

        self.graph_cleanup(self.matches_graph, num_of_datasources=num_ds)

    def load_and_save_pairwise_matches_preds(self, args, candidate_df):
        """
        Loads the pairwise_matches_preds from the prediction_log and saves them to the processed folder
        """
        file_name = "".join([self.model.args.model_name, '__prediction_log__ep', str(args.epoch), '.csv'])
        log_path = experiment_file_path(args.experiment_name, file_name)

        pairwise_matches_preds = pd.read_csv(log_path)
        # Add the match_type column to the pairwise_matches_preds
        pairwise_matches_preds = pairwise_matches_preds.merge(candidate_df[['lid', 'rid', 'match_type']], left_on=['lids', 'rids'], right_on=['lid', 'rid'])
        pairwise_matches_preds = pairwise_matches_preds.drop(columns=['labels', 'predictions', 'lid', 'rid'])
        # Rename the column 'prediction_proba' to 'prob'
        pairwise_matches_preds = pairwise_matches_preds.rename(columns={'prediction_proba': 'prob'})
        # Rename the lids and rids columns to lid and rid
        pairwise_matches_preds = pairwise_matches_preds.rename(columns={'lids': 'lid', 'rids': 'rid'})

        # Save the pairwise_matches_preds
        pairwise_matches_preds.to_csv(os.path.join(self.results_path, 'pairwise_matches_preds.csv'), index=False)

        return pairwise_matches_preds
    

    ###########################################################################

    # Blocking Utils

    ###########################################################################

    def get_tknzd_records_and_overlap_indicators(self, test_entity_data):
        tokenized_records = self.model.dataset.get_tokenized_data()
        # The tokenized records are indexed by the id of the raw data
        tokenized_test_records = tokenized_records[tokenized_records.index.isin(test_entity_data['id'])]
        # Generate the list of all tokens seen in the test records
        tmp_list = tokenized_test_records['tokenized'].apply(lambda x: list(set(x)))
        tmp_list = tmp_list.apply(lambda x: list(set(x)))
        all_tokens = np.array(list(set(string for sublist in tmp_list for string in sublist)))
        # Index structure for much faster lookup of the index positions of every token in the all_tokens list
        # (so that we do not have to call all_tokens.index but rather get a O(1) lookup)
        index_lookup = {value: i for i, value in enumerate(all_tokens)}
        # Generating a sparse matrix with (n_records, n_tokens), where a 1 at (recordX, tokenY) indicates,
        # that recordX contains the tokenY
        #
        data = []
        row = []
        col = []
        for i, (record_id, tokenized_record) in tqdm(enumerate(tokenized_test_records.iterrows()),
                                                        total=tokenized_test_records.shape[0],
                                                        desc='Building indices matrix'):
            token_indexes_in_record = sorted(set([index_lookup[t] for t in tokenized_record['tokenized']]))

            n_tokens = len(token_indexes_in_record)
            data.extend([True for _ in range(n_tokens)])
            row.extend([i for _ in range(n_tokens)])
            col.extend(token_indexes_in_record)
        indicators = csr_matrix((data, (row, col)), shape=(tokenized_test_records.shape[0], len(all_tokens)),
                                dtype=np.int8)
        return indicators, tokenized_test_records

    def get_top_overlap_idx(self, i, indicators, test_entity_data):
            lookup = np.array(indicators[i, :].dot(indicators.transpose()).todense())[0]
            # Set all records from the same data source to zero, because we only want matches with other data sources
            current_data_source = test_entity_data.iloc[i]['data_source_id']
            multiplication_mask = np.ones(test_entity_data.shape[0], dtype=np.int8) - \
                np.array(test_entity_data['data_source_id'] == current_data_source)
            lookup *= multiplication_mask
            top_overlap_idx = np.argpartition(lookup, -self.number_of_candidates)[-self.number_of_candidates:]
            return top_overlap_idx

    def get_top_overlap_idx_one_source(self, i, indicators, test_entity_data):
            lookup = np.array(indicators[i, :].dot(indicators.transpose()).todense())[0]
            top_overlap_idx = np.argpartition(lookup, -self.number_of_candidates)[-self.number_of_candidates:]
            return top_overlap_idx


class SecurityMatcher(Matcher):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def blocking(self, test_entity_data: pd.DataFrame, company_matching: str) -> pd.DataFrame:

        candidates_path = os.path.join(self.processed_folder_path, 'full_test_candidates.csv')
        if file_exists_or_create(candidates_path):
            return pd.read_csv(candidates_path)

        raw_data_df = self.model.dataset.get_raw_df()
        
        candidate_pairs = []
        # need to iterate ONLY over those records, that are actually part of the 20% test set,
        # rather than over all records that are part of the test dataset. That means, if (A,B) is
        # a negative match, while (A,C) is a positive match, then (A,C) are part of the test set, but
        # B is not (however it is required to be available in the entity data to generate the sequence
        # for the transformer)
        #
        test_records = self.test_records_from_positive_matches(test_entity_data)
        for index, row in tqdm(test_records.iterrows(), total=test_records.shape[0], desc='Gathering id_overlap candidate pairs'):
            identifiers = dict()
            for identifier in full_data_utils.ID_ATTRIBUTES_REAL.values():
                id_value = row[identifier]
                try:
                    identifiers[identifier] = [] if (id_value == '' or pd.isnull(id_value)) else [id_value]
                except TypeError:
                    identifiers[identifier] = [] 

            # Check for securities that share any of the identifiers in other data sources
            matching_securities = test_records[
                (
                    (test_records['active_isin'].isin(identifiers['active_isin']))
                    | (test_records['active_cusip'].isin(identifiers['active_cusip']))
                    | (test_records['active_valor'].isin(identifiers['active_valor']))
                    | (test_records['active_sedol'].isin(identifiers['active_sedol']))
                ) &
                (test_records['data_source_id'] != row['data_source_id'])
            ]

            # add label
            for _, sec in matching_securities.iterrows():
                matching_sec_raw = raw_data_df[raw_data_df['id'] == sec['id']]
                entity_row_raw = raw_data_df[raw_data_df['id'] == row['id']]
                if ((matching_sec_raw['nexus_id'].item() == entity_row_raw['nexus_id'].item())
                    & (matching_sec_raw['tag'].item() == 1)
                        & (entity_row_raw['tag'].item() == 1)):
                    label = 1
                else:
                    label = 0

                candidate_pairs.append((row['id'], sec['id'], label, 'id_overlap'))

        # Use the company_matching as a blocking for the securities, i.e. only consider securities that are
        # issued by companies that are considered as a match by the company_matching
        company_matches_df = pd.read_csv(os.path.join(dataset_results_folder_path__with_subfolders(subfolder_list=['companies', company_matching]), 'post_graph_cleanup_matches.csv'))
        companies_df = pd.read_csv(dataset_raw_file_path(Config.DATASETS['companies']))
        companies_df.reset_index(drop=True, inplace=True)
        companies_df.drop(columns=['id'], inplace=True)
        companies_df.insert(0, 'id', value=companies_df.index)

        for index, row in tqdm(test_records.iterrows(), total=test_records.shape[0], desc = 'Gathering candidate pairs from the company_matching'):
            # Get the issuer_id, data_source_id of the current security
            issuer_id_data_source, data_source_id = row['issuer_id'], row['data_source_id']
            # Get the issuing company of the current security
            issuing_company = companies_df[(companies_df['external_id'] == issuer_id_data_source) & (companies_df['data_source_id'] == data_source_id)]
            if len(issuing_company) > 0:
                issuer_id = issuing_company['id'].item()
                # Check for all the records matched to the issuing company in the company_matching
                matching_companies_pairs = company_matches_df[(company_matches_df['lid'] == issuer_id) | (company_matches_df['rid'] == issuer_id)]
                # Get the ids of the matching companies
                matching_companies_ids = list(set(matching_companies_pairs['lid'].unique()).union(set(matching_companies_pairs['rid'].unique())).difference(set([issuer_id])))
                # Get the external_id, data_source_id of the matching companies
                matching_companies = companies_df[companies_df['id'].isin(matching_companies_ids)]
                matching_companies = matching_companies[['external_id', 'data_source_id']]
                # Get the securities issued by the matching companies for each (external_id, data_source_id) pair
                for external_id, data_source_id in list(matching_companies.values):
                    matching_securities = raw_data_df[(raw_data_df['issuer_id'] == external_id) & (raw_data_df['data_source_id'] == data_source_id)]
                    # Add the label and finalize the candidate pair
                    for _, sec in matching_securities.iterrows():
                        matching_sec_raw = raw_data_df[raw_data_df['id'] == sec['id']]
                        entity_row_raw = raw_data_df[raw_data_df['id'] == row['id']]
                        if ((matching_sec_raw['nexus_id'].item() == entity_row_raw['nexus_id'].item())
                            & (matching_sec_raw['tag'].item() == 1)
                                & (entity_row_raw['tag'].item() == 1)):
                            label = 1
                        else:
                            label = 0
                        candidate_pairs.append((row['id'], sec['id'], label, 'company_matching'))

        test_candidates_df = pd.DataFrame(data=candidate_pairs, columns=['lid', 'rid', 'label', 'match_type'])

        self.save_test_candidates(test_candidates_df)

        return test_candidates_df
        

class CompanyMatcher(Matcher, ABC):
    def __init__(self, id_attributes, number_of_candidates: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.number_of_candidates = number_of_candidates
        self.id_attributes = id_attributes

    def get_matching_securities(self, company, raw_sec_df) -> pd.DataFrame:
        associated_securities = raw_sec_df[(raw_sec_df['issuer_id'] == company['external_id']) & (
            raw_sec_df['data_source_id'] == company['data_source_id'])]
        if len(associated_securities) == 0:
            return associated_securities
        identifiers = dict()
        for identifier in self.id_attributes.values():
            id_values = associated_securities[identifier].dropna().unique()
            try:
                identifiers[identifier] = [] if len(id_values) == 0 else id_values
            except TypeError:
                identifiers[identifier] = []
        # Check for securities that share any of the identifiers in other data sources
        matching_securities = raw_sec_df[
            (
                (raw_sec_df[self.id_attributes['isin']].isin(identifiers[self.id_attributes['isin']]))
                | (raw_sec_df[self.id_attributes['cusip']].isin(identifiers[self.id_attributes['cusip']]))
                | (raw_sec_df[self.id_attributes['valor']].isin(identifiers[self.id_attributes['valor']]))
                | (raw_sec_df[self.id_attributes['sedol']].isin(identifiers[self.id_attributes['sedol']]))
            ) &
            (raw_sec_df['data_source_id'] != company['data_source_id'])
        ]
        return matching_securities

class RealCompanyMatcher(CompanyMatcher):
    def __init__(self, **kwargs):
        super().__init__(id_attributes=full_data_utils.ID_ATTRIBUTES_REAL, **kwargs)

    def blocking(self, test_entity_data: pd.DataFrame) -> pd.DataFrame:

        candidates_path = os.path.join(self.processed_folder_path, 'full_test_candidates.csv')
        if file_exists_or_create(candidates_path):
            return pd.read_csv(candidates_path)

        raw_data_df = self.model.dataset.get_raw_df()

        # join external_id to test_entity_data
        test_entity_data = test_entity_data.join(
            raw_data_df[['id', 'external_id', 'data_source_id']], on='id', rsuffix='_raw').drop(columns='id_raw')

        pos_matches = self.model.dataset.get_matches()

        raw_sec_file_path = dataset_raw_file_path(Config.DATASETS['securities_full_with_company_desc_and_names'])
        raw_sec_df = pd.read_csv(raw_sec_file_path)

        candidate_pairs = []
        # need to iterate ONLY over those records, that are actually part of the 20% test set,
        # rather than over all records that are part of the test dataset. That means, if (A,B) is
        # a negative match, while (A,C) is a positive match, then (A,C) are part of the test set, but
        # B is not (however it is required to be available in the entity data to generate the sequence
        # for the transformer)
        #
        test_records = self.test_records_from_positive_matches(test_entity_data)
        for index, company in tqdm(test_records.iterrows(), total=test_records.shape[0],
                                   desc='Gathering id_overlap candidate pairs'):
            matching_securities = self.get_matching_securities(company, raw_sec_df)

            if len(matching_securities) == 0:
                continue
            else:
                # Find the matching companies based on the matching securities
                company_candidate_ids = test_records.merge(matching_securities[['issuer_id', 'data_source_id']],
                                                               left_on=['external_id', 'data_source_id'],
                                                               right_on=['issuer_id', 'data_source_id'])['id'].unique()

                # Assign the label and finalize candidate pair
                cur_comp_id = company['id']
                for cand_id in company_candidate_ids:
                    if cur_comp_id == cand_id:
                        continue

                    # because both A, B and B, A are in pos_matches we only need to check for one
                    label = 1 if 0 < len(pos_matches[(pos_matches['lid'] == cur_comp_id) & (pos_matches['rid'] == cand_id)]) else 0
                    candidate_pairs.append((cur_comp_id, cand_id, label, 'id_overlap'))

        indicators, tokenized_test_records = self.get_tknzd_records_and_overlap_indicators(test_records)

        # Looping over the records to find the most similar (most overlapping tokens) records
        for i, (record_id, tokenized_record) in tqdm(enumerate(tokenized_test_records.iterrows()), total=tokenized_test_records.shape[0], desc='Finding closest records'):
            top_overlap_idx = self.get_top_overlap_idx(i, indicators, test_records)

            # Assign the label and finalize candidate pair
            # left_gen_id = test_records.iloc[i]['nexus_id']
            for idx in top_overlap_idx:
                if idx == i:
                    continue

                right_record = test_records.iloc[idx]
                label = 1 if 0 < len(pos_matches[(pos_matches['lid'] == cur_comp_id) & (pos_matches['rid'] == right_record['id'])]) else 0
                candidate_pairs.append((record_id, right_record['id'], label, 'text_match'))

        test_candidates_df = pd.DataFrame(data=candidate_pairs, columns=['lid', 'rid', 'label', 'match_type'])

        self.save_test_candidates(test_candidates_df)

        return test_candidates_df

    def pre_cleanup(self, pairwise_matches_preds: pd.DataFrame, threshold: float = 0.999, num_of_datasources: int = 5):
        """
        Pre cleanup step before running the graph cleanup. We remove all the text_match predictions belonging to a connected component bigger than 10*number_of_datasources nodes.
        """

        matches_graph = super().pre_cleanup(pairwise_matches_preds, threshold=threshold, num_of_datasources=num_of_datasources)

        ###############################################################################################################

        # Synthetic Companies pre-cleanup: Remove all the text_match edges from subgraphs bigger than 10*number_of_datasources nodes

        ###############################################################################################################

        print('=' * 50)
        print('Starting pre-cleanup')
        print('=' * 50)

        subgraphs = list(nx.connected_components(matches_graph))

        while any([len(c) > 10 * num_of_datasources for c in subgraphs]):
            largest_subgraph = max(subgraphs, key=len)
            print('Largest subgraph size: {}, Number of subgraphs: {}'.format(len(largest_subgraph),
                                                                              len(subgraphs)))
            # Gather all subgraphs with the maximum size
            largest_subgraphs = [c for c in subgraphs if len(c) == len(largest_subgraph)]

            # Check whether the largest subgraphs are bigger than 10*number_of_datasources nodes
            if len(largest_subgraph) > 10 * num_of_datasources:
                for subgraph in largest_subgraphs:
                    # We remove all the text_match positive edges of the subgraph from the matches_graph
                    subgraph = matches_graph.subgraph(subgraph)
                    subgraph_edges = list(subgraph.edges())
                    for lid, rid in subgraph_edges:
                        if matches_graph[lid][rid]['match_type'] == 'text_match':
                            matches_graph.remove_edge(lid, rid)

            # Recompute the subgraphs
            subgraphs = list(nx.connected_components(matches_graph))
            largest_subgraph = max(subgraphs, key=len)
            largest_subgraphs = [c for c in subgraphs if len(c) == len(largest_subgraph)]

            # Check whether any of the largest subgraphs has a text_match edge, if not, we finish the pre-cleanup
            no_text_match_edges = True
            for subgraph in largest_subgraphs:
                subgraph = matches_graph.subgraph(subgraph)
                subgraph_edges = list(subgraph.edges())
                for lid, rid in subgraph_edges:
                    if matches_graph[lid][rid]['match_type'] == 'text_match':
                        no_text_match_edges = False
            
            if no_text_match_edges:
                break

        return matches_graph

class SynCompanyMatcher(CompanyMatcher):
    def __init__(self, **kwargs):
        super().__init__(id_attributes=full_data_utils.ID_ATTRIBUTES, **kwargs)

    def get_test_entity_data(self) -> pd.DataFrame:
        # Check if the test records have already been previously saved
        test_folder_path = os.path.join(self.processed_folder_path, 'test_entity_data.csv')
        if file_exists_or_create(test_folder_path):
            return pd.read_csv(test_folder_path)

        syn_data_path = os.path.join('data', 'raw', 'synthetic_data', 'seed_0')
        companies_mapping_df_path = os.path.join(syn_data_path, 'companies_master_mapping_seed_0.csv')
        mapping_df = pd.read_csv(companies_mapping_df_path)
        split = get_train_val_test_split(mapping_df)

        syn_records_file_path = os.path.join(syn_data_path, 'synthetic_records_dicts_seed_0.pkl')
        with open(syn_records_file_path, 'rb') as f:
            syn_records_dicts = pickle.load(f)
        f.close()

        syn_records_test = [dict for dict in syn_records_dicts if dict['gen_id'] in split['test']]
        test_records = []
        for dict in syn_records_test:
            test_records.append(dict['company_records'])

        companies_data_test_df = pd.concat(test_records)
        companies_data_test_df = companies_data_test_df.sort_values(['data_source_id', 'external_id'],
                                                                    ascending=[True, True])
        companies_data_test_df = companies_data_test_df.drop('gen_id', axis=1)

        # We need to set the id column for the test records according to the raw data
        companies_data_test_df = full_data_utils.add_id_to_records(companies_data_test_df,
                                                                   raw_df=self.model.dataset.get_raw_df())

        # Save the test records
        companies_data_test_df.to_csv(test_folder_path, index=False)

        return companies_data_test_df

    def blocking(self, test_entity_data: pd.DataFrame) -> pd.DataFrame:

        candidates_path = os.path.join(self.processed_folder_path, 'full_test_candidates.csv')
        if file_exists_or_create(candidates_path):
            return pd.read_csv(candidates_path)

        master_mapping_df = full_data_utils.load_syn_master_mapping(dataset_name=self.model.dataset.name)
        master_mapping_df = master_mapping_df.astype(str)

        # join the gen_id into the test_entity_data dataframe for easier lookup lateron
        test_entity_data = test_entity_data.merge(master_mapping_df.astype(
            int), left_on=['external_id', 'data_source_id'], right_on=['external_id', 'data_source_id'])

        candidate_pairs = []
        # We load the securities dataset to get id_overlap pairs
        securities_raw_file_path = dataset_raw_file_path(
            os.path.join('synthetic_data', 'seed_0', 'synthetic_securities_dataset_seed_0_size_984942_sorted.csv'))
        securities_data_df = pd.read_csv(securities_raw_file_path)

        for index, company in tqdm(test_entity_data.iterrows(), total=test_entity_data.shape[0],
                                   desc='Gathering id_overlap candidate pairs'):

            matching_securities = self.get_matching_securities(company, securities_data_df)

            if len(matching_securities) == 0:
                continue
            else:
                # Find the matching companies based on the matching securities
                company_candidate_ids = test_entity_data.merge(matching_securities[['issuer_id', 'data_source_id']],
                                                               left_on=['external_id', 'data_source_id'],
                                                               right_on=['issuer_id', 'data_source_id'])['id'].unique()

                # Assign the label and finalize candidate pair
                left_gen_id = company['gen_id']
                for idx in company_candidate_ids:
                    if idx == company['id']:
                        continue

                    right_gen_id = test_entity_data[test_entity_data['id'] == idx]['gen_id'].item()

                    label = 1 if left_gen_id == right_gen_id else 0
                    candidate_pairs.append((company['id'], idx, label, 'id_overlap'))

        indicators, tokenized_test_records = self.get_tknzd_records_and_overlap_indicators(test_entity_data)

        # Looping over the records to find the most similar (most overlapping tokens) records
        for i, (record_id, tokenized_record) in tqdm(enumerate(tokenized_test_records.iterrows()), total=tokenized_test_records.shape[0], desc='Finding closest records'):
            top_overlap_idx = self.get_top_overlap_idx(i, indicators, test_entity_data)

            # Assign the label and finalize candidate pair
            left_gen_id = test_entity_data.iloc[i]['gen_id']
            for idx in top_overlap_idx:
                if idx == i:
                    continue

                right_record = test_entity_data.iloc[idx]
                right_gen_id = right_record['gen_id']

                label = 1 if left_gen_id == right_gen_id else 0
                candidate_pairs.append((record_id, right_record['id'], label, 'text_match'))

        test_candidates_df = pd.DataFrame(data=candidate_pairs, columns=['lid', 'rid', 'label', 'match_type'])

        self.save_test_candidates(test_candidates_df)

        return test_candidates_df
    
    def pre_cleanup(self, pairwise_matches_preds: pd.DataFrame, threshold: float = 0.999, num_of_datasources: int = 5):
        """
        Pre cleanup step before running the graph cleanup. We remove all the text_match predictions belonging to a connected component bigger than 10*number_of_datasources nodes.
        """

        matches_graph = super().pre_cleanup(pairwise_matches_preds, threshold=threshold, num_of_datasources=num_of_datasources)

        ###############################################################################################################

        # Synthetic Companies pre-cleanup: Remove all the text_match edges from subgraphs bigger than 10*number_of_datasources nodes

        ###############################################################################################################

        print('=' * 50)
        print('Starting pre-cleanup')
        print('=' * 50)


        subgraphs = list(nx.connected_components(matches_graph))

        subgraphs = list(nx.connected_components(matches_graph))

        while any([len(c) > 10 * num_of_datasources for c in subgraphs]):
            largest_subgraph = max(subgraphs, key=len)
            print('Largest subgraph size: {}, Number of subgraphs: {}'.format(len(largest_subgraph),
                                                                              len(subgraphs)))
            # Gather all subgraphs with the maximum size
            largest_subgraphs = [c for c in subgraphs if len(c) == len(largest_subgraph)]

            # Check whether the largest subgraphs are bigger than 10*number_of_datasources nodes
            if len(largest_subgraph) > 10 * num_of_datasources:
                for subgraph in largest_subgraphs:
                    # We remove all the text_match positive edges of the subgraph from the matches_graph
                    subgraph = matches_graph.subgraph(subgraph)
                    subgraph_edges = list(subgraph.edges())
                    for lid, rid in subgraph_edges:
                        if matches_graph[lid][rid]['match_type'] == 'text_match':
                            matches_graph.remove_edge(lid, rid)

            # Recompute the subgraphs
            subgraphs = list(nx.connected_components(matches_graph))
            largest_subgraph = max(subgraphs, key=len)
            largest_subgraphs = [c for c in subgraphs if len(c) == len(largest_subgraph)]

            # Check whether any of the largest subgraphs has a text_match edge, if not, we finish the pre-cleanup
            no_text_match_edges = True
            for subgraph in largest_subgraphs:
                subgraph = matches_graph.subgraph(subgraph)
                subgraph_edges = list(subgraph.edges())
                for lid, rid in subgraph_edges:
                    if matches_graph[lid][rid]['match_type'] == 'text_match':
                        no_text_match_edges = False
            
            if no_text_match_edges:
                break

        return matches_graph

class SynSecurityMatcher(Matcher):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_test_entity_data(self) -> pd.DataFrame:
        # Check if the test records have already been previously saved
        test_folder_path = os.path.join(self.processed_folder_path, 'test_entity_data.csv')
        if file_exists_or_create(test_folder_path):
            return pd.read_csv(test_folder_path)

        syn_data_path = os.path.join('data', 'raw', 'synthetic_data', 'seed_0')
        companies_mapping_df_path = os.path.join(syn_data_path, 'companies_master_mapping_seed_0.csv')
        mapping_df = pd.read_csv(companies_mapping_df_path)
        companies_split = get_train_val_test_split(mapping_df)

        syn_records_file_path = os.path.join(syn_data_path, 'synthetic_records_dicts_seed_0.pkl')
        with open(syn_records_file_path, 'rb') as f:
            syn_records_dicts = pickle.load(f)
        f.close()

        syn_records_test = [dict for dict in syn_records_dicts if dict['gen_id'] in companies_split['test']]
        test_records = []
        for dict in syn_records_test:

            test_records.append(dict['security_records'])

        securities_data_df = pd.concat(test_records)
        securities_data_df = securities_data_df.sort_values(['data_source_id', 'external_id'], ascending=[True, True])
        securities_data_df = securities_data_df.drop('gen_id', axis=1)

        securities_data_df = full_data_utils.add_id_to_records(securities_data_df,
                                                               raw_df=self.model.dataset.get_raw_df())

        # Save the test records
        securities_data_df.to_csv(test_folder_path, index=False)

        return securities_data_df

    def blocking(self, test_entity_data: pd.DataFrame, company_matching: str) -> pd.DataFrame:

        candidates_path = os.path.join(self.processed_folder_path, 'full_test_candidates.csv')
        if file_exists_or_create(candidates_path):
            return pd.read_csv(candidates_path)

        master_mapping_df = full_data_utils.load_syn_master_mapping(dataset_name=self.model.dataset.name)
        candidate_pairs = []

        # check dtype of external_id and data_source_id on both dataframes
        dtypes_master_mapping_df = master_mapping_df.dtypes
        dtypes_test_entity_data = test_entity_data.dtypes
        if dtypes_master_mapping_df['external_id'] != dtypes_test_entity_data['external_id']:
            test_entity_data['external_id'] = test_entity_data['external_id'].astype(dtypes_master_mapping_df['external_id'])
        if dtypes_master_mapping_df['data_source_id'] != dtypes_test_entity_data['data_source_id']:
            test_entity_data['data_source_id'] = test_entity_data['data_source_id'].astype(dtypes_master_mapping_df['data_source_id'])

        # join the gen_id into the test_entity_data dataframe for easier lookup lateron
        test_entity_data = test_entity_data.merge(master_mapping_df, left_on=['external_id', 'data_source_id'], right_on=['external_id', 'data_source_id'])


        for index, row in tqdm(test_entity_data.iterrows(), total=test_entity_data.shape[0], desc='Gathering id_overlap candidate pairs'):
            identifiers = dict()
            for identifier in full_data_utils.ID_ATTRIBUTES.values():
                id_value = row[identifier]
                identifiers[identifier] = [] if (id_value == '' or pd.isnull(id_value)) else [id_value]

            # Check for securities that share any of the identifiers in other data sources
            matching_securities = test_entity_data[
                (
                    (test_entity_data['ISIN'].isin(identifiers['ISIN']))
                    | (test_entity_data['CUSIP'].isin(identifiers['CUSIP']))
                    | (test_entity_data['VALOR'].isin(identifiers['VALOR']))
                    | (test_entity_data['SEDOL'].isin(identifiers['SEDOL']))
                ) &
                (test_entity_data['data_source_id'] != row['data_source_id'])
            ]
            left_gen_id = row['gen_id']
            # add label
            for _, sec in matching_securities.iterrows():
                right_gen_id = sec['gen_id']
                if left_gen_id == right_gen_id:
                    label = 1
                else:
                    label = 0

                candidate_pairs.append((row['id'], sec['id'], label, 'id_overlap'))

        # Use the company_matching as a blocking for the securities, i.e. only consider securities that are
        # issued by companies that are considered as a match by the company_matching
        company_matches_df = pd.read_csv(os.path.join(dataset_results_folder_path__with_subfolders(subfolder_list=['synthetic_companies', company_matching]), 'post_graph_cleanup_matches.csv'))
        companies_df = pd.read_csv(dataset_raw_file_path(Config.DATASETS['synthetic_companies']))
        companies_df.reset_index(drop=True, inplace=True)
        companies_df.insert(0, 'id', value=companies_df.index)
        for index, row in tqdm(test_entity_data.iterrows(), total=test_entity_data.shape[0], desc = 'Gathering candidate pairs from the company_matching'):
            # Get the issuer_id, data_source_id of the current security
            issuer_id_data_source, data_source_id = row['issuer_id'], row['data_source_id']
            # Get the issuing company of the current security
            issuing_company = companies_df[(companies_df['external_id'] == issuer_id_data_source) & (companies_df['data_source_id'] == data_source_id)]
            if len(issuing_company) > 0:
                issuer_id = issuing_company['id'].item()
                # Check for all the records matched to the issuing company in the company_matching
                matching_companies_pairs = company_matches_df[(company_matches_df['lid'] == issuer_id) | (company_matches_df['rid'] == issuer_id)]
                # Get the ids of the matching companies
                matching_companies_ids = list(set(matching_companies_pairs['lid'].unique()).union(set(matching_companies_pairs['rid'].unique())).difference(set([issuer_id])))
                # Get the external_id, data_source_id of the matching companies
                matching_companies = companies_df[companies_df['id'].isin(matching_companies_ids)]
                matching_companies = matching_companies[['external_id', 'data_source_id']]
                # Get the securities issued by the matching companies for each (external_id, data_source_id) pair
                for external_id, data_source_id in list(matching_companies.values):
                    matching_securities = test_entity_data[(test_entity_data['issuer_id'] == external_id) & (test_entity_data['data_source_id'] == data_source_id)]
                    # Add the label and finalize the candidate pair
                    left_gen_id = row['gen_id']
                    for _, sec in matching_securities.iterrows():
                        right_gen_id = sec['gen_id']
                        if left_gen_id == right_gen_id:
                            label = 1
                        else:
                            label = 0
                        candidate_pairs.append((row['id'], sec['id'], label, 'company_matching'))
        test_candidates_df = pd.DataFrame(data=candidate_pairs, columns=['lid', 'rid', 'label', 'match_type'])

        self.save_test_candidates(test_candidates_df)

        return test_candidates_df


class WDCMatcher(Matcher):
    def __init__(self, number_of_candidates: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.number_of_candidates = number_of_candidates

    def get_test_entity_data(self) -> pd.DataFrame:
        # Check if the test records have already been previously saved
        test_folder_path = os.path.join(self.processed_folder_path, 'test_entity_data.csv')
        if file_exists_or_create(test_folder_path):
            return pd.read_csv(test_folder_path)

        wdc_data_path = os.path.join('data', 'raw', 'wdc_80pair')
        wdc_test_pairs_file_path = os.path.join(wdc_data_path, 'test.csv')
        self.wdc_test_pairs = pd.read_csv(wdc_test_pairs_file_path)
        wdc_records_file_path = os.path.join(wdc_data_path, 'wdc_80pair.csv')
        wdc_records = pd.read_csv(wdc_records_file_path)

        # Filter the test records 
        test_records = wdc_records[(wdc_records['id'].isin(self.wdc_test_pairs['lid'])) | (wdc_records['id'].isin(self.wdc_test_pairs['rid']))]
        
        # Save the test records
        test_records.to_csv(test_folder_path, index=False)
        return test_records

    def blocking(self, test_entity_data: pd.DataFrame) -> pd.DataFrame:
        candidates_path = os.path.join(self.processed_folder_path, 'full_test_candidates.csv')
        if file_exists_or_create(candidates_path):
            return pd.read_csv(candidates_path)
        
        # Check if self.wdc_test_pairs is already loaded
        if not hasattr(self, 'wdc_test_pairs'):
            wdc_data_path = os.path.join('data', 'raw', 'wdc_80pair')
            wdc_test_pairs_file_path = os.path.join(wdc_data_path, 'test.csv')
            self.wdc_test_pairs = pd.read_csv(wdc_test_pairs_file_path)

        candidate_pairs = []

        indicators, tokenized_test_records = self.get_tknzd_records_and_overlap_indicators(test_entity_data)

        # Looping over the records to find the most similar (most overlapping tokens) records

        for i, (record_id, tokenized_record) in tqdm(enumerate(tokenized_test_records.iterrows()), total=tokenized_test_records.shape[0],
                                                      desc='Finding closest records via token overlap'):
            top_overlap_idx = self.get_top_overlap_idx_one_source(i, indicators, test_entity_data)

            top_overlap_ids = test_entity_data.iloc[top_overlap_idx]['id'].values

            # Assign the label and finalize candidate pair. First check the pairs where record_id appears (either as lid or rid)
            record_pairs = self.wdc_test_pairs[(self.wdc_test_pairs['lid'] == record_id) | (self.wdc_test_pairs['rid'] == record_id)]
            for idx in top_overlap_ids:
                if idx == record_id:
                    continue

                # Check if the pair is in the record_pairs
                if len(record_pairs[(record_pairs['lid'] == idx) | (record_pairs['rid'] == idx)]) > 0:
                    label = 1
                else:
                    label = 0


                candidate_pairs.append((record_id, idx, label, 'text_match'))

        test_candidates_df = pd.DataFrame(data=candidate_pairs, columns=['lid', 'rid', 'label', 'match_type'])

        self.save_test_candidates(test_candidates_df)

        return test_candidates_df
    


            
