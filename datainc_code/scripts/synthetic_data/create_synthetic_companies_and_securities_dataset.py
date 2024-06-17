import os
import sys
import pickle
import networkx as nx
import numpy as np
sys.path.append(os.getcwd())
from src.helpers.path_helper import *

from src.data.synthetic_data_generation.synthetic_companies_generator import generate_companies_dataset
from src.data.synthetic_data_generation.synthetic_securities_generator import generate_securities_dataset
from src.data.synthetic_data_generation.utils import *


def generate_datasets(seed):
    random.seed(seed)
    np.random.seed(seed)
    synthetic_records_dicts_list = generate_companies_dataset(seed=seed, save_results=False)
    synthetic_records_dicts_list = generate_securities_dataset(
        synthetic_records_dicts_list=synthetic_records_dicts_list,
        seed=seed, save_results=False)
    return synthetic_records_dicts_list


def get_all_edges(list):
    edgelist = []
    for node in list:
        edges = [(node, other_node) for other_node in list if node != other_node]
        for edge in edges:
            edgelist.append(edge)
    return edgelist


def find_no_overlap_groups(syn_records_dicts_list: list):
    """ We may have groups of securities with no id overlaps between securities records, induced by the
    MissingAttributeArtifactSingleSecurity, that have not been logged as a "no_id_overlap_group". We detect and log
    them here.

    """

    id_attributes = ['ISIN', 'CUSIP', 'VALOR', 'SEDOL']

    for records_dict in syn_records_dicts_list:
        if 'no_id_overlap_multi_security_artifact' not in records_dict['data_artifacts'][
            'applied_MultiSecurityDataArtifacts']:
            for name, security_records in records_dict['security_records'].groupby('issuer_id'):
                edgelist = []
                for id_attribute in id_attributes:
                    unique_values = security_records[id_attribute].unique()
                    for unique_id_value in unique_values:
                        if unique_id_value != '' and not pd.isna(unique_id_value):
                            appearing_idxs = security_records.index[
                                security_records[id_attribute] == unique_id_value].tolist()
                            if len(appearing_idxs) > 1:
                                edges = get_all_edges(appearing_idxs)
                                for edge in edges:
                                    edgelist.append(edge)

                group_graph = nx.from_edgelist(edgelist=edgelist)
                connected_components = [len(c) for c in
                                        sorted(nx.connected_components(group_graph), key=len, reverse=True)]

                if len(connected_components) > 1:
                    records_dict['data_artifacts']['applied_MultiSecurityDataArtifacts'].append(
                        'no_id_overlap_multi_security_artifact')
    return syn_records_dicts_list


def get_master_mappings(syn_records_dicts_list):
    def get_ids(syn_records_dict, id_type, id_names_list):
        ids = []
        for id_name in id_names_list:
            ids.append(syn_records_dict[id_type][id_name].astype(str))
        return ids

    companies_ids_list = []
    securities_ids_list = []

    for syn_records_dict in syn_records_dicts_list:
        company_ids = get_ids(syn_records_dict, 'company_records', ['external_id', 'data_source_id', 'gen_id'])
        security_ids = get_ids(syn_records_dict, 'security_records', ['external_id', 'data_source_id', 'gen_id'])

        company_ids_df = pd.concat(company_ids, axis=1).astype(str)
        security_ids_df = pd.concat(security_ids, axis=1).astype(str)

        companies_ids_list.append(company_ids_df)
        securities_ids_list.append(security_ids_df)

    companies_master_mapping = pd.concat(companies_ids_list).astype(str)
    securities_master_mapping = pd.concat(securities_ids_list).astype(str)

    return companies_master_mapping, securities_master_mapping


def drop_gen_ids_sort_and_save(synthetic_companies_dataset_df, synthetic_securities_dataset_df,
                               synthetic_dataset_result_path):
    synthetic_companies_dataset_df = synthetic_companies_dataset_df.drop('gen_id', axis=1)
    synthetic_securities_dataset_df = synthetic_securities_dataset_df.drop('gen_id', axis=1)

    synthetic_companies_dataset_df = synthetic_companies_dataset_df.sort_values(['data_source_id', 'external_id'], ascending=[True, True])
    synthetic_securities_dataset_df = synthetic_securities_dataset_df.sort_values(['data_source_id', 'issuer_id', 'external_id'], ascending=[True, True, True])

    synthetic_companies_dataset_df.to_csv(os.path.join(synthetic_dataset_result_path,
                                                       'synthetic_companies_dataset_seed_{}_size_{}_sorted.csv'.format(
                                                           seed,
                                                           len(synthetic_companies_dataset_df))), index= False)
    synthetic_securities_dataset_df.to_csv(os.path.join(synthetic_dataset_result_path,
                                                        'synthetic_securities_dataset_seed_{}_size_{}_sorted.csv'.format(
                                                            seed,
                                                            len(synthetic_securities_dataset_df))), index= False)


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    synthetic_dataset_result_path = dataset_results_file_path__with_subfolders(['synthetic_data', 'seed_{}'.format(seed)], '')

    synthetic_records_dicts_list = generate_datasets(seed=seed)

    synthetic_records_dicts_list = find_no_overlap_groups(synthetic_records_dicts_list)

    synthetic_records_dicts_file_path = os.path.join(synthetic_dataset_result_path,
                                                     'synthetic_records_dicts_seed_{}.pkl'.format(seed))

    file_exists_or_create(synthetic_records_dicts_file_path)
    with open(synthetic_records_dicts_file_path, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(synthetic_records_dicts_list, outp, pickle.HIGHEST_PROTOCOL)
    outp.close()

    companies_master_mapping, securities_master_mapping = get_master_mappings(synthetic_records_dicts_list)

    companies_master_mapping_file_path = os.path.join(synthetic_dataset_result_path,
                                                      'companies_master_mapping_seed_{}.csv'.format(seed))
    securities_master_mapping_file_path = os.path.join(synthetic_dataset_result_path,
                                                       'securities_master_mapping_seed_{}.csv'.format(seed))

    companies_master_mapping.to_csv(companies_master_mapping_file_path, index=False)
    securities_master_mapping.to_csv(securities_master_mapping_file_path, index=False)

    synthetic_companies_dataset_df = build_df_from_records(synthetic_records_dicts_list, 'company_records')
    synthetic_securities_dataset_df = build_df_from_records(synthetic_records_dicts_list, 'security_records')

    # synthetic_companies_dataset_df.to_csv(os.path.join(synthetic_dataset_result_path,
    #                                                    'synthetic_companies_dataset_seed_{}_size_{}_unshuffled.csv'.format(
    #                                                        seed,
    #                                                        len(synthetic_companies_dataset_df))))
    # synthetic_securities_dataset_df.to_csv(os.path.join(synthetic_dataset_result_path,
    #                                                     'synthetic_securities_dataset_seed_{}_size_{}_unshuffled.csv'.format(
    #                                                         seed,
    #                                                         len(synthetic_securities_dataset_df))))

    drop_gen_ids_sort_and_save(synthetic_companies_dataset_df, synthetic_securities_dataset_df,
                               synthetic_dataset_result_path)


    # syn_records_file_path = os.path.join(synthetic_dataset_result_path,
    #                                      'synthetic_records_dicts_seed_0_.pkl')
    # with open(syn_records_file_path, 'rb') as f:
    #     syn_records = pickle.load(f)
    # f.close()
