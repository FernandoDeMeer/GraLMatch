import os
import sys
import pickle
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from tqdm.auto import tqdm


def filter_pairs_no_data_drift(pairs_df, value_df, map_df, syn_records_dict, pairs_to_filter):
    filtered_pairs = [[],[],[]]

    for index, pair in tqdm(pairs_df[:pairs_to_filter].iterrows(), total= pairs_to_filter,
                            desc='Filtering out pairs with corporate acquisitions'):
        record1 = value_df.iloc[pair['lid']]
        pair_external_id = record1['external_id']
        pair_data_source_id = record1['data_source_id']
        pair_gen_id = map_df[(map_df['external_id'] == pair_external_id) &
                             (map_df['data_source_id'] == pair_data_source_id)]['gen_id'].values[0]
        syn_dict = syn_records_dict[pair_gen_id]
        if not 'create_corporate_acquisition_artifact' in syn_dict['data_artifacts']['applied_DataDriftDataArtifacts'] \
        and not 'no_id_overlap_multi_security_artifact' in syn_dict['data_artifacts']['applied_MultiSecurityDataArtifacts']:
            filtered_pairs[0].append(pair['lid'])
            filtered_pairs[1].append(pair['rid'])
            filtered_pairs[2].append(pair['label'])

    filtered_pairs_df = pd.DataFrame({'lid': filtered_pairs[0], 'rid': filtered_pairs[1], 'label': filtered_pairs[2]})

    return filtered_pairs_df
def main():
    # We need to select a subset of "manually checked" pairs for training and validating a model. To do this,
    # we discard the pairs coming from a gen_id group with a "corporate_acquisition" data_drift event among
    # the first N gen_id groups.

    syn_data_path = os.path.join('data','raw','synthetic_data','seed_0')

    companies_df_path = os.path.join(syn_data_path, 'synthetic_companies_dataset_seed_0_size_868254_sorted.csv')
    companies_df = pd.read_csv(companies_df_path, low_memory=False)
    companies_mapping_df_path = os.path.join(syn_data_path,'companies_master_mapping_seed_0.csv')
    companies_mapping_df = pd.read_csv(companies_mapping_df_path)

    syn_records_file_path = os.path.join(syn_data_path,
                                         'synthetic_records_dicts_seed_0.pkl')
    with open(syn_records_file_path, 'rb') as f:
        syn_records_dicts = pickle.load(f)
    f.close()

    syn_pairs_data_path = os.path.join(syn_data_path, 'companies')
    syn_pairs_train = pd.read_csv(os.path.join(syn_pairs_data_path, 'train.csv'),index_col=0)
    syn_pairs_val = pd.read_csv(os.path.join(syn_pairs_data_path, 'val.csv'),index_col=0)

    checked_train_pairs = filter_pairs_no_data_drift(syn_pairs_train, companies_df, companies_mapping_df, syn_records_dicts, 10000)
    checked_train_pairs.to_csv(os.path.join(syn_pairs_data_path, 'filtered_train.csv'))

    checked_val_pairs = filter_pairs_no_data_drift(syn_pairs_val, companies_df, companies_mapping_df, syn_records_dicts, 5000)
    checked_val_pairs.to_csv(os.path.join(syn_pairs_data_path, 'filtered_val.csv'))

if __name__ == '__main__':
    main()
