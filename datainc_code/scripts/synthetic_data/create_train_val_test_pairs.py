import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import itertools
import re
import argparse


def add_id_column(value_df,
                  new_value_df_name: str,
                  save_df: bool):

    value_df.reset_index(inplace=True)
    value_df.rename(columns={"index":'id'}, inplace=True)

    if save_df:
        value_df.to_csv(new_value_df_name, index=False)

    return  value_df

def get_train_val_test_split(mapping_df, **kwargs):

    number_of_gen_ids = kwargs.get('number_of_gen_ids', None)
    if number_of_gen_ids:
        gen_ids = mapping_df['gen_id'].unique()[:number_of_gen_ids]
    else:
        gen_ids = mapping_df['gen_id'].unique()

    train, val, test = np.split(gen_ids, [int(.6 * len(gen_ids)), int(.8 * len(gen_ids))])

    split_dict = {'train': train,
                  'val': val,
                  'test': test}
    return split_dict

def get_securities_split_from_comp_split(comp_split, comp_mapping, security_mapping, security_value):

    train_comp_mapping = comp_mapping[comp_mapping['gen_id'].isin(comp_split['train'])]
    val_comp_mapping = comp_mapping[comp_mapping['gen_id'].isin(comp_split['val'])]
    test_comp_mapping = comp_mapping[comp_mapping['gen_id'].isin(comp_split['test'])]

    def get_sec_from_issuer_ids(issuer_ids, security_value, security_mapping):
        issuer_ids = issuer_ids.astype(str)
        security_mapping = security_mapping.astype(str)
        issuer_ids_w_securities = issuer_ids.merge(right= security_value[['external_id', 'issuer_id', 'data_source_id']].astype(str),
                                                   left_on= ['external_id', 'data_source_id'],
                                                   how= 'left', right_on= ['issuer_id', 'data_source_id'])

        issuer_ids_w_securities = issuer_ids_w_securities.merge(right = security_mapping,
                                                                left_on=['external_id_y', 'data_source_id'],
                                                                how= 'left', right_on=['external_id', 'data_source_id'])
        issuer_ids_w_securities = issuer_ids_w_securities.dropna(axis= 0, how= 'any')
        gen_ids = issuer_ids_w_securities['gen_id_y'].unique()
        return list(gen_ids)

    train = get_sec_from_issuer_ids(train_comp_mapping, security_value, security_mapping)
    val = get_sec_from_issuer_ids(val_comp_mapping, security_value, security_mapping)
    test = get_sec_from_issuer_ids(test_comp_mapping, security_value, security_mapping)

    split_dict = {'train': train,
                'val': val,
                'test': test}

    return split_dict



def create_train_val_test_sets(value_df, mapping_df, split_dict):

    def get_pairs_df(value_df,mapping_df, split, split_name):

        pairs_list = [[], [], []]

        for gen_id in tqdm(split, total= len(split),leave=True, colour='green',
                                           desc='Gathering {} labelled pairs'.format(split_name)) :
            records_ids = mapping_df[mapping_df['gen_id'] == gen_id]
            group = []
            for idx, external_and_data_source_id in records_ids.iterrows():
                group.append(value_df[(value_df['external_id'] == external_and_data_source_id['external_id']) &
                         (value_df['data_source_id'] == external_and_data_source_id['data_source_id'])]['id'].values[0])

            combs = list(itertools.combinations(list(group), 2))

            for comb in combs:
                pairs_list[0].append(comb[0])
                pairs_list[1].append(comb[1])
                pairs_list[2].append(1)

        pairs_df = pd.DataFrame({'lid': pairs_list[0], 'rid': pairs_list[1], 'label': pairs_list[2]})
        return pairs_df
    split_dfs_dict = {}
    for split_name, split in split_dict.items():
        split_dfs_dict[split_name] = get_pairs_df(value_df,
                                                          mapping_df,
                                                          split,
                                                          split_name)
    return split_dfs_dict

def save_split_dfs(syn_data_path, split_dfs_dict, dataset_name):
    folder_path = os.path.join(syn_data_path, dataset_name)
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    for split_name, split_df in split_dfs_dict.items():
        split_df.to_csv(os.path.join(folder_path, split_name + '.csv'))

def main(args):
    syn_data_path = os.path.join('data','raw','synthetic_data','seed_0')

    companies_df_path = os.path.join(syn_data_path, 'synthetic_companies_dataset_seed_0_size_868254_sorted.csv')
    companies_df = pd.read_csv(companies_df_path)
    companies_mapping_df_path = os.path.join(syn_data_path,'companies_master_mapping_seed_0.csv')
    mapping_df = pd.read_csv(companies_mapping_df_path)

    value_df = add_id_column(value_df = companies_df,
                            new_value_df_name='synthetic_companies_dataset_seed_0_with_ids.csv',
                            save_df= False)
    split = get_train_val_test_split(mapping_df)

    if args.dataset_name == 'securities':
        securities_df_path = os.path.join(syn_data_path, 'synthetic_securities_dataset_seed_0_size_984942_sorted.csv')
        securities_df = pd.read_csv(securities_df_path)
        securities_mapping_df_path = os.path.join(syn_data_path,'securities_master_mapping_seed_0.csv')
        securities_mapping_df = pd.read_csv(securities_mapping_df_path)


        securities_df = add_id_column(value_df = securities_df,
                                    new_value_df_name='synthetic_securities_dataset_seed_0_with_ids.csv',
                                    save_df= False)

        split = get_securities_split_from_comp_split(comp_split=split, comp_mapping=mapping_df,
                                                     security_mapping=securities_mapping_df, security_value=securities_df)

        mapping_df = securities_mapping_df
        value_df = securities_df




    split_dfs = create_train_val_test_sets(value_df=value_df, mapping_df=mapping_df,
                                                     split_dict=split)

    save_split_dfs(syn_data_path, split_dfs, args.dataset_name)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run splitting with following arguments')

    parser.add_argument('--dataset_name', default='companies',
                        choices=['companies', 'securities'],
                        help='Choose a dataset to be processed.')

    args = parser.parse_args()
    main(args)
