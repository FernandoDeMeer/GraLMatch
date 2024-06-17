import os
import sys
import json

import pandas as pd

wdc_dict = {'train' : 'wdcproducts80cc20rnd000un_train_large.json.gz',
            'val' : 'wdcproducts80cc20rnd000un_valid_large.json.gz',
            'test' : 'wdcproducts80cc20rnd100un_gs.json.gz'
            }

def load_wdc_data(path): 
    return pd.read_json(path, compression='gzip', lines=True)

def process_wdc_data(pairs_df, split:str):
    ''' From https://webdatacommons.org/largescaleproductcorpus/wdc-products/
    Each row in a file represents either a pair of offers (pair-wise files) or a single offer (multi-class files). Each offer is represented by the following attributes (see example files for example values):

    id: An integer id that uniquely identifies an offer.
    brand: The brand attribute, usually a short string with a median of 1 word, e.g. "Nikon", "Crucial", ...
    title: The title attribute, a string with a median of 8 words, e.g. "Nikon AF-S NIKKOR 50mm f1.4G Lens", "Crucial 4GB (1x4GB) DDR3l PC3-12800 1600MHz SODIMM Module", ...
    description: The description attribute, a longer string with a median of 32 words.
    price: The price attribute, a string containing a number, e.g. "749", "20.57", ...
    priceCurrency: The priceCurrency attribute, usually containing a three character string, e.g. "AUD", "GBP"
    cluster_id: The integer cluster_id referring to the cluster an offer belongs to. All offers in a cluster refer to the same real-word entity.
    label: The label for the classification. This is either 1 or 0 for pair-wise matching or it corresponds to the cluster_id for multi-class matching.
    pair_id: The concatenation of ids of the right and left offer in the form "id_left#id_right". This uniquely identifies a pair of offers. This attribute only exists in the pair-wise files.
    is_hard_negative: Only exists in the pair-wise files. This attribute is boolean and signifies if a negative pair was selected using a similarity metric or randomly.
    unseen: Only exists in the multi-class files. This attribute is boolean and signifies if a certain offer has training examples in the development set.
    url: Only exists in the PDC2020 corpus file. This attribute references the webpage the product offer was extracted from.

    We want to extract from the pairs DataFrame the following columns:
    - id
    - brand
    - title
    - description
    - price
    - priceCurrency

    For all records of each pair.
    '''
    entity_data_df = pd.DataFrame(columns=['id', 'brand', 'title', 'description', 'price', 'priceCurrency'])

    for suffix in ['_left', '_right']:
        entity_data_df = pd.concat([entity_data_df, pairs_df[['id{}'.format(suffix), 
                                                         'brand{}'.format(suffix), 
                                                         'title{}'.format(suffix), 
                                                         'description{}'.format(suffix), 
                                                         'price{}'.format(suffix), 
                                                         'priceCurrency{}'.format(suffix)]].rename(columns={'id{}'.format(suffix): 'id', 
                                                                                                            'brand{}'.format(suffix): 'brand', 
                                                                                                            'title{}'.format(suffix): 'title', 
                                                                                                            'description{}'.format(suffix): 'description', 
                                                                                                            'price{}'.format(suffix): 'price', 
                                                                                                            'priceCurrency{}'.format(suffix): 'priceCurrency'})], ignore_index=True)

    # Remove duplicate rows
    entity_data_df = entity_data_df.drop_duplicates(subset='id')
            
    return entity_data_df
    
def merge_wdc_data(entity_data_df_list):
    ''' Merge the entity data DataFrames into a single DataFrame and save it as a CSV file 
    '''
    merged_df = pd.concat(entity_data_df_list)  
    merged_df.to_csv('data/raw/wdc_80pair/wdc_80pair.csv', index=False)
    print(f"Saved merged WDC data as CSV file with {len(merged_df)} examples")

def process_wdc_labels(pairs_df, split:str):
    '''
    We want to process the pairs DataFrame to make a labels DataFrame with the following columns:
    - lid
    - rid
    - label

    To have a CSV file for each split
    '''
    labels_df = pairs_df[['id_left', 'id_right', 'label']].rename(columns={'id_left': 'lid', 'id_right': 'rid'})
    labels_df.to_csv('data/raw/wdc_80pair/{}.csv'.format(split), index=False)
    print(f"Saved {split} labels as CSV file with {len(labels_df)} examples")

def main(): 
    wdc_dir = 'data/raw/wdc_80pair'
    wdc_data = {}

    for split, file in wdc_dict.items():
        wdc_data[split] = load_wdc_data(os.path.join(wdc_dir, file))
        print(f"Loaded {len(wdc_data[split])} {split} examples")

    entity_data_df_list = []

    for split, df in wdc_data.items():
        entity_data_df = process_wdc_data(df, split)
        entity_data_df_list.append(entity_data_df)
        process_wdc_labels(df, split)
        print(f"Processed {len(entity_data_df)} {split} examples")
    
    merge_wdc_data(entity_data_df_list)

if __name__ == "__main__":
    main()
