from src.data.dataset_utils import SplitMethod
from src.helpers.seed_helper import init_seed
from src.helpers.logging_helper import setup_logging
from src.data.dataset_utils import *
from src.data.security_tokenizer import SecurityTokenizer
from src.data.default_benchmark_tokenizer import DefaultBenchmarkTokenizer
from src.data.syn_company_tokenizer import SynCompanyTokenizer
from src.data.preprocessor import  SynCompanyPreprocessor, \
    DefaultBenchmarkPreprocessor, SynSecurityPreprocessor, WDCPreprocessor
from src.data.tokenizer import DataIncTokenizer
from src.models.config import Config, DEFAULT_NONMATCH_RATIO, DEFAULT_SEED, DEFAULT_SEQ_LENGTH, DEFAULT_TRAIN_FRAC
from src.helpers.path_helper import *
import copy
import os
import sys

import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod


sys.path.append(os.getcwd())


setup_logging()


# Wrapper class for our DataInc datasets. The goal is to
# have all relevant CSVs accessed through this class, so that
# we do not have to wrangle with files and paths directly,
# but rather get what we need easily.
#
# To add a new dataset, simply add the Config in models/config.py
class DataincDataset(ABC):
    # Static method to expose available datasets
    @staticmethod
    def available_datasets():
        return Config.DATASETS.keys()

    @staticmethod
    def create_instance(name: str,
                        model_name: str,
                        use_val: bool,
                        split_method: SplitMethod = SplitMethod.PRE_SPLIT,
                        seed: int = DEFAULT_SEED,
                        do_lower_case=True,
                        max_seq_length: int = DEFAULT_SEQ_LENGTH,
                        train_frac: float = DEFAULT_TRAIN_FRAC,
                        nonmatch_ratio: int = DEFAULT_NONMATCH_RATIO):

        if name == 'synthetic_companies' or name == 'synthetic_companies_small':
            return SynCompanyDataset(name=name, model_name=model_name, split_method=SplitMethod.PRE_SPLIT, seed=seed,
                                            do_lower_case=do_lower_case, max_seq_length=max_seq_length,
                                            train_frac=train_frac, nonmatch_ratio=nonmatch_ratio, use_val=use_val)
        elif name == 'synthetic_securities' or name == 'synthetic_securities_small':
            return SynSecurityDataset(name=name, model_name=model_name, split_method=SplitMethod.PRE_SPLIT, seed=seed,
                                             do_lower_case=do_lower_case, max_seq_length=max_seq_length,
                                             train_frac=train_frac, nonmatch_ratio=nonmatch_ratio, use_val=use_val)
        elif name == 'wdc':
            return WDCBenchmarkDataset(name=name, model_name=model_name, split_method=SplitMethod.PRE_SPLIT, seed=seed,
                                         do_lower_case=do_lower_case, max_seq_length=max_seq_length,
                                         train_frac=train_frac, nonmatch_ratio=nonmatch_ratio, use_val=use_val)
        else:
            raise Exception('This dataset has not been implemented.')

    def __init__(self, name: str, model_name: str, use_val: bool,
                 split_method: SplitMethod = SplitMethod.PRE_SPLIT,
                 seed: int = DEFAULT_SEED, do_lower_case=True, max_seq_length: int = DEFAULT_SEQ_LENGTH,
                 train_frac: float = DEFAULT_TRAIN_FRAC, nonmatch_ratio: int = DEFAULT_NONMATCH_RATIO):

        self.name = self._check_dataset_name(name)
        self.raw_file_path = dataset_raw_file_path(Config.DATASETS[self.name])
        self.model_name = self._check_model_name(model_name)
        self.seed = seed
        self.do_lower_case = do_lower_case
        self.max_seq_length = max_seq_length
        self.split_method = split_method
        self.use_val = use_val
        self.train_frac = train_frac
        # "X-1 to 1" ratio nonmatches to matches (for generation)
        self.nonmatch_ratio = nonmatch_ratio
        self.tokenizer = None

        # Set the seed on all libraries
        init_seed(self.seed)
        self.preprocessor = None

    def _check_dataset_name(self, name):
        configured_dataset_names = Config.DATASETS.keys()

        if name not in configured_dataset_names:
            raise ValueError(f':dataset_name {name} should be one of [{configured_dataset_names}]')
        return name

    def _check_model_name(self, model_name):
        configured_model_names = Config.MODELS.keys()

        if model_name not in configured_model_names:
            raise ValueError(f':model_name {model_name} should be one of [{configured_model_names}]')
        return model_name

    def get_split_method_name(self):
        return str(self.split_method).split('.')[1].lower()

    def get_raw_df(self):
        return self.preprocessor.get_raw_df()

    def get_entity_data(self):
        return self.preprocessor.get_entity_data()

    # Generates the tokenized data for every entity in the dataset.
    #
    def get_tokenized_data(self):
        try:
            return self.tokenized_data
        except AttributeError:
            tokenized_file_path = dataset_processed_file_path(self.name, 'tokenized_data__' + self.model_name + '.json',
                                                              seed=self.seed)

            if file_exists_or_create(tokenized_file_path):
                self.tokenized_data = pd.read_json(tokenized_file_path)

            else:
                self.tokenized_data, _ = self.tokenizer.tokenize_df(self.get_entity_data())
                self.tokenized_data.to_json(tokenized_file_path)

        return self.tokenized_data

    # Generates all known positive matches from the raw data
    def get_matches(self):
        try:
            self.matches_df
        except AttributeError:
            processed_matches_path = dataset_processed_file_path(self.name, 'pos_matches.csv', seed=self.seed)

            if file_exists_or_create(processed_matches_path):
                self.matches_df = pd.read_csv(processed_matches_path)
            else:
                self.matches_df = self.get_matches__implementation()
                self.matches_df.to_csv(processed_matches_path, index=False)

        return self.matches_df

    @abstractmethod
    def get_matches__implementation(self):
        raise NotImplementedError("Needs to be implemented on subclass.")

    # randomly assigning matches to train/test
    #
    def _random_split(self):
        def split_fn(df: pd.DataFrame, train_frac: float):
            train_df = df.sample(frac=train_frac, random_state=self.seed)
            test_df = df.drop(train_df.index)
            val_df = pd.DataFrame()
            if self.use_val:
                # split the validation set as half of the test set, i.e.
                # both test and valid sets will be of the same size
                #
                val_df = test_df.sample(frac=0.5, random_state=self.seed)
                test_df = test_df.drop(val_df.index)
            return train_df, test_df, val_df

        return split_fn


    def pre_split(self):
        raise NotImplementedError("Needs to be implemented on subclass.")

    # Separates the matches into train and test
    #
    def _get_train_test_val_given_matches(self, train_frac: float):
        if self.split_method == SplitMethod.RANDOM:
            split_fn = self._random_split()
        elif self.split_method == SplitMethod.PRE_SPLIT:
            split_fn = self.pre_split()
        else:
            raise NotImplementedError(
                f"Split method '{self.split_method}' not implemented. \
                Make sure to include the seed when implementing a new one.")

        try:
            if self.use_val:
                return self.train_given, self.test_given, self.val_given
            else:
                return self.train_given, self.test_given, pd.DataFrame()
        except AttributeError:
            method_name = self.get_split_method_name()
            train_file_path = dataset_processed_file_path(self.name, f'train__{method_name}__given_matches.csv',
                                                          seed=self.seed)
            test_file_path = dataset_processed_file_path(self.name, f'test__{method_name}__given_matches.csv',
                                                         seed=self.seed)
            validation_file_path = dataset_processed_file_path(self.name, f'val__{method_name}__given_matches.csv',
                                                         seed=self.seed)

            check_val = file_exists_or_create(validation_file_path) if self.use_val else True

            if file_exists_or_create(train_file_path) and file_exists_or_create(train_file_path) and check_val:
                self.train_given = pd.read_csv(train_file_path)
                self.test_given = pd.read_csv(test_file_path)
                self.validation_given = pd.read_csv(validation_file_path) if self.use_val else None
            else:
                matches = self.get_matches()
                self.train_given, self.test_given, self.validation_given = split_fn(matches, train_frac)
                self.train_given.to_csv(train_file_path, index=False)
                self.test_given.to_csv(test_file_path, index=False)
                if not self.validation_given.empty:
                    self.validation_given.to_csv(validation_file_path, index=False)

        return self.train_given, self.test_given, self.validation_given

    def get_train_test_val(self):
        try:
            return self.train_df, self.test_df, self.validation_df
        except AttributeError:
            method_name = self.get_split_method_name()
            train_file_path = dataset_processed_file_path(self.name, f'train__{method_name}__all_matches.csv',
                                                          seed=self.seed)
            test_file_path = dataset_processed_file_path(self.name, f'test__{method_name}__all_matches.csv',
                                                         seed=self.seed)
            validation_file_path = dataset_processed_file_path(self.name, f'val__{method_name}__all_matches.csv',
                                                               seed=self.seed)

            check_val = file_exists_or_create(validation_file_path) if self.use_val else True

            if file_exists_or_create(train_file_path) and file_exists_or_create(train_file_path) and check_val:
                self.train_df = pd.read_csv(train_file_path)
                self.test_df = pd.read_csv(test_file_path)
                self.validation_df = pd.read_csv(validation_file_path) if self.use_val else pd.DataFrame()
            else:
                # prebuild given matches self.train_given / self.test_given
                train_given, test_given, validation_given = \
                    self._get_train_test_val_given_matches(train_frac=self.train_frac)

                self.train_df, self.test_df, self.validation_df = \
                    self.get_train_test_val__implementation(train_given, test_given, validation_given)

                self.train_df.to_csv(train_file_path, index=False)
                self.test_df.to_csv(test_file_path, index=False)
                if not self.validation_df.empty:
                    self.validation_df.to_csv(validation_file_path, index=False)

        return self.train_df, self.test_df, self.validation_df

    def get_validation(self):
        """
        reads the validation file if it exists
        does NOT create one if it does not exist yet, unlike train/test set
        """
        try:
            return self.validation_df
        except AttributeError:
            self.validation_df = None
            validation_file_path = dataset_processed_file_path(self.name, 'validation.csv', seed=self.seed)

            if file_exists_or_create(validation_file_path):
                self.validation_df = pd.read_csv(validation_file_path)


            return self.validation_df

    @abstractmethod
    def get_train_test_val__implementation(self, train_given, test_given, val_given):
        # return train_df, test_df, val_df
        raise NotImplementedError("Should be implemented in the respective subclasses.")

    # returns the PyTorch dataloaders ready for use
    def get_data_loaders(self, batch_size: int = 8):
        train_df, test_df, validation_df = self.get_train_test_val()

        train_ds = PytorchDataset(model_name=self.name, idx_df=train_df, data_df=self.get_tokenized_data(),
                                  tokenizer=self.tokenizer, max_seq_length=self.max_seq_length)
        test_ds = PytorchDataset(model_name=self.name, idx_df=test_df, data_df=self.get_tokenized_data(),
                                 tokenizer=self.tokenizer, max_seq_length=self.max_seq_length)

        train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
        test_dl = DataLoader(test_ds, shuffle=False, batch_size=batch_size)

        if validation_df.empty:
            val_dl = None
        else:
            val_ds = PytorchDataset(model_name=self.name, idx_df=validation_df, data_df=self.get_tokenized_data(),
                                 tokenizer=self.tokenizer, max_seq_length=self.max_seq_length)
            val_dl = DataLoader(val_ds, shuffle=False, batch_size=batch_size)

        return train_dl, test_dl, val_dl


class DittoBenchmarkDataset(DataincDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = DefaultBenchmarkTokenizer(self.model_name, self.do_lower_case, self.max_seq_length)
        self.preprocessor = DefaultBenchmarkPreprocessor(self.raw_file_path, self.name, seed=self.seed)
        self.col_names = []

    def __create_entity_data(self, filepath, dataset=1):
        def split_to_cols(input):
            return re.split("COL{1} {1}\w* VAL{1} {1}", input)

        def get_col_names(input):
            return re.findall("(?<!COL)(\w+)(?=\s+VAL)", input)

        data_dict = {}

        with open(filepath) as file:
            for i, line in enumerate(file):
                # first line only
                if not i:
                    # remove duplicates and keep order
                    self.col_names = list(dict.fromkeys(get_col_names(line)))
                    data_dict = {col_name: [] for col_name in self.col_names}
                    data_dict['dataset'] = dataset
                split_list = line.split("\t")
                # disregard the last one, as it is the label
                for row in split_list[:-1]:
                    col_list = split_to_cols(row)[1:]
                    for i, col in enumerate(self.col_names):
                        data_dict[col].append(col_list[i].strip())

        return pd.DataFrame(data_dict)

    def __create_df_from_txt_file(self, filepath) -> pd.DataFrame:
        def remove_formatting(txt_input):
            return re.sub("COL{1} {1}\w* VAL{1} {1}", '', txt_input).strip()

        l_txt = []
        r_txt = []
        labels = []
        with open(filepath) as file:
            for line in file:
                split_list = line.split("\t")
                l_txt.append(remove_formatting(split_list[0]))
                r_txt.append(remove_formatting(split_list[1]))
                labels.append(int(split_list[2]))

        return pd.DataFrame({'lid': l_txt, 'rid': r_txt, 'label': labels})

    def __replace_txt_with_ids(self, entity_data, df_txt, dataset=1) -> pd.DataFrame:
        df = df_txt.copy(deep=True)
        entity_set = entity_data[entity_data['dataset'] == dataset]
        for i in entity_set.index:
            curr = ' '.join(f'{entity_set[col][i]}' for col in self.col_names).strip()
            df = df.replace(curr, i)

        return df

    def get_train_test_val__implementation(self, train_given, test_given, val_given):
        train_file_path = dataset_raw_file_path(os.path.join('ditto_files', self.name, 'train.txt'))
        test_file_path = dataset_raw_file_path(os.path.join('ditto_files', self.name, 'test.txt'))
        val_file_path = dataset_raw_file_path(os.path.join('ditto_files', self.name, 'valid.txt'))

        if file_exists_or_create(train_file_path) and file_exists_or_create(test_file_path):
            entity_train = self.__create_entity_data(filepath=train_file_path, dataset=0)
            entity_test = self.__create_entity_data(filepath=test_file_path, dataset=1)

            entity_val = pd.DataFrame()
            # separate because the val dataset is not mandatory for training
            if file_exists_or_create(val_file_path):
                entity_val = self.__create_entity_data(filepath=val_file_path, dataset=2)

            entity_data = pd.concat([entity_train, entity_test, entity_val])
            entity_data = entity_data.drop_duplicates()
            entity_data.reset_index(inplace=True, drop=True)

            df_train_txt = self.__create_df_from_txt_file(filepath=train_file_path)
            df_test_txt = self.__create_df_from_txt_file(filepath=test_file_path)

            df_train = self.__replace_txt_with_ids(entity_data=entity_data, df_txt=df_train_txt, dataset=0)
            df_test = self.__replace_txt_with_ids(entity_data=entity_data, df_txt=df_test_txt, dataset=1)

            entity_file_path = \
                dataset_processed_file_path(dataset_name=self.name, file_name='entity_data.csv', seed=self.seed)
            file_exists_or_create(entity_file_path)

            df_val = None
            if not entity_val.empty:
                df_val_txt = self.__create_df_from_txt_file(filepath=val_file_path)
                df_val = self.__replace_txt_with_ids(entity_data=entity_data, df_txt=df_val_txt, dataset=2)
                df_val.to_csv(dataset_processed_file_path(dataset_name=self.name,
                                                          file_name='validation.csv',
                                                          seed=self.seed),
                              index=False)

            entity_data.drop(columns=['dataset'], inplace=True)
            entity_data.insert(0, 'id', entity_data.index)

            entity_data.to_csv(entity_file_path, index=False)

            return df_train, df_test, df_val
        else:
            raise FileNotFoundError(f'{self.name} benchmark files could not be found at '
                                    f'{train_file_path} and {test_file_path}')

    def get_matches__implementation(self):
        """
        for this class we do not need the file pos_matches, hence we return an empty df
        """
        return pd.DataFrame()

    def pre_split(self):
        """
        for this class we do not need the files train_given and test_given, hence we return two empty dfs
        """
        def get_empty_dfs(matches, train_frac):
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        return get_empty_dfs


class WDCBenchmarkDataset(DataincDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = DefaultBenchmarkTokenizer(self.model_name, self.do_lower_case, self.max_seq_length)
        self.preprocessor = WDCPreprocessor(self.raw_file_path, self.name, seed=self.seed)

    def get_matches__implementation(self):
        """
        For this class we do not need the file pos_matches, hence we return an empty df
        """
        return pd.DataFrame()
        
    
    def get_train_test_val__implementation(self, train_given, test_given, val_given):
        """
        In this class we do not need to add additional negative training samples (they're already in the benchmark)
        """
        return train_given, test_given, val_given
        
    
    def pre_split(self):
        def get_pre_split_train_test_val(matches, train_frac):
            pos_train_matches_path = dataset_raw_file_path(os.path.join('wdc_80pair', 'train.csv'))
            pos_train_matches_df = pd.read_csv(pos_train_matches_path)

            pos_val_matches_path = dataset_raw_file_path(os.path.join('wdc_80pair', 'val.csv'))
            pos_val_matches_df = pd.read_csv(pos_val_matches_path)

            pos_test_matches_path = dataset_raw_file_path(os.path.join('wdc_80pair', 'test.csv'))
            pos_test_matches_df = pd.read_csv(pos_test_matches_path)

            return pos_train_matches_df, pos_test_matches_df, pos_val_matches_df
        
        return get_pre_split_train_test_val
    


class SynBaseDataset(DataincDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.full_ds = False

        if self.name == 'synthetic_companies' or self.name == 'synthetic_companies_small':
            self.ds_type = 'companies'
            if self.name == 'synthetic_companies':
                self.full_ds = True
        elif self.name == 'synthetic_securities' or self.name == 'synthetic_securities_small':
            self.ds_type = 'securities'
            if self.name == 'synthetic_securities':
                self.full_ds = True
        else:
            raise ValueError(f'{self.name} does not have a supported synthetic dataset type.')

    def pre_split(self):
        def get_pre_split_train_test_val(matches, train_frac):
            pos_val_matches_path = dataset_raw_file_path(os.path.join('synthetic_data', 'seed_0', self.ds_type,
                                                                      f'{"" if self.full_ds  else "filtered_"}val.csv'))
            pos_val_matches_df = pd.read_csv(pos_val_matches_path, index_col= 0)

            pos_train_matches_path = dataset_raw_file_path(os.path.join('synthetic_data', 'seed_0', self.ds_type,
                                                                        f'{"" if self.full_ds  else "filtered_"}train.csv'))
            pos_train_matches_df = pd.read_csv(pos_train_matches_path, index_col=0)

            pos_test_matches_path = dataset_raw_file_path(os.path.join('synthetic_data', 'seed_0', self.ds_type,
                                                                       'test.csv'))
            pos_test_matches_df = pd.read_csv(pos_test_matches_path, index_col=0)

            return pos_train_matches_df, pos_test_matches_df, pos_val_matches_df
        return get_pre_split_train_test_val

    def get_matches__implementation(self):
        """
        for this class we do not need the file pos_matches, hence we return an empty df
        """
        return pd.DataFrame()

    def get_train_test_val__implementation(self, train_given, test_given, val_given):
        test_df, train_df, val_df = add_random_non_matches_train_val_test(train_given=train_given,
                                      test_given=test_given,
                                      val_given=val_given,
                                      nonmatch_ratio=self.nonmatch_ratio,
                                      name=self.name)
        return train_df, test_df, val_df


class SynCompanyDataset(SynBaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = SynCompanyTokenizer(self.model_name, self.do_lower_case, self.max_seq_length)
        self.preprocessor = SynCompanyPreprocessor(self.raw_file_path, self.name, seed=self.seed)


class SynSecurityDataset(SynBaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = SecurityTokenizer(self.model_name, self.do_lower_case, self.max_seq_length)
        self.preprocessor = SynSecurityPreprocessor(self.raw_file_path, self.name, seed=self.seed)


class PytorchDataset(Dataset):
    def __init__(self, model_name: str, idx_df: pd.DataFrame, data_df: pd.DataFrame, tokenizer: DataIncTokenizer,
                 max_seq_length: int):
        self.model_name = model_name
        self.idx_df = idx_df
        self.data_df = data_df
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # self.label_list = sorted(self.idx_df.label.unique())

    def __len__(self):
        return len(self.idx_df)

    def __getitem__(self, idx):
        row = self.idx_df.iloc[idx]

        l_txt = copy.deepcopy(self.data_df.loc[row['lid'], 'tokenized'])
        r_txt = copy.deepcopy(self.data_df.loc[row['rid'], 'tokenized'])
        label = row['label']

        seq = self.tokenizer.generate_sample(l_txt, r_txt, label)

        # Also return the initial IDs for easier logging
        raw_batch = (
            torch.tensor([row['lid']], dtype=torch.long),
            torch.tensor([row['rid']], dtype=torch.long),
        )

        return seq + raw_batch
