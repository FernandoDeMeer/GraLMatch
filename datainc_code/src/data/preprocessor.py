from abc import ABC, abstractmethod
import pandas as pd
import copy

import logging
from src.helpers.path_helper import *
from src.data.text_preprocessing import TextProcessor
from src.data.dataset_utils import *
from src.models.config import DEFAULT_SEED

class Preprocessor(ABC):
    def __init__(self, raw_file_path: str, dataset_name: str, seed: int=DEFAULT_SEED):
        self.raw_file_path = raw_file_path
        self.dataset_name = dataset_name
        self.seed = seed


    def get_raw_df(self):
        try:
            self.raw_df
        except AttributeError:
            df = pd.read_csv(self.raw_file_path)
            self.raw_df = self._preprocess_raw_df(df)
        return self.raw_df


    @abstractmethod
    def _preprocess_raw_df(self, df: pd.DataFrame):
        """Function that deals with making the raw_df consistent, i.e. changing 'NOTOK'
        and 'notok' to 0 and 'OK'/'ok' to 1 in the tag column"""
        raise NotImplementedError("Should be implemented in the respective subclasses.")


    def get_entity_data(self):
        try:
            self.entity_data_df
        except AttributeError:
            entity_data_file_path = dataset_processed_file_path(self.dataset_name, 'entity_data.csv', seed=self.seed)

            if file_exists_or_create(entity_data_file_path):
                self.entity_data_df = pd.read_csv(entity_data_file_path)
            else:
                raw_df = self.get_raw_df()
                df = self._get_entity_data__implementation(raw_df)

                self.entity_data_df = df
                self.entity_data_df.to_csv(entity_data_file_path, index=False)

        return self.entity_data_df


    @abstractmethod
    def _get_entity_data__implementation(self, df):
        raise NotImplementedError("Needs to be implemented on subclass.")


    def preprocess_descriptions(self, df):
        text_processor = TextProcessor(stem=False, tf_idf=True)
        # if too slow -> multiple jobs/cores
        # We first remove punctuation characters from the "name" attribute
        df.loc[pd.notnull(df['name']), 'name'] = df.loc[pd.notnull(df['name']), 'name'] \
            .swifter \
            .progress_bar(desc='[Desc. Preprocessing (1/5)] Remove punctuation chars...') \
            .apply(lambda x: text_processor.remove_punctuation_characters(x))

        # We now extract the "formerly known as" sentence from company descriptions
        df["formerly_known"] = ""
        df.loc[pd.notnull(df['description']), 'formerly_known'] = df.loc[pd.notnull(df['description']), 'description'] \
            .swifter \
            .progress_bar(desc='[Desc. Preprocessing (2/5)] Extract formerly sentences...') \
            .apply(lambda x: text_processor.extract_formerly_sentence(x))

        # We remove the "formerly known as" sentence from descriptions
        df.loc[pd.notnull(df['description']), 'description'] = df.loc[pd.notnull(df['description']), 'description'] \
            .swifter \
            .progress_bar(desc='[Desc. Preprocessing (3/5)] Remove formerly sentences from description...') \
            .apply(lambda x: text_processor.remove_formerly_sentence(x))

        # We clean the "formerly known as" sentence of punctuation + stopwords
        df.loc[pd.notnull(df['formerly_known']), 'formerly_known'] = df.loc[pd.notnull(df['formerly_known']), 'formerly_known'] \
            .swifter \
            .allow_dask_on_strings(enable=True) \
            .progress_bar(desc='[Desc. Preprocessing (4/5)] Normalize formerly known text...') \
            .apply(lambda x: text_processor.normalize_text(x))

        df.loc[pd.notnull(df['description']), 'description'] = df.loc[pd.notnull(df['description']), 'description'] \
            .swifter \
            .allow_dask_on_strings(enable=True) \
            .progress_bar(desc='[Desc. Preprocessing (5/5)] Normalize descriptions...') \
            .apply(lambda x: text_processor.normalize_text(x))

        try:
            df = text_processor.tf_idf_ordering(df=df)
        except ValueError:
            print("None of these data sources have company descriptions")
        # Make sure company descriptions are at the end of the DataFrame for truncation purposes
        new_cols = [col for col in df.columns if col != 'description'] + ['description']
        df = df[new_cols]

        return df
    
    def preprocess_syn_descriptions(self, df):
        text_processor = TextProcessor(stem=False, tf_idf=True)
        # if too slow -> multiple jobs/cores
        # We first remove punctuation characters from the "name" attribute
        df.loc[pd.notnull(df['name']), 'name'] = df.loc[pd.notnull(df['name']), 'name'] \
            .swifter \
            .progress_bar(desc='[Desc. Preprocessing (1/2)] Remove punctuation chars...') \
            .apply(lambda x: text_processor.remove_punctuation_characters(x))

        # We clean the company description of punctuation + stopwords
        df.loc[pd.notnull(df['description']), 'description'] = df.loc[pd.notnull(df['description']), 'description'] \
            .swifter \
            .allow_dask_on_strings(enable=True) \
            .progress_bar(desc='[Desc. Preprocessing (2/2)] Normalize descriptions...') \
            .apply(lambda x: text_processor.normalize_text(x))

        try:
            df = text_processor.tf_idf_ordering(df=df)
        except ValueError:
            print("None of these data sources have company descriptions")
        # Make sure company descriptions are at the end of the DataFrame for truncation purposes
        new_cols = [col for col in df.columns if col != 'description'] + ['description']
        df = df[new_cols]

        return df



    def fix_float_columns(self, df: pd.DataFrame):
        float_cols = df.select_dtypes(include=['float64']).columns  # This will select float columns only
        for column in float_cols:
            # If we apply the astype.('Int64') operation to a column with all NaNs, then we wipe the whole DataFrame
            if not df[column].isnull().all():
                df[column] = df[column].astype('Int64')
        return df
    @staticmethod
    def set_id_column(df: pd.DataFrame):

        df.reset_index(drop=True, inplace=True)

        try:
            df.insert(0, column='id', value=df.index)
        except ValueError:
            df.drop(columns=['id'], inplace= True)
            df.insert(0, column='id', value=df.index)

        df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
        return df

class DefaultBenchmarkPreprocessor(Preprocessor):
    def _preprocess_raw_df(self, df: pd.DataFrame):
        return df

    def _get_entity_data__implementation(self, df):
        self.get_entity_data()


class SynCompanyPreprocessor(Preprocessor):
    def _preprocess_raw_df(self, df: pd.DataFrame):
        
        df = self.set_id_column(df)
        return df

    def _get_entity_data__implementation(self, df):
        df = copy.deepcopy(df)
        df = df.rename(columns={'short_description': 'description'})

        # Remove label-esque columns
        # external_id and data_source_id do not contain any semantic info of the company
        df = df.drop(columns=['external_id', 'data_source_id'])

        # Remove inserted/last_modified and other metadata columns if present
        if 'last_modified' in df.columns:
            df = df.drop(columns=['last_modified'])
        if 'inserted' in df.columns:
            df = df.drop(columns=['inserted'])

        df = self.set_id_column(df)
        df = self.fix_float_columns(df)

        if 'companies' in self.dataset_name:
            df = self.preprocess_syn_descriptions(df)
        else:
            raise Exception('This Prepocessor is implemented for companies datasets ONLY.')

        return df


class SynSecurityPreprocessor(Preprocessor):
    def _preprocess_raw_df(self, df: pd.DataFrame):

        df.columns = [col.lower() for col in df.columns]
        df = df.rename(columns={'isin': 'active_isin',
                                'cusip': 'active_cusip',
                                'valor': 'active_valor',
                                'sedol': 'active_sedol'})

        df = self.fix_float_columns(df)
        df = self.set_id_column(df)
        return df

    def _get_entity_data__implementation(self, df):
        df = copy.deepcopy(df)

        # Remove label-esque columns
        # external_id and data_source_id do not contain any semantic info of the company
        df = df.drop(columns=['data_source_id'])

        # Remove inserted/last_modified and other metadata columns if present
        if 'last_modified' in df.columns:
            df = df.drop(columns=['last_modified'])
        if 'inserted' in df.columns:
            df = df.drop(columns=['inserted'])

        return df


class WDCPreprocessor(Preprocessor):
    def _preprocess_raw_df(self, df: pd.DataFrame):
        return df

    def _get_entity_data__implementation(self, df):
        df = copy.deepcopy(df)

        # Set the id column as the index
        df = df.set_index('id')

        return df
