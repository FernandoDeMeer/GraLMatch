import copy
import random

import pandas as pd
from tqdm.auto import tqdm

from src.data.synthetic_data_generation.data_artifacts_companies import *
from src.data.synthetic_data_generation.utils import build_df_from_records


class SyntheticCompanyDataSource():

    def __init__(self, data_source_name: str, data_source_id: int, single_record_artifacts_params: dict):

        self.name = data_source_name
        self.data_source_id = str(data_source_id)
        self.artifacts_params = single_record_artifacts_params

class SyntheticCompaniesGenerator():

    def __init__(self, real_data_df: pd.DataFrame, max_company_seeds: int, data_sources_list: list,
                 multi_record_artifacts_params: dict, data_drift_artifacts_params: dict,
                 batch_data_artifacts_params: dict):

        self.real_data_df = real_data_df
        self.companies_data_df = real_data_df[real_data_df['primary_role'] == 'company'][:max_company_seeds][
            ['name', 'city', 'region', 'country_code', 'short_description']].reset_index(drop=True)
        self.investors_data_df = real_data_df[real_data_df['primary_role'] == 'investor'][
            ['name', 'city', 'region', 'country_code', 'short_description']].reset_index(drop=True)
        self.data_sources_list = data_sources_list
        self.last_gen_id = 0
        self.external_ids = {data_source.data_source_id: None for data_source in self.data_sources_list}
        self.multi_record_artifacts = get_artifacts(multi_record_artifacts_params, MultiCompanyDataArtifact)
        self.data_drift_artifacts = get_artifacts(data_drift_artifacts_params, DataDriftDataArtifact)
        self.batch_data_artifacts = get_artifacts(batch_data_artifacts_params, BatchCompanyDataArtifact)
    def _init_syn_record(self, company_record: pd.Series):
        synthetic_records_dict = {}
        new_gen_id = self.get_new_gen_id()
        synthetic_records_dict['gen_id'] = new_gen_id
        synthetic_records_dict['data_drift_dicts'] = []
        synthetic_records_dict['company_seed'] = company_record
        synthetic_records_dict['data_artifacts'] = {}
        synthetic_records_dict['timestamp'] = random_date(start="1/1/1990 0:0:0", end="31/12/2025 23:59:59",
                                                          prop=random.uniform(0, 1))
        company_records = []
        for data_source in self.data_sources_list:
            data_source_company_record = copy.deepcopy(company_record)
            data_source_company_record['data_source_id'] = data_source.data_source_id
            timestamp = random_close_date(synthetic_records_dict['timestamp'])
            data_source_company_record['inserted'] = timestamp
            data_source_company_record['last_modified'] = timestamp
            data_source_company_record['external_id'], self.external_ids = get_new_external_id(data_source.data_source_id,
                                                                                               self.external_ids)
            data_source_company_record['gen_id'] = str(synthetic_records_dict['gen_id'])
            company_records.append(data_source_company_record)

        synthetic_records_dict['company_records'] = pd.DataFrame(company_records)
        return synthetic_records_dict

    def generate_dataset(self, seed: int, save_results: bool):
        synthetic_dataset_result_path = dataset_results_file_path__with_subfolders(['synthetic_data'], '')
        path_exists_or_create(synthetic_dataset_result_path)

        self.synthetic_companies_base_df = pd.DataFrame(
            columns=list(self.companies_data_df.columns) + ['external_id', 'data_source_id'])

        self.synthetic_records_dicts_list = []
        # We do an initial loop to assign to the SingleCompanyDataArtifacts the data sources they will be applied to

        single_record_artifact_subclasses = SingleCompanyDataArtifact.__subclasses__()
        single_record_artifacts_params_dict = {}
        for artifact_subclass in single_record_artifact_subclasses:
            for data_source in self.data_sources_list:
                if artifact_subclass.__name__ in data_source.artifacts_params:
                    if artifact_subclass.__name__ in single_record_artifacts_params_dict:
                        single_record_artifacts_params_dict[artifact_subclass.__name__][data_source.data_source_id] = data_source.artifacts_params[artifact_subclass.__name__]
                    else:
                        single_record_artifacts_params_dict[artifact_subclass.__name__] = {}
                        single_record_artifacts_params_dict[artifact_subclass.__name__][data_source.data_source_id] = data_source.artifacts_params[artifact_subclass.__name__]

        single_company_artifacts_list = []

        for artifact_subclass in single_record_artifact_subclasses:
            artifact_params = single_record_artifacts_params_dict[artifact_subclass.__name__]
            single_company_artifacts_list.append(artifact_subclass(artifact_params))

        # We do an initial loop applying the SingleCompanyDataArtifacts for each SyntheticCompanyDataSource
        for index, company_record in tqdm(self.companies_data_df.iterrows(), total=len(self.companies_data_df),
                                          position=0, leave=True, colour='green',
                                          desc='Applying SingleCompanyDataArtifacts', ):

            # We store the original company record in a dict
            synthetic_records_dict = self._init_syn_record(company_record)
            single_comp_data_artifacts = []
            for single_company_artifact in single_company_artifacts_list:
                synthetic_records_dict, applied_artifact = single_company_artifact.apply_artifact(synthetic_records_dict)
                single_comp_data_artifacts.append(applied_artifact)

            synthetic_records_dict['data_artifacts']['applied_SingleCompanyDataArtifacts'] = single_comp_data_artifacts
            self.synthetic_records_dicts_list.append(synthetic_records_dict)

        # We now apply the MultiCompanyDataArtifacts to each set of synthetic records

        for synthetic_records_dict in tqdm(self.synthetic_records_dicts_list,
                                           total=len(self.synthetic_records_dicts_list),
                                           leave=True, colour='green',
                                           desc='Applying MultiCompanyDataArtifacts', ):
            multi_comp_data_artifacts = []
            for artifact in self.multi_record_artifacts:

                prob = random.uniform(0, 1)

                if prob < artifact.prob:
                    synthetic_records_dict, applied_artifact = artifact.apply_artifact(synthetic_records_dict)
                    multi_comp_data_artifacts.append(applied_artifact)

            synthetic_records_dict['data_artifacts']['applied_MultiCompanyDataArtifacts'] = multi_comp_data_artifacts

        # We now apply the DataDriftDataArtifacts to each set of synthetic records

        for synthetic_records_dict in tqdm(self.synthetic_records_dicts_list,
                                           total=len(self.synthetic_records_dicts_list),
                                           leave=True, colour='green',
                                           desc='Applying DataDriftDataArtifacts', ):
            data_drift_data_artifacts = []
            # We choose one DataDriftDataArtifact to potentially apply
            artifact = random.choice(self.data_drift_artifacts)

            prob = random.uniform(0, 1)
            # We multiply the artifact.prob by the number of data_drift artifacts in order to respect the final
            # proportion of company groups that will have a specific data_drift_artifact applied to them (which can't
            # be higher than 1/len(self.data_drift_artifacts)
            if prob < artifact.prob * len(self.data_drift_artifacts):
                synthetic_records_dict, self.investors_data_df, applied_artifact, self.external_ids = artifact.apply_artifact(
                    synthetic_records_dict,
                    self.investors_data_df,
                    self.synthetic_records_dicts_list,
                    self.external_ids)
                data_drift_data_artifacts.append(applied_artifact)

            synthetic_records_dict['data_artifacts']['applied_DataDriftDataArtifacts'] = data_drift_data_artifacts

        # We now apply the BatchCompanyDataArtifacts to each set of synthetic records

        for artifact in self.batch_data_artifacts:

            prob = random.uniform(0, 1)

            if prob < artifact.prob:
                self.synthetic_records_dicts_list = artifact.apply_artifact(self.synthetic_records_dicts_list)

        if save_results:
            # We save the generated company records to a csv
            self.synthetic_companies_dataset_df = build_df_from_records(self.synthetic_records_dicts_list,
                                                                        'company_records')
            self.synthetic_companies_dataset_df.to_csv(os.path.join(synthetic_dataset_result_path,
                                                                    'synthetic_companies_dataset_seed_{}_size_{}_unshuffled.csv'.format(
                                                                        seed,
                                                                        len(self.synthetic_companies_dataset_df))))

        return self.synthetic_records_dicts_list

    def get_new_gen_id(self, ):

        gen_id = self.last_gen_id

        self.last_gen_id = self.last_gen_id + 1

        return gen_id


def generate_companies_dataset(seed: int, save_results: bool):
    artifact_params_dict = {
        'data_source_1_artifact_params': {'TextLengtheningArtifactSingleCompany': {'modified_attribute': 'name',
                                                                                   'prob': 0.75},
                                          'CountryCodeFormatChangeArtifactSingleCompany': {
                                              'modified_attribute': 'country_code',
                                              'prob': 0.5},
                                          'MissingAttributeArtifactSingleCompany': [
                                              {'modified_attribute': 'short_description',
                                               'prob': 0.1}]},
        'data_source_2_artifact_params': {'TextShorteningArtifactSingleCompany': {'modified_attribute': 'name',
                                                                                  'prob': 1},
                                          'SplitUpperCaseInWordArtifactSingleCompany': {'modified_attribute': 'name',
                                                                                        'prob': 1},
                                          'RegionFormatChangeArtifactSingleCompany': {'modified_attribute': 'region',
                                                                                      'prob': 0.75},
                                          'MissingAttributeArtifactSingleCompany': [
                                              {'modified_attribute': 'short_description',
                                               'prob': 1},
                                              {'modified_attribute': 'city',
                                               'prob': 1},
                                              {'modified_attribute': 'country_code',
                                               'prob': 0.75},
                                          ],
                                          'DeletePunctAndStopWordsArtifactSingleCompany': {'modified_attribute': 'name',
                                                                                           'prob': 0.75},
                                          },

        'data_source_3_artifact_params': {
            'TextShorteningArtifactSingleCompany': {'modified_attribute': 'name',
                                                    'prob': 1},
            'MissingAttributeArtifactSingleCompany': [{'modified_attribute': 'short_description',
                                                       'prob': 1},
                                                      {'modified_attribute': 'city',
                                                       'prob': 1},
                                                      {'modified_attribute': 'region',
                                                       'prob': 0.75}],
            'DeletePunctAndStopWordsArtifactSingleCompany': {'modified_attribute': 'name',
                                                             'prob': 1},
            'MergeUpperCaseInWordArtifactSingleCompany': {'modified_attribute': 'name',
                                                          'prob': 1},
            'MergeDotComWordArtifactSingleCompany': {'modified_attribute': 'name',
                                                     'prob': 1},
            'CountryCodeFormatChangeArtifactSingleCompany': {'modified_attribute': 'country_code',
                                                             'prob': 0.75},
            'RegionFormatChangeArtifactSingleCompany': {'modified_attribute': 'region',
                                                        'prob': 0.75},
            'CityOverwriteArtifactSingleCompany': {'modified_attribute': 'city',
                                                   'prob': 0.25},
        },
        'data_source_4_artifact_params': {
            'TextLengtheningArtifactSingleCompany': {'modified_attribute': 'name',
                                                     'prob': 0.75},
            'DeletePunctAndStopWordsArtifactSingleCompany': {'modified_attribute': 'name',
                                                             'prob': 0.75},
            'MissingAttributeArtifactSingleCompany': [{'modified_attribute': 'short_description',
                                                       'prob': 1},
                                                      {'modified_attribute': 'city',
                                                       'prob': 1},
                                                      {'modified_attribute': 'region',
                                                       'prob': 0.75},
                                                      ],
            'SplitUpperCaseInWordArtifactSingleCompany': {'modified_attribute': 'name',
                                                          'prob': 1},
            'AcronymNameArtifactSingleCompany': {'modified_attribute': 'name',
                                                 'prob': 0.05},
            'CountryCodeFormatChangeArtifactSingleCompany': {'modified_attribute': 'country_code',
                                                             'prob': 0.75},
            'RegionOverwriteArtifactSingleCompany': {'modified_attribute': 'region',
                                                     'prob': 0.05},
            'RegionCountrySwapArtifactSingleCompany': {'modified_attribute': 'region',
                                                       'prob': 0.1},
        },
        'data_source_5_artifact_params': {
            'TextShorteningArtifactSingleCompany': {'modified_attribute': 'name',
                                                     'prob': 1},
            'DeletePunctAndStopWordsArtifactSingleCompany': {'modified_attribute': 'name',
                                                             'prob': 1},
            'MissingAttributeArtifactSingleCompany': [{'modified_attribute': 'short_description',
                                                        'prob': 0.5},
                                                      {'modified_attribute': 'city',
                                                       'prob': 0.5},
                                                      {'modified_attribute': 'region',
                                                       'prob': 0.5},
                                                      ],
            'SplitUpperCaseInWordArtifactSingleCompany': {'modified_attribute': 'name',
                                                          'prob': 1},
            'AcronymNameArtifactSingleCompany': {'modified_attribute': 'name',
                                                 'prob': 0.05},
            'CountryCodeFormatChangeArtifactSingleCompany': {'modified_attribute': 'country_code',
                                                             'prob': 1},
            'RegionFormatChangeArtifactSingleCompany': {'modified_attribute': 'region',
                                                        'prob': 0.75},
            'RegionOverwriteArtifactSingleCompany': {'modified_attribute': 'region',
                                                     'prob': 0.1},
            'RegionCountrySwapArtifactSingleCompany': {'modified_attribute': 'region',
                                                       'prob': 0.1},

                           },

    }
    multi_company_artifacts_params = {
        'InsertCorporateTermArtifactMultiCompany': {'modified_attributes': 'name', 'prob': 0.75},
        'MissingRecordArtifactMultiCompany': {'modified_attributes': '', 'prob': 1, 'missing_per_by_data_source':{
            '1': 0, '2': 0.05 + random.uniform(-0.1,0.1), '3': 0.25 + random.uniform(-0.1,0.1),
            '4': 0.25 + random.uniform(-0.1,0.1), '5': 0.05 + random.uniform(-0.1,0.1)}},
        'MultipleRegionsDataArtifactMultiCompany': {'modified_attributes': ['region', 'city'], 'prob': 0.25},
        'MultipleCountriesDataArtifactMultiCompany': {'modified_attributes': ['country_code', 'region', 'city'],
                                                      'prob': 0.01},

    }
    data_drift_artifacts_params = {'CreateCorporateAcquisitionDataArtifact': {'modified_attributes':
                                                                                  ['name', 'city', 'region',
                                                                                   'country_code', 'short_description'],
                                                                              'prob': 0.1},
                                   'CreateCorporateMergerDataArtifact': {'modified_attributes':
                                                                             ['name', 'city', 'region',
                                                                              'country_code', 'short_description'],
                                                                         'prob': 0.1}}
    batch_data_artifacts_params = {'BatchParaphraseAttributeArtifact':{'modified_attribute' :'short_description',
                                   'prob': 1,
                                   'gpu_seed': seed,
                                   'data_source_id': '5'},
    }
    data_sources_list = []

    for idx, artifacts_params_dict in enumerate(artifact_params_dict):
        data_source_id = int(re.findall(r'\d+', artifacts_params_dict)[0])
        data_sources_list.append(SyntheticCompanyDataSource(data_source_name='data_source_{}'.format(data_source_id),
                                                            data_source_id=data_source_id,
                                                            single_record_artifacts_params=artifact_params_dict[
                                                                artifacts_params_dict]))

    real_data_df = pd.read_csv(dataset_raw_file_path(os.path.join('synthetic_data', 'organizations.csv')), low_memory= False).fillna('')
    companies_generator = SyntheticCompaniesGenerator(real_data_df=real_data_df,
                                                      max_company_seeds= 200000,
                                                      data_sources_list=data_sources_list,
                                                      multi_record_artifacts_params=multi_company_artifacts_params,
                                                      data_drift_artifacts_params=data_drift_artifacts_params,
                                                      batch_data_artifacts_params = batch_data_artifacts_params)

    synthetic_records_dicts_list = companies_generator.generate_dataset(seed, save_results=save_results)

    return synthetic_records_dicts_list
