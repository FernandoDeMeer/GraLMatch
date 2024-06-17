import copy

import pandas as pd
from tqdm.auto import tqdm
import re
from src.helpers.path_helper import *
from src.data.synthetic_data_generation.data_artifacts_securities import *
from src.data.synthetic_data_generation.utils import build_df_from_records

class SyntheticSecurityDataSource():

    def __init__(self, data_source_name: str, data_source_id: int, single_record_artifacts_params: dict):


        self.name = data_source_name
        self.data_source_id = str(data_source_id)
        self.artifacts_params = single_record_artifacts_params


class SyntheticSecuritiesGenerator():

    def __init__(self, synthetic_records_dicts_list: list,
                 data_sources_list: list, multi_security_artifacts_params: dict,
                 data_drift_artifacts_params: dict):

        self.synthetic_records_dicts_list = synthetic_records_dicts_list
        self.synthetic_companies_data_df = build_df_from_records(self.synthetic_records_dicts_list,
                                                                 'company_records')

        self.data_sources_list = data_sources_list
        data_source_ids = set([str(data_source.data_source_id) for data_source in data_sources_list])
        companies_data_sources = set(self.synthetic_companies_data_df['data_source_id'].astype(str).unique())
        if data_source_ids != companies_data_sources:
            raise Exception('Synthetic Data Sources across companies and securities need to match')

        self.external_ids = {data_source.data_source_id: None for data_source in self.data_sources_list}

        self.multi_record_artifacts = get_artifacts(multi_security_artifacts_params, MultiSecurityDataArtifact)
        self.data_drift_artifacts = get_artifacts(data_drift_artifacts_params, DataDriftSecurityDataArtifact)


    def init_security_records(self, syn_records_dict: dict):

        data_sources_list = syn_records_dict['company_records']['data_source_id'].unique()
        security_seed = syn_records_dict['security_seed']

        security_records = []
        for data_source_id in data_sources_list:
            data_source_security_record = copy.deepcopy(security_seed)
            issuer_record = syn_records_dict['company_records'][syn_records_dict['company_records']['data_source_id'] == data_source_id].squeeze()
            data_source_security_record['data_source_id'] = data_source_id
            data_source_security_record['issuer_id'] = issuer_record['external_id']
            data_source_security_record['inserted'] = issuer_record['inserted']
            data_source_security_record['last_modified'] = issuer_record['last_modified']
            data_source_security_record['external_id'], self.external_ids = get_new_external_id(data_source_id,
                                                                                                     self.external_ids)
            data_source_security_record['gen_id'] = str(syn_records_dict['gen_id'])
            security_records.append(data_source_security_record)

        syn_records_dict['security_records'] = pd.DataFrame(security_records)

    def generate_dataset(self, seed:int, save_results: bool):
        synthetic_dataset_result_path = dataset_results_file_path__with_subfolders(['synthetic_data'], '')
        path_exists_or_create(synthetic_dataset_result_path)

        self.synthetic_securities_dataset_df = pd.DataFrame(
            columns=['name', 'type', 'ISIN', 'CUSIP', 'VALOR', 'SEDOL', 'primary_currency', 'external_id', 'data_source_id', 'issuer_id', 'inserted', 'last_modified', 'gen_id'], )

        # We do an initial loop creating a security seed from each original record that we will apply
        # data artifacts to.
        self.seeds_ids = {key:[] for key in ['ISIN', 'CUSIP', 'VALOR', 'SEDOL',]}
        for synthetic_records_dict in tqdm(self.synthetic_records_dicts_list,
                                           total=len(self.synthetic_records_dicts_list),
                                           leave=True, colour='green',
                                           desc='Creating security seeds ', ):
            self.create_security_seed(synthetic_records_dict)

        # We do an initial loop to assign to the SingleSecurityDataArtifacts the data sources they will be applied to

        single_record_artifact_subclasses = SingleSecurityDataArtifact.__subclasses__()
        single_record_artifacts_params_dict = {}
        for artifact_subclass in single_record_artifact_subclasses:
            for data_source in self.data_sources_list:
                if artifact_subclass.__name__ in data_source.artifacts_params:
                    if artifact_subclass.__name__ in single_record_artifacts_params_dict:
                        single_record_artifacts_params_dict[artifact_subclass.__name__][data_source.data_source_id] = data_source.artifacts_params[artifact_subclass.__name__]
                    else:
                        single_record_artifacts_params_dict[artifact_subclass.__name__] = {}
                        single_record_artifacts_params_dict[artifact_subclass.__name__][data_source.data_source_id] = data_source.artifacts_params[artifact_subclass.__name__]

        single_security_artifacts_list = []

        for artifact_subclass in single_record_artifact_subclasses:
            artifact_params = single_record_artifacts_params_dict[artifact_subclass.__name__]
            single_security_artifacts_list.append(artifact_subclass(artifact_params))

        # We do an initial loop applying the SingleSecurityDataArtifacts for each SyntheticCompanyDataSource and
        # init the security records

        for synthetic_records_dict in tqdm(self.synthetic_records_dicts_list,
                                           total=len(self.synthetic_records_dicts_list),
                                           leave=True, colour='green',
                                           desc='Applying SingleSecurityDataArtifacts', ):
            self.init_security_records(synthetic_records_dict)
            single_sec_data_artifacts = []
            for single_security_artifact in single_security_artifacts_list:
                synthetic_records_dict, applied_artifact = single_security_artifact.apply_artifact(synthetic_records_dict)
                single_sec_data_artifacts.append(applied_artifact)

            synthetic_records_dict['data_artifacts']['applied_SingleSecurityDataArtifacts'] = single_sec_data_artifacts

        # We now apply the MultiSecurityDataArtifact to each set of synthetic records

        syn_records_ids = self.get_syn_records_ids()

        for synthetic_records_dict in tqdm(self.synthetic_records_dicts_list,
                                           total=len(self.synthetic_records_dicts_list),
                                           leave=True, colour='green',
                                           desc='Applying MultiSecurityDataArtifacts', ):
            multi_comp_data_artifacts = []
            for artifact in self.multi_record_artifacts:

                prob = random.uniform(0, 1)
                if prob < artifact.prob:
                    synthetic_records_dict, applied_artifact, syn_records_ids, self.external_ids = artifact.apply_artifact(synthetic_records_dict,
                                                                     syn_records_ids, self.external_ids)
                    multi_comp_data_artifacts.append(applied_artifact)

            synthetic_records_dict['data_artifacts']['applied_MultiSecurityDataArtifacts'] = multi_comp_data_artifacts

        # We now apply the DataDriftDataArtifacts for each recorded data_drift event

        for synthetic_records_dict in tqdm(self.synthetic_records_dicts_list,
                                             total=len(self.synthetic_records_dicts_list),
                                             leave=True,colour='green',
                                             desc='Applying DataDriftDataArtifacts'):
            data_drift_data_artifacts = []
            for artifact in self.data_drift_artifacts:
                synthetic_records_dict, applied_artifact, self.external_ids = artifact.apply_artifact(synthetic_records_dict,
                                                              self.synthetic_records_dicts_list, self.external_ids)
                data_drift_data_artifacts.append(applied_artifact)

            synthetic_records_dict['data_artifacts']['applied_DataDriftSecurityDataArtifacts'] = data_drift_data_artifacts

        if save_results:
            # We save the generated records to a csv
            self.synthetic_securities_dataset_df = build_df_from_records(self.synthetic_records_dicts_list,
                                                                         'security_records')
            self.synthetic_securities_dataset_df.to_csv(os.path.join(synthetic_dataset_result_path,
                                                                     'synthetic_securities_dataset_seed_{}_size_{}_unshuffled.csv'.format(
                                                                         seed,
                                                                         len(self.synthetic_securities_dataset_df))))

        return self.synthetic_records_dicts_list

    def create_security_seed(self, synthetic_records_dict: dict):
        company_seed = synthetic_records_dict['company_seed']
        security_seed = pd.Series(dtype=object)

        # We set the name of the seed to be the issuer's name
        security_seed['name'] = company_seed['name']

        for id_attribute in ['ISIN', 'CUSIP', 'VALOR', 'SEDOL']:

            gen_id = generate_id_attribute(company_seed, id_attribute, security_seed, self.seeds_ids)
            while gen_id in self.seeds_ids[id_attribute]:
                gen_id = generate_id_attribute(company_seed, id_attribute, security_seed, self.seeds_ids)

            security_seed[id_attribute] = gen_id
            self.seeds_ids[id_attribute].append(gen_id)

        security_seed['primary_currency'] = self.get_primary_currency(company_seed)

        synthetic_records_dict['security_seed'] = security_seed


    def get_primary_currency(self, original_company: pd.Series):

        country_code = original_company['country_code']
        country_dict = next(
            (item for item in countries_states_cities_dict if item['iso3'] == country_code), None)

        if country_dict is not None:
            currency = country_dict['currency']
        else:
            currency = None

        return currency

    def get_syn_records_ids(self):

        syn_records_ids = {}

        for id_attribute in ID_ATTRIBUTES:
            attribute_values = [list(syn_record_dict['security_records'][id_attribute].dropna().unique()) for syn_record_dict in
                                self.synthetic_records_dicts_list]
            # We flatten the list of lists
            syn_records_ids[id_attribute] = [x for l in attribute_values for x in l if x != '']

        return syn_records_ids


def generate_securities_dataset(synthetic_records_dicts_list: list, seed: int, save_results: bool):
    random.seed(seed)
    artifact_params_dict = {
        'data_source_1_artifact_params': {
            'GenericNamingWCompanyNameArtifactSingleSecurity': {'modified_attribute': 'name',
                                                                'prob': 1},
            'MissingAttributeArtifactSingleSecurity': [{'modified_attribute': 'ISIN',
                                                        'prob': 0},
                                                       {'modified_attribute': 'CUSIP',
                                                        'prob': 0.25},
                                                       {'modified_attribute': 'SEDOL',
                                                        'prob': 0.5},
                                                       {'modified_attribute': 'VALOR',
                                                        'prob': 0.25}
                                                       ], },

        'data_source_2_artifact_params': {
            'GenericNamingWCompanyNameArtifactSingleSecurity': {'modified_attribute': 'name',
                                                                'prob': 1},
            'CurrencyFormatChangeArtifactSingleSecurity': {'modified_attribute': 'primary_currency',
                                                           'prob': 0.5},
            'MissingAttributeArtifactSingleSecurity': [{'modified_attribute': 'ISIN',
                                                        'prob': 0.25},
                                                       {'modified_attribute': 'CUSIP',
                                                        'prob': 0.5},
                                                       {'modified_attribute': 'SEDOL',
                                                        'prob': 0.5},
                                                       {'modified_attribute': 'VALOR',
                                                        'prob': 0.5},
                                                       {'modified_attribute': 'primary_currency',
                                                        'prob': 0.5}
                                                       ], },

        'data_source_3_artifact_params': {
            'GenericNamingNoCompanyNameArtifactSingleSecurity': {'modified_attribute': 'name',
                                                                 'prob': 1},
            'CurrencyFormatChangeArtifactSingleSecurity': {'modified_attribute': 'primary_currency',
                                                           'prob': 0.5},
            'MissingAttributeArtifactSingleSecurity': [{'modified_attribute': 'ISIN',
                                                        'prob': 0.25},
                                                       {'modified_attribute': 'CUSIP',
                                                        'prob': 0.5},
                                                       {'modified_attribute': 'SEDOL',
                                                        'prob': 0.75},
                                                       {'modified_attribute': 'VALOR',
                                                        'prob': 0.75},
                                                       {'modified_attribute': 'primary_currency',
                                                        'prob': 0.5}
                                                       ], },

        'data_source_4_artifact_params': {
            'GenericNamingNoCompanyNameArtifactSingleSecurity': {'modified_attribute': 'name',
                                                                 'prob': 1},
            'CurrencyFormatChangeArtifactSingleSecurity': {'modified_attribute': 'primary_currency',
                                                           'prob': 0.75},
            'MissingAttributeArtifactSingleSecurity': [{'modified_attribute': 'ISIN',
                                                        'prob': 0.5},
                                                       {'modified_attribute': 'CUSIP',
                                                        'prob': 0.75},
                                                       {'modified_attribute': 'SEDOL',
                                                        'prob': 0.75},
                                                       {'modified_attribute': 'VALOR',
                                                        'prob': 0.75},
                                                       {'modified_attribute': 'primary_currency',
                                                        'prob': 0.8}
                                                       ], },
        'data_source_5_artifact_params': {
            'GenericNamingNoCompanyNameArtifactSingleSecurity': {'modified_attribute': 'name',
                                                                 'prob': 1},
            'CurrencyFormatChangeArtifactSingleSecurity': {'modified_attribute': 'primary_currency',
                                                           'prob': 0.5},
            'MissingAttributeArtifactSingleSecurity': [{'modified_attribute': 'ISIN',
                                                        'prob': 0.05},
                                                       {'modified_attribute': 'CUSIP',
                                                        'prob': 0.75},
                                                       {'modified_attribute': 'SEDOL',
                                                        'prob': 0.75},
                                                       {'modified_attribute': 'VALOR',
                                                        'prob': 0.75},
                                                       {'modified_attribute': 'primary_currency',
                                                        'prob': 0.8}
                                                       ], },

    }
    multi_security_artifacts_params = {
        'MultipleIDsArtifactMultiSecurity': {'modified_attributes': ['ISIN', 'CUSIP', 'VALOR', 'SEDOL'], 'prob': 0.25},
        'NoIdOverlapsArtifactMultiSecurity': {'modified_attributes': ['ISIN', 'CUSIP', 'VALOR', 'SEDOL'], 'prob': 0.25},
        'MultipleSecuritiesArtifactMultiSecurity': {'modified_attributes': 'all', 'prob': 0.25},
        'MissingRecordArtifactMultiCompany': {'modified_attributes': 'all', 'prob': 0.25}}

    data_drift_artifacts_params = {
        'CorporateMergerSecurityDataArtifact': {'modified_attributes': ['ISIN', 'CUSIP', 'VALOR', 'SEDOL']},
    }

    data_sources_list = []

    for idx, artifacts_params_dict in enumerate(artifact_params_dict):
        data_source_id = int(re.findall(r'\d+', artifacts_params_dict)[0])
        data_sources_list.append(SyntheticSecurityDataSource(data_source_name='data_source_{}'.format(data_source_id),
                                                             data_source_id=data_source_id,
                                                             single_record_artifacts_params=artifact_params_dict[
                                                                 artifacts_params_dict]))

    securities_generator = SyntheticSecuritiesGenerator(synthetic_records_dicts_list,
                                                        data_sources_list,
                                                        multi_security_artifacts_params,
                                                        data_drift_artifacts_params)

    synthetic_records_dicts_list = securities_generator.generate_dataset(seed=seed, save_results = save_results)

    return synthetic_records_dicts_list

