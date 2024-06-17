import random
import string
from abc import ABC, abstractmethod
import copy

from src.data.synthetic_data_generation.utils import *
from src.data.synthetic_data_generation.data_artifacts_companies import countries_states_cities_dict

# We use an extended version of the dict used in src.data.security_tokenizer
SECURITY_TYPES_DICT = {
    '[right]': ['Equity Rights', 'Right', 'Pref. Right', 'Temporary Rights', 'Dividend Rights', 'Capital Rights',
                'Preferred Rights', 'Class A Rights', 'Preferred Equity Rights'],
    '[equity]': ['Shares/Units with shares/Particip. Cert.', 'Preferred Equity/Derivative Unit',
             'Equity Convertible Preference', 'Equity Depositary Receipts', 'Equity Depositary Receipt',
             'Equity Depository Receipt',
             'Equity/Preferred Unit', 'Equity Derivatives', 'Equity Preference', 'US Dep Rect (ADR)',
             'Preferred Equity', 'Preference Share', 'Preferred Issue', 'Preferred Stock', 'Ordinary Shares',
             'registered shs', 'Ordinary Share', 'Equity Shares', 'Common Shares', 'Common Equity',
             'Equity Issue', 'Common Stock', 'Ord Shs', 'Equity', 'Series A', 'Domestic Shares'],
    '[unit]': ['Dept/Equity Composite Units', 'Investment trust unit/share', 'Equity/Derivative Unit',
             'Dept/Derivative Unit', 'Dept/Preferred Unit', 'Units', 'Composite Units'],
    '[bond]': ['Bond', 'Invest Grade Bond', 'IG Bond', 'Zero Coupon Bond', 'ZC Bond', 'Floating Rate Bond',
               'FR Bond', 'Convertible Bond', 'Conv Bond', 'High-Yield Bond', 'High-Yield', 'HY Bond'],
}

ID_ATTRIBUTES = ['ISIN', 'CUSIP', 'VALOR', 'SEDOL']
class SingleSecurityDataArtifact(ABC):

    def __init__(self, artifact_name: str, artifact_params: dict):
        self.name = artifact_name
        self.params = artifact_params
    @abstractmethod
    def _apply_artifact(self, security_records: pd.DataFrame, data_source_id: str, modified_attributes: list,
                        syn_records_dict: dict):
        """Basic function of each SingleSecurityDataArtifact subclass. It takes a security_record
         pandas Series which contains the following attributes:
            -security_name
            -ISIN (id)
            -CUSIP (id)
            -VALOR (id)
            -SEDOL (id)
            -primary_currency
            -data_source_id
            -issuer_id (gen_id of the issuing company)
        and applies the corresponding single-security data artifact.
        """
        raise NotImplementedError("Should be implemented in the respective subclasses.")

    def apply_artifact(self, syn_records_dict: dict):
        applied_artifact = {}
        active_data_sources = syn_records_dict['company_records']['data_source_id'].unique()
        for data_source_id, params_dict in self.params.items():
            # A company may be missing from a given data_source
            if active_data_sources in active_data_sources:
                modified_attributes = []
                probs = []
                if type(params_dict) != list:
                    params_dict = [params_dict]
                for params in params_dict:
                    modified_attributes.append(params['modified_attribute'])
                    probs.append(params['prob'])
                modified_attributes = list(zip(modified_attributes, probs))
                modified_attributes = [modified_attribute[0] for modified_attribute in modified_attributes if
                                       random.uniform(0, 1) < modified_attribute[1]]
                if len(modified_attributes) > 0:

                    syn_records_dict['security_records'] = self._apply_artifact(syn_records_dict['security_records'],
                                                                               data_source_id, modified_attributes,
                                                                                syn_records_dict,)
                    applied_artifact[data_source_id] = self.name + '_'.join(modified_attributes)

        return syn_records_dict, applied_artifact


class GenericNamingWCompanyNameArtifactSingleSecurity(SingleSecurityDataArtifact):

    def __init__(self, artifact_params: dict):
        artifact_name = 'generic_security_naming_w_company_name_artifact'
        super().__init__(artifact_name, artifact_params)
        # Each data source has a fixed generic_equity_term
        self.generic_equity_terms = {}
        for data_source_id in artifact_params.keys():
            self.generic_equity_terms[data_source_id] = random.choice(SECURITY_TYPES_DICT['[equity]'])

    def _apply_artifact(self, security_records: pd.DataFrame, data_source_id: str, modified_attributes: list,
                        syn_records_dict: dict):
        """" Renames the security to "issuer name + generic equity term"
        """
        issuer_record = syn_records_dict['company_records'][
            syn_records_dict['company_records']['data_source_id'] == data_source_id].squeeze()

        try:
            issuer_name = issuer_record['name']
            security_records.loc[security_records['data_source_id'] == data_source_id,
                                 'name'] = issuer_name + ' ' + self.generic_equity_terms[data_source_id]
            security_records.loc[security_records['data_source_id'] == data_source_id,
                                 'type'] = self.generic_equity_terms[data_source_id]
        except:
            pass

        return security_records


class GenericNamingNoCompanyNameArtifactSingleSecurity(SingleSecurityDataArtifact):

    def __init__(self, artifact_params: dict):
        artifact_name = 'generic_security_naming_no_company_name_artifact'
        super().__init__(artifact_name, artifact_params)
        # Each data source has a fixed generic_equity_term
        self.generic_equity_terms = {}
        for data_source_id in artifact_params.keys():
            self.generic_equity_terms[data_source_id] = random.choice(SECURITY_TYPES_DICT['[equity]'])
    def _apply_artifact(self, security_records: pd.DataFrame, data_source_id: str, modified_attributes: list,
                        syn_records_dict: dict):
        """" Renames the security to a generic equity term
        """

        try:
            security_records.loc[security_records['data_source_id'] == data_source_id,
                                 'name'] = self.generic_equity_terms[data_source_id]
            security_records.loc[security_records['data_source_id'] == data_source_id,
                                 'type'] = self.generic_equity_terms[data_source_id]
        except:
            pass
        return security_records


class MissingAttributeArtifactSingleSecurity(SingleSecurityDataArtifact):

    def __init__(self, artifact_params: dict):
        artifact_name = 'missing_attribute_security_artifact'
        super().__init__(artifact_name, artifact_params)

    def _apply_artifact(self, security_records: pd.DataFrame, data_source_id: str, modified_attributes: list,
                        syn_records_dict: dict):
        """" Deletes the corresponding attribute from the security record.
        """
        try:
            security_records.loc[security_records['data_source_id'] == data_source_id,
                                 modified_attributes] = ''
        except:
            pass
        return security_records


class CurrencyFormatChangeArtifactSingleSecurity(SingleSecurityDataArtifact):

    def __init__(self, artifact_params: dict):
        artifact_name = 'currency_format_change_security_artifact'
        super().__init__(artifact_name, artifact_params)

    def _apply_artifact(self, security_records: pd.DataFrame, data_source_id: str, modified_attributes: list,
                        syn_records_dict: dict):
        """" Changes the format of the primary_currency attribute with any of its variations (currency, currency_name or
        currency_symbol).
        """
        try:
            country_code = syn_records_dict['company_seed']['country_code']
            country_dict = next(
                (item for item in countries_states_cities_dict if item['iso3'] == country_code), None)
            if country_dict is not None:
                security_records.loc[security_records['data_source_id'] == data_source_id,
                                     'primary_currency'] = random.choice(
                    [country_dict['currency'], country_dict['currency_name'], country_dict['currency_symbol']])
        except:
            pass
        return security_records


class MultiSecurityDataArtifact(ABC):

    def __init__(self, artifact_name: str, prob: float):
        self.name = artifact_name
        self.prob = prob

    def _check_for_duplicates(self, attributes_list, new_attribute):
        return new_attribute in attributes_list

    def _insert_letters(self, string, index, letters):
        return string[:index] + letters + string[index:]

    @abstractmethod
    def apply_artifact(self, syn_records_dict: dict, syn_records_ids: dict, external_ids: dict):
        """Basic function of each MultiSecurityDataArtifact subclass. It takes a synthetic_records_dict which contains a
         pandas Dataframe of securities with the following attributes:
            -security_name
            -ISIN (id)
            -CUSIP (id)
            -VALOR (id)
            -SEDOL (id)
            -primary_currency
            -data_source_id
            -external_id (id of the security assigned by the data source)
            -issuer_id (id of the issuing company)
        and applies the corresponding multi-security data artifact.
        """
        raise NotImplementedError("Should be implemented in the respective subclasses.")


class MultipleIDsArtifactMultiSecurity(MultiSecurityDataArtifact):

    def __init__(self, modified_attributes: str, prob: float):
        artifact_name = 'multiple_ids_multi_security_artifact'.format(modified_attributes)
        super().__init__(artifact_name, prob)
        self.modified_attribute = modified_attributes
        self.id_attributes = ID_ATTRIBUTES

    def apply_artifact(self, syn_records_dict: dict, syn_records_ids: dict, external_ids: dict):
        """ Creates new ID(s) and assigns them to multiple records of a single security creating transitive matches
        (i.e. only some records of the same security can be matched via id overlaps).
        """
        no_of_ids_to_change = random.randint(1, 4)

        ids_to_change = random.sample(self.id_attributes, no_of_ids_to_change)
        company_seed = syn_records_dict['company_seed']
        security_seed = syn_records_dict['security_seed']
        security_records = syn_records_dict['security_records']

        # First we select which rows of security_records we are going to modify, we leave at least one record unmodified
        if len(security_records) > 1:
            no_of_rows_to_change = random.randint(1, security_records.shape[0] - 1)

            rows_to_change = random.sample(list(security_records.index), no_of_rows_to_change)

            for id_attribute in ids_to_change:
                # We generate a new value for the chosen id attribute and check its uniqueness
                new_id_value = generate_id_attribute(company_seed, id_attribute, security_seed, syn_records_ids)
                while self._check_for_duplicates(syn_records_ids[id_attribute], new_attribute= new_id_value):
                    new_id_value = generate_id_attribute(company_seed, id_attribute, security_seed, syn_records_ids)
                syn_records_ids[id_attribute].append(new_id_value)
                # We assign the new_id_value to the rows_to_change
                security_records.loc[rows_to_change, id_attribute] = new_id_value

            # We delete the id attributes that we did not modify, so that there are no other id overlaps present

            for id_attribute in self.id_attributes:
                if id_attribute not in ids_to_change:
                    security_records[id_attribute] = ''

            syn_records_dict['security_records'] = security_records

            # Finally, we modify the 'last_modified' attribute of the modified records
            years_after_event = random.randint(1,10)
            for row in rows_to_change:
                initial_timestamp = syn_records_dict['timestamp']
                event_timestamp = initial_timestamp
                while event_timestamp == initial_timestamp:
                    try:
                        event_timestamp = initial_timestamp.replace(year = initial_timestamp.year + years_after_event) + datetime.timedelta(
                            days=random.randint(0, 30),
                            hours=random.randint(0, 23),
                            minutes=random.randint(0, 59),
                            seconds=random.randint(0, 59))
                    except:
                        # We can get a ValueError if the initial timestamp is a Feb 29th
                        event_timestamp = initial_timestamp.replace(year = initial_timestamp.year + years_after_event, month = initial_timestamp.month
                                                                    + random.choice([-1,1])) + datetime.timedelta(
                            days=random.randint(0, 30),
                            hours=random.randint(0, 23),
                            minutes=random.randint(0, 59),
                            seconds=random.randint(0, 59))

                security_records.loc[row, 'last_modified'] = random_close_date(
                    event_timestamp)
        return syn_records_dict, self.name, syn_records_ids, external_ids


class NoIdOverlapsArtifactMultiSecurity(MultiSecurityDataArtifact):

    def __init__(self, modified_attributes: str, prob: float):
        artifact_name = 'no_id_overlap_multi_security_artifact'.format(modified_attributes)
        super().__init__(artifact_name, prob)
        self.modified_attribute = modified_attributes
        self.id_attributes = ID_ATTRIBUTES

    def apply_artifact(self, syn_records_dict: dict, syn_records_ids: dict, external_ids: dict):
        """ Wipes all overlaps between id attributes of a group of security_records.
        """
        security_records = syn_records_dict['security_records']

        for id_attribute in self.id_attributes:
            # We check if there's any overlap in the column
            if len(security_records[id_attribute]) != len(security_records[id_attribute].unique()):
                value_counts = security_records[id_attribute].value_counts()
                for value in value_counts.keys():
                    if value != '':
                        value_count = value_counts.loc[value]
                        if value_count > 1:
                            rows_to_wipe = random.sample(
                                list(security_records[security_records[id_attribute] == value].index), value_count - 1)
                            security_records.loc[rows_to_wipe,id_attribute] = ''

        syn_records_dict['security_records'] = security_records

        return syn_records_dict, self.name, syn_records_ids, external_ids


class MultipleSecuritiesArtifactMultiSecurity(MultiSecurityDataArtifact):

    def __init__(self, modified_attributes: str, prob: float):
        artifact_name = 'multiple_securities_multi_security_artifact'.format(modified_attributes)
        super().__init__(artifact_name, prob)
        self.modified_attribute = modified_attributes
        self.id_attributes = ID_ATTRIBUTES
        security_types = copy.deepcopy(SECURITY_TYPES_DICT)
        security_types.pop('[equity]', None)
        self.generic_security_terms = security_types

    def _init_new_security(self, idx, syn_records_dict, generic_security_term_list, data_source_id, external_ids):
        issuer_records = syn_records_dict['company_records']
        security_records = syn_records_dict['security_records']
        issuer_record = issuer_records[issuer_records['data_source_id'] == data_source_id].squeeze()
        company_seed = syn_records_dict['company_seed']

        new_security = {}

        new_security_term = random.choice(generic_security_term_list)

        new_security['name'] = random.choice([new_security_term,
                                              company_seed['name'] + ' ' + random.choice(generic_security_term_list)])
        new_security['type'] = new_security_term
        # We set the primary_currency, data_source_id, external_id and issuer_id
        new_security['primary_currency'] = random.choice(list(security_records['primary_currency']))
        new_security['data_source_id'] = data_source_id
        new_security['issuer_id'] = issuer_record['external_id']

        new_security['external_id'], external_ids = get_new_external_id(data_source_id, external_ids)

        new_security['gen_id'] = str(syn_records_dict['gen_id']) + '_additional_{}'.format(idx)
        return new_security, external_ids

    def apply_artifact(self, syn_records_dict: dict, syn_records_ids: dict, external_ids: dict):
        """ Adds new securities (with different names and id attributes) to an issuer. The securities will have a
        different external_id but the same issuer_id as the initial records.
        """
        no_of_new_securities = random.randint(1, len(self.generic_security_terms))

        security_records = syn_records_dict['security_records']
        company_seed = syn_records_dict['company_seed']
        security_seed = syn_records_dict['security_seed']
        data_sources = list(security_records['data_source_id'].unique())

        for idx in range(no_of_new_securities):

            new_securities_list = []

            # For each new security we create, we randomly sample which data_sources will contain its records.
            no_of_data_sources = random.randint(1, len(data_sources))
            active_data_sources = random.sample(data_sources, no_of_data_sources)

            # We sample which kind of security the new security will be
            generic_security_term_list = random.choice(list(self.generic_security_terms.values()))

            for data_source in active_data_sources:
                new_security, external_ids = self._init_new_security(idx, syn_records_dict, generic_security_term_list,
                                                       data_source, external_ids)
                new_securities_list.append(new_security)

            for id_attribute in self.id_attributes:
                # We generate a new value for the chosen id attribute and check its uniqueness
                new_id_value = generate_id_attribute(company_seed, id_attribute, security_seed, syn_records_ids)
                while self._check_for_duplicates(syn_records_ids[id_attribute], new_attribute=new_id_value):
                    new_id_value = generate_id_attribute(company_seed, id_attribute, security_seed, syn_records_ids)
                syn_records_ids[id_attribute].append(new_id_value)
                # We assign the new_id_value to a random number of data_sources
                no_of_id_active_data_sources = random.randint(1, no_of_data_sources)
                id_active_data_sources = random.sample(
                    [new_security['data_source_id'] for new_security in new_securities_list],
                    no_of_id_active_data_sources)
                for data_source in id_active_data_sources:
                    for new_security in new_securities_list:
                        if new_security['data_source_id'] == data_source:
                            new_security[id_attribute] = new_id_value


            new_securities = pd.DataFrame(new_securities_list)

            # Finally, we set the 'inserted' and 'last_modified' attributes of the new records
            years_after_event = random.randint(0, 10)
            for data_source_id in active_data_sources:
                initial_timestamp = syn_records_dict['timestamp']
                event_timestamp = initial_timestamp
                while event_timestamp == initial_timestamp:
                    try:
                        event_timestamp = initial_timestamp.replace(year = initial_timestamp.year + years_after_event) + datetime.timedelta(
                            days=random.randint(0, 30),
                            hours=random.randint(0, 23),
                            minutes=random.randint(0, 59),
                            seconds=random.randint(0, 59))
                    except:
                        # We can get a ValueError if the initial timestamp is a Feb 29th
                        event_timestamp = initial_timestamp.replace(year = initial_timestamp.year + years_after_event, month = initial_timestamp.month
                                                                    + random.choice([-1,1])) + datetime.timedelta(
                            days=random.randint(0, 30),
                            hours=random.randint(0, 23),
                            minutes=random.randint(0, 59),
                            seconds=random.randint(0, 59))

                new_securities.loc[
                    new_securities['data_source_id'] == data_source_id, 'last_modified'] = random_close_date(
                    event_timestamp)
                new_securities.loc[
                    new_securities['data_source_id'] == data_source_id, 'inserted'] = random_close_date(
                    event_timestamp)


            security_records = pd.concat([security_records, new_securities], axis=0, ignore_index=True)

        syn_records_dict['security_records'] = security_records
        return syn_records_dict, self.name, syn_records_ids, external_ids


class MissingRecordArtifactMultiCompany(MultiSecurityDataArtifact):

    def __init__(self, modified_attributes: str, prob: float):
        artifact_name = 'missing_record_multi_security_artifact'
        super().__init__(artifact_name, prob)
        self.modified_attribute = modified_attributes

    def apply_artifact(self, syn_records_dict: dict, syn_records_ids: dict, external_ids:dict):
        """ Randomly deletes securities records from 1 or more SyntheticDataSources.
        """
        security_records = syn_records_dict['security_records']

        n_of_data_sources = random.randint(0, len(security_records))
        data_source_ids_list = list(security_records['data_source_id'])
        data_source_ids = random.sample(data_source_ids_list, n_of_data_sources)

        # We check that the selected syn_records are not involved in any recorded_merger data_drift event.
        recorded_merger_flag = False
        for data_drift_dict in syn_records_dict['data_drift_dicts']:
            if data_drift_dict['data_drift_type'] == 'recorded_merger':
                recorded_merger_flag = True

        for data_source_id in data_source_ids:
            if recorded_merger_flag == False:
                security_records.loc[security_records['data_source_id'] == data_source_id] = None

        security_records.dropna(axis=0, how='all', inplace=True)
        syn_records_dict['security_records'] = security_records

        return syn_records_dict, self.name, syn_records_ids, external_ids


class DataDriftSecurityDataArtifact(ABC):

    def __init__(self, artifact_name: str):
        self.name = artifact_name

    @abstractmethod
    def apply_artifact(self, syn_records_dict: dict, syn_records_dict_list: list, external_ids : dict):
        """Basic function of each DataDriftSecurityDataArtifact subclass. It takes a pandas Dataframe of securities
        which contain the following attributes:
            -security_name
            -ISIN (id)
            -CUSIP (id)
            -VALOR (id)
            -SEDOL (id)
            -primary_currency
            -data_source_id
            -external_id (id of the security assigned by the data source)
            -issuer_id (id of the issuing company)
        and accounts for the DataDriftDataArtifact of their issuers, modifying the id attributes of securities.

        """
        raise NotImplementedError("Should be implemented in the respective subclasses.")

class CorporateMergerSecurityDataArtifact(DataDriftSecurityDataArtifact):

    def __init__(self, modified_attributes: str):
        artifact_name = 'corporate_merger_security_data_drift_artifact'
        super().__init__(artifact_name)
        self.modified_attribute = modified_attributes
        self.id_attributes = ID_ATTRIBUTES

    def _get_security_record(self, security_ids, syn_records_dict_list: list):
        security_dict = next(
            (item for item in syn_records_dict_list if item['gen_id'] == security_ids[1]), None)

        security_records_df = security_dict['security_records']

        security_record = security_records_df[security_records_df['data_source_id'] == security_ids[0]]

        return security_record.squeeze()

    def apply_artifact(self, syn_records_dict: dict, syn_records_dict_list: list, external_ids : dict):
        """ Accounts for a CreateCorporateMergerDataArtifact of the issuer. We modify the id attributes of the
        involved records in order to account for the recorded_merger data drift.
        """
        security_records = syn_records_dict['security_records']

        for data_drift_dict in syn_records_dict['data_drift_dicts']:

            if data_drift_dict['data_drift_type'] == 'recorded_merger':
                # The 2nd tuple of the list is the (data_source_id, gen_id) of the merger company

                merger_company_record = self._get_security_record(data_drift_dict['affected_ids'][1],
                                                                  syn_records_dict_list)
                data_source_recorded_merger = data_drift_dict['affected_ids'][1][0]

                for id_attribute in self.id_attributes:
                    # The 1st tuple of the list is the (data_source_id, gen_id) of the merger company
                    # We set the values of the id attribute to those of the merger company.
                    merger_id = merger_company_record[id_attribute]
                    security_records.loc[security_records['data_source_id'] == data_source_recorded_merger, id_attribute] = merger_id

        syn_records_dict['security_records'] = security_records

        return syn_records_dict, self.name, external_ids
