import random
import re
from abc import ABC, abstractmethod
import numpy as np
import copy

from tqdm.auto import tqdm

from nltk import word_tokenize
from nltk.corpus import stopwords
from transformers import *

import torch

from src.helpers.seed_helper import initialize_gpu_seed

from src.data.synthetic_data_generation.utils import *


class SingleCompanyDataArtifact(ABC):

    def __init__(self, artifact_name: str, artifact_params: dict):
        self.name = artifact_name
        self.params = artifact_params

    @abstractmethod
    def _apply_artifact(self, company_records: pd.DataFrame, data_source_id: str, modified_attributes: list):
        """Basic function of each SingleCompanyDataArtifact subclass. It modifies a company record pd.Series containing
        the following attributes:
            -company_name
            -city
            -region
            -country code
            -short description
        and applies the corresponding single record data artifact, returning the modified company record.
        """
        raise NotImplementedError("Should be implemented in the respective subclasses.")

    def apply_artifact(self, syn_records_dict: dict):
        applied_artifact = {}
        for data_source_id, params_dict in self.params.items():
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
                syn_records_dict['company_records'] = self._apply_artifact(syn_records_dict['company_records'],
                                                                           data_source_id, modified_attributes)
                applied_artifact[data_source_id] = self.name + '_'.join(modified_attributes)

        return syn_records_dict, applied_artifact


class MissingAttributeArtifactSingleCompany(SingleCompanyDataArtifact):

    def __init__(self, artifact_params: dict):
        artifact_name = 'missing_attribute_artifact_'
        super().__init__(artifact_name, artifact_params)

    def _apply_artifact(self, company_records, data_source_id, modified_attributes):
        """ Deletes attributes from company records.
        """
        company_records.loc[company_records['data_source_id'] == data_source_id, modified_attributes] = ''
        return company_records


class TextLengtheningArtifactSingleCompany(SingleCompanyDataArtifact):

    def __init__(self, artifact_params: dict):
        artifact_name = 'text_lengthening_artifact_'
        super().__init__(artifact_name, artifact_params)
        # Define a dictionary of substring replacements
        self.short_to_long_dict = {
            r'\bInc\.?\b': 'Incorporated',
            r'\bCorp\.?\b': 'Corporation',
            r'\bLtd\.?\b': 'Limited',
            r'\bLLC\b': 'Limited Liability Company',
            r'\bPLC\b': 'Public Limited Company',
            r'\bPCL\b': 'Public Company Limited',
            r'\bL\.P\.?\b': 'Limited Partnership',
            r'\bLLP\b': 'Limited Liability Partnership',
            r'\bLLLP\b': 'Limited Liability Limited Partnership',
            r'\bCo\.?\b': 'Company',
            r'\bAG\b': 'Aktiengesellschaft',
            r'\bSA\b': 'Société Anonyme',
            r'\bGmbH\b': 'Gesellschaft mit beschränkter Haftung',
            r'\bAB\b': 'Aktiebolag',
            r'\bBV\b': 'Besloten Vennootschap',
            r'\bNV\b': 'Naamloze Vennootschap',
            r'\bInc\b': 'Incorporated',
            r'\bCorp\b': 'Corporation',
            r'\bLTD\b': 'Limited',
            r'\bPC\b': 'Professional Corporation',
            r'\bPA\b': 'Professional Association',
            r'\bP\.C\.?\b': 'Professional Corporation',
            r'\bP\.A\.?\b': 'Professional Association',
            r'\bS\.A\.?\b': 'Sociedad Anónima',
            r'\bS\.r\.l\.?\b': 'Società a responsabilità limitata',
            r'\bS\.p\.A\.?\b': 'Società per azioni',
            r'\bS\.à\.r\.l\.?\b': 'Société à responsabilité limitée',
            r'\bS\.c\.r\.l\.?\b': 'Società cooperativa a responsabilità limitata',
            r'\bS\.n\.c\.?\b': 'Société en nom collectif',
            r'\bS\.e\.n\.c\.?\b': 'Société en commandite',
            r'\bPte\.?\b': 'Private',
            r'\bPrv\.?\b': 'Private',
            r'\bABO\b': 'Agricultural Business Office',
            r'\bAEC\b': 'Architects Engineers Contractors',
            r'\bP\.?C\.?\b': 'Professional Corporation',
            r'\bP\.?A\.?\b': 'Professional Association',
            r'\bP\.?L\.?\b': 'Public Limited',
            r'\bP\.?L\.?C\.?\b': 'Public Limited Company',
            r'\bP\.?S\.?\b': 'Professional Service',
            r'\bP\.?S\.C\.?\b': 'Professional Service Corporation', }

    def _apply_artifact(self, company_records, data_source_id, modified_attributes):
        """
        Lengthens substrings if they match with a key of the short_to_long_dict and replaces them with
        the respective value.
        """
        try:
            company_record = company_records.loc[company_records['data_source_id'] == data_source_id].squeeze()
            company_name = company_record['name']
            # Iterate over the short to long dictionary and apply each replacement
            for pattern, replacement in self.short_to_long_dict.items():
                company_name = re.sub(pattern, replacement, company_name, flags=re.IGNORECASE)
            company_records.loc[company_records['data_source_id'] == data_source_id, modified_attributes] = company_name
        except:
            pass

        return company_records


class TextShorteningArtifactSingleCompany(SingleCompanyDataArtifact):

    def __init__(self, artifact_params: dict):
        artifact_name = 'text_shortening_artifact_'
        super().__init__(artifact_name, artifact_params)
        self.long_to_short_dict = {
            'Incorporated': 'Inc.',
            'Corporation': 'Corp.',
            'Limited Liability Company': 'LLC',
            'Limited': 'Ltd.',
            'Company': 'Co.',
            'Technologies': 'Tech',
            'Capital': 'Cap',
            'Systems': 'Sys',
            'Venture Capital': 'VC',
            'Venture Capital Fund': 'VCF',
            'Ventures': 'VC',
            'Gesellschaft mit beschränkter Haftung': 'GmbH',
            'Aktiengesellschaft': 'AG',
            'Kommanditgesellschaft': 'KG',
            'Besloten Vennootschap': 'B.V.',
            'Société Anonyme': 'S.A.',
            'Sociedad Anónima': 'S.A.',
            'Société à responsabilité limitée': 'SARL',
            'Societas Europaea': 'SE',
            'Société privée à responsabilité limitée': 'SPRL',
            'Società a responsabilità limitata': 'SRL',
            'Société Anonyme/NV': 'SA/NV',
            'Anpartsselskab': 'ApS',
            'Aktieselskab': 'A/S',
            'Allmennaksjeselskap': 'ASA',
            'Bolagsaktiebolag': 'BA',
            'Handelsbolag': 'HB',
            'Aktiebolag': 'AB',
            'Osakeyhtiö': 'Oy',
            'Limitada': 'Ltda.',
            'Limited Liability Partnership': 'LLP',
            'Professional Limited Liability Company': 'PLLC',
            'Professional Corporation': 'PC',
            'Limited Partnership': 'LP',
            'Sociedad Limitada': 'S.L.',
            'Société par actions simplifiée': 'S.A.S.',
            'Société Civile': 'S.C.',
            'Société en Commandite par Actions': 'S.C.A.',
            'Société Civile Professionnelle': 'S.C.P.',
            'Società Europea': 'S.E.',
            'Sociedad Limitada Unipersonal': 'S.L.U.',
            'Société en Nom Collectif': 'S.N.',
            'Spoločnosť s ručením obmedzeným': 'S.r.o.',
            'Société à responsabilité limitée unipersonnelle': 'S.r.l.U.',
            'Sociedad de Responsabilidad Limitada': 'S.s.',
            'Sociedad Anónima Unipersonal': 'S.A.U.',
            'Sociedad en Comandita Simple': 'S.C.S.',
            'Sociedad en Comandita por Acciones': 'S.C.S.A.',
            'Sociedad en Comandita Simple Limitada': 'S.C.S.L.',
            'Société en Développement Agricole': 'S.D.A.',
            'Telecom': 'Tel',
            'Media': 'Med',
            'Solutions': 'Sol'
        }

    def _apply_artifact(self, company_records, data_source_id, modified_attributes):
        """
        Shortens substrings if they match with a key of the long_to_short_dict and replaces them with
        the respective value.
        """
        try:
            company_record = company_records.loc[company_records['data_source_id'] == data_source_id].squeeze()
            company_name = company_record['name']
            pattern = re.compile(r'\b(' + '|'.join(map(re.escape, self.long_to_short_dict.keys())) + r')\b')
            company_name = pattern.sub(lambda x: self.long_to_short_dict[x.group()], company_name)
            company_records.loc[company_records['data_source_id'] == data_source_id, modified_attributes] = company_name
        except:
            pass

        return company_records


class DeletePunctAndStopWordsArtifactSingleCompany(SingleCompanyDataArtifact):

    def __init__(self, artifact_params: dict):
        artifact_name = 'delete_punct_and_stopwords_artifact_'
        super().__init__(artifact_name, artifact_params)
        self.stop_words = set(stopwords.words('english'))
        self.punct_words = set(string.punctuation)

    def _apply_artifact(self, company_records, data_source_id, modified_attributes):
        try:
            company_record = company_records.loc[company_records['data_source_id'] == data_source_id].squeeze()
            company_name = company_record['name']
            normalized = [w for w in word_tokenize(company_name)
                          if (w not in self.punct_words)
                          and (w not in self.stop_words)]
            if len(normalized) > 0:
                company_records.loc[
                    company_records['data_source_id'] == data_source_id, modified_attributes] = ' '.join(normalized)
        except:
            pass

        return company_records


class SplitUpperCaseInWordArtifactSingleCompany(SingleCompanyDataArtifact):

    def __init__(self, artifact_params: dict):
        artifact_name = 'split_uppercase_artifact_'
        super().__init__(artifact_name, artifact_params)

    def _apply_artifact(self, company_records, data_source_id, modified_attributes):
        try:
            company_record = company_records.loc[company_records['data_source_id'] == data_source_id].squeeze()
            company_name = company_record['name']
            substrings = re.sub(r'(?<![A-Z\W])(?=[A-Z])', ' ', company_name).split()
            new_name = ' '.join(substrings)

            if company_name != new_name:
                company_records.loc[company_records['data_source_id'] == data_source_id, modified_attributes] = new_name
                # Substitute all mentions of the old name in the short description with the new name
                case_insensitive_new_name = re.compile(re.escape(company_name), re.IGNORECASE)
                company_records.loc[company_records[
                                        'data_source_id'] == data_source_id, 'short_description'] = case_insensitive_new_name.sub(
                    new_name,
                    company_record['short_description'])
        except:
            pass

        return company_records


class MergeUpperCaseInWordArtifactSingleCompany(SingleCompanyDataArtifact):

    def __init__(self, artifact_params: dict):
        artifact_name = 'merge_uppercase_artifact_'
        super().__init__(artifact_name, artifact_params)

    def _apply_artifact(self, company_records, data_source_id, modified_attributes):
        try:
            company_record = company_records.loc[company_records['data_source_id'] == data_source_id].squeeze()
            company_name = company_record['name']
            substrings = company_name.split()

            capital_letters_list = [substring[0].isupper() for substring in substrings]
            if all(capital_letters_list):
                company_records.loc[company_records['data_source_id'] == data_source_id, modified_attributes] = ''.join(
                    substrings)
        except:
            pass

        return company_records


class MergeDotComWordArtifactSingleCompany(SingleCompanyDataArtifact):

    def __init__(self, artifact_params: dict):
        artifact_name = 'merge_dot_com_artifact_'
        super().__init__(artifact_name, artifact_params)

    def _apply_artifact(self, company_records, data_source_id, modified_attributes):
        try:
            company_record = company_records.loc[company_records['data_source_id'] == data_source_id].squeeze()
            company_name = company_record['name']
            substrings = company_name.split()

            if any(['.com' in substrings[-1]]):
                company_records.loc[company_records['data_source_id'] == data_source_id, modified_attributes] = ''.join(
                    substrings)
        except:
            pass

        return company_records


class AcronymNameArtifactSingleCompany(SingleCompanyDataArtifact):

    def __init__(self, artifact_params: dict):
        artifact_name = 'acronym_artifact_'
        super().__init__(artifact_name, artifact_params)

    def _apply_artifact(self, company_records, data_source_id, modified_attributes):
        try:
            company_record = company_records.loc[company_records['data_source_id'] == data_source_id].squeeze()
            company_name = company_record['name']
            substrings = company_name.split()

            capital_letters_list = [substring[0].isupper() for substring in substrings]
            if all(capital_letters_list) and len(capital_letters_list) > 3:
                company_records.loc[company_records['data_source_id'] == data_source_id, modified_attributes] = ''.join(
                    word[0] for word in substrings)
        except:
            pass

        return company_records


class CountryCodeFormatChangeArtifactSingleCompany(SingleCompanyDataArtifact):

    def __init__(self, artifact_params: dict):
        artifact_name = 'country_code_format_change_artifact_'
        super().__init__(artifact_name, artifact_params)
        self.countries_states_cities = countries_states_cities_dict

    def _apply_artifact(self, company_records, data_source_id, modified_attributes):
        try:
            company_record = company_records.loc[company_records['data_source_id'] == data_source_id].squeeze()

            country_dict = next(
                (item for item in self.countries_states_cities if item['iso3'] == company_record['country_code']), None)
            # We change the country code with a different format
            if country_dict is not None:
                company_records.loc[
                    company_records['data_source_id'] == data_source_id, 'country_code'] = random.choice(
                    [country_dict['iso2'], country_dict['name']])
        except:
            pass

        return company_records


class RegionFormatChangeArtifactSingleCompany(SingleCompanyDataArtifact):

    def __init__(self, artifact_params: dict):
        artifact_name = 'region_format_change_artifact_'
        super().__init__(artifact_name, artifact_params)
        self.countries_states_cities = countries_states_cities_dict

    def _apply_artifact(self, company_records, data_source_id, modified_attributes):
        try:
            company_record = company_records.loc[company_records['data_source_id'] == data_source_id].squeeze()
            country_dict = next(
                (item for item in self.countries_states_cities if item['iso3'] == company_record['country_code']), None)
            # We change the region with a different format
            if country_dict is not None:
                state_dict = next((item for item in country_dict['states'] if item['name'] in company_record['region']),
                                  None)
                if state_dict is not None:
                    company_records.loc[company_records['data_source_id'] == data_source_id, 'region'] = random.choice(
                        [state_dict['name'], state_dict['state_code']])

        except:
            pass

        return company_records


class RegionCountrySwapArtifactSingleCompany(SingleCompanyDataArtifact):

    def __init__(self, artifact_params: dict):
        artifact_name = 'region_&_country_swap_artifact_'
        super().__init__(artifact_name, artifact_params)

    def _apply_artifact(self, company_records, data_source_id, modified_attributes):
        try:
            company_record = company_records.loc[company_records['data_source_id'] == data_source_id].squeeze()

            company_country = company_record['country_code']
            company_region = company_record['region']

            company_records.loc[company_records['data_source_id'] == data_source_id, 'country_code'] = company_region
            company_records.loc[company_records['data_source_id'] == data_source_id, 'region'] = company_country
        except:
            pass

        return company_records


class RegionOverwriteArtifactSingleCompany(SingleCompanyDataArtifact):

    def __init__(self, artifact_params: dict):
        artifact_name = 'region_overwrite_artifact'
        super().__init__(artifact_name, artifact_params)

    def _apply_artifact(self, company_records, data_source_id, modified_attributes):
        try:
            company_record = company_records.loc[company_records['data_source_id'] == data_source_id].squeeze()

            company_country = company_record['country_code']
            company_records.loc[company_records['data_source_id'] == data_source_id, 'region'] = company_country
        except:
            pass

        return company_records


class CityOverwriteArtifactSingleCompany(SingleCompanyDataArtifact):

    def __init__(self, artifact_params: dict):
        artifact_name = 'city_overwrite_artifact'
        super().__init__(artifact_name, artifact_params)

    def _apply_artifact(self, company_records, data_source_id, modified_attributes):
        try:
            company_record = company_records.loc[company_records['data_source_id'] == data_source_id].squeeze()

            company_region = company_record['region']
            company_records.loc[company_records['data_source_id'] == data_source_id, 'city'] = company_region
        except:
            pass

        return company_records


class MultiCompanyDataArtifact(ABC):

    def __init__(self, artifact_name: str, prob: float):
        self.name = artifact_name
        self.prob = prob

    @abstractmethod
    def apply_artifact(self, syn_records_dict: dict):
        """Basic function of each MultiRecordDataArtifact subclass. It takes a synthetic_records_dict with a
         Dataframe of company records which contain the following attributes:
            -company_name
            -city
            -region
            -country code
            -short description
        and applies the corresponding multi-record data artifact.
        """
        raise NotImplementedError("Should be implemented in the respective subclasses.")


class MissingAttributeArtifactMultiCompany(MultiCompanyDataArtifact):

    def __init__(self, modified_attributes: str, prob: float):
        artifact_name = 'missing_{}_multi_record_artifact'.format(modified_attributes)
        super().__init__(artifact_name, prob)
        self.modified_attribute = modified_attributes

    def apply_artifact(self, syn_records_dict: dict):
        """ Deletes the corresponding attribute from all company records.
        """
        syn_records_dict['company_records'][self.modified_attribute] = ''
        return syn_records_dict, self.name


class InsertCorporateTermArtifactMultiCompany(MultiCompanyDataArtifact):

    def __init__(self, modified_attributes: str, prob: float):
        artifact_name = 'insert_corporate_term_{}_multi_record_artifact'.format(modified_attributes)
        super().__init__(artifact_name, prob)
        self.modified_attribute = modified_attributes
        self.corporate_structures_dict = {
            "Holding Corp": ["Holding Company", "Holding Co.", "Holding Corp", "HOLD. CO.", "HC", "Parent Company",
                             "Parent Corp", "Parent Co."],
            "Subsidiary Corp": ["Subsidiary Company", "Subsidiary Co.", "Subsidiary", "Subs Corp.", "Subs. Co.", "SC"],
            "Limited Liability Company": ["Ltd. Liability Co.", "Ltd. Liab. Corp", "Ltd. LCorp", "LLCorp", "LLCo.",
                                          "LLC"],
            "General Partnership": ["Gen. Ptshp", "GPartnership", "GP"],
            "Limited Partnership": ["Ltd. Ptshp", "LPartnership", "LP"],
            "Limited Liability Partnership": ["Ltd. Liability Partnership", "Ltd.LP", "LLP"],
            "Nonprofit Corporation": ["Nonprofit Co.", "Nonprofit Corp.", "Non-profit Corp", "Non-profit Co.",
                                      "Non-profit", "NC", ],
            "Professional Corporation": ["Professional Corp", "Prof. Corp", "Prof. Co.", "PC"],
            "Cooperative": ["Co-op", "Coop."],
            "Joint Venture": ["JV", "Joint Vent."],
            "Benefit Corporation": ["Benefit Corp", "Benefit Co.", "Ben. Co.", "B-Corp"],
            'Incorporated': ['Inc.', 'INC'],
            'Corporation': ['Corp.', 'CORP'],
            'Limited': ['Ltd.', 'LTD'],
            'Company': ['Co.', 'CO'],
            'Technologies': ['TECH', 'Tech.'],
            'Systems': ['Sys.', 'SYS'],
            'Ventures': ['Vent.', 'Vents.'],
            'Platforms': ['Pltf.', 'Plt.']
        }

    def apply_artifact(self, syn_records_dict: dict):
        """ Inserts a random corporate structure term (and its variations) into all company records if the company
        has a single-word name (in order to avoid repeating terms or having nonsensical combinations).
        """
        company_records = syn_records_dict['company_records']
        company_name = syn_records_dict['company_seed']['name']
        substrings = company_name.split()

        if len(substrings) < 2:
            random_corporate_term_item = random.choice(list(self.corporate_structures_dict.items()))
            random_corporate_terms_list = [random_corporate_term_item[0]] + random_corporate_term_item[1]

            for index, company_record in company_records.iterrows():
                data_source_id = company_record['data_source_id']
                # We first choose a random key from the corporate_structures_dict
                old_name = company_record['name']
                new_name = company_record['name'] + ' ' + random.choice(random_corporate_terms_list)
                try:
                    company_records.loc[company_records['data_source_id'] == data_source_id, 'name'] = new_name
                    # Substitute all mentions of the old name in the short description with the new name
                    case_insensitive_new_name = re.compile(re.escape(old_name), re.IGNORECASE)
                    company_records.loc[company_records['data_source_id'] == data_source_id, 'short_description'] = case_insensitive_new_name.sub(
                        new_name, company_record['short_description'])
                except:
                    pass

        return syn_records_dict, self.name


class MissingRecordArtifactMultiCompany(MultiCompanyDataArtifact):

    def __init__(self, modified_attributes: str, prob: float, missing_per_by_data_source):
        artifact_name = 'missing_record_multi_record_artifact'
        super().__init__(artifact_name, prob)
        self.modified_attribute = modified_attributes
        self.missing_per_by_data_source = missing_per_by_data_source

    def apply_artifact(self, syn_records_dict: dict):
        """ Randomly deletes records from 1 or more SyntheticDataSources
        """
        company_records = syn_records_dict['company_records']
        data_sources_to_remove = [data_source_id for data_source_id in company_records['data_source_id'] if
                                  random.uniform(0, 1) < self.missing_per_by_data_source[data_source_id]]

        # We drop the chosen rows
        syn_records_dict['company_records'][
            syn_records_dict['company_records']['data_source_id'].isin(data_sources_to_remove)] = np.nan
        syn_records_dict['company_records'].dropna(axis=0, how='all', inplace=True)

        return syn_records_dict, self.name


class MultipleRegionsDataArtifactMultiCompany(MultiCompanyDataArtifact):

    def __init__(self, modified_attributes: str, prob: float):
        artifact_name = 'multiple_regions_multi_record_artifact'
        super().__init__(artifact_name, prob)
        self.modified_attribute = modified_attributes
        self.countries_states_cities = countries_states_cities_dict

    def apply_artifact(self, syn_records_dict: dict):
        """ Randomly chooses a new (region, city) pair and assigns it to 1 or more records with varying degrees of
        coverage. This simulates a change of company location within their original country.
        """
        company_records = syn_records_dict['company_records']
        n_of_data_sources = random.randint(0,
                                           len(company_records) - 1)  # At least 1 record should keep the original location
        data_source_ids_list = list(company_records['data_source_id'])
        data_source_ids = random.sample(data_source_ids_list, n_of_data_sources)

        country_dict = next(
            (item for item in self.countries_states_cities if
             item['iso3'] == syn_records_dict['company_seed']['country_code']),
            None)
        # We now look for random region of the country to assign to the modified records
        if country_dict is not None:
            if len(country_dict['states']) > 0:
                random_region = random.choice(country_dict['states'])
                if len(random_region['cities']) > 0:
                    random_city = random.choice(random_region['cities'])

        if country_dict is not None:
            if len(country_dict['states']) > 0:

                for data_source_id in data_source_ids:
                    company_records.loc[company_records['data_source_id'] == data_source_id, 'region'] = ''
                    company_records.loc[company_records['data_source_id'] == data_source_id, 'city'] = ''
                    if random.uniform(0, 1) < 0.5:
                        company_records.loc[company_records['data_source_id'] == data_source_id, 'region'] = \
                            random_region['name']
                    else:
                        company_records.loc[company_records['data_source_id'] == data_source_id, 'region'] = \
                            random_region['state_code']

                    if random.uniform(0, 1) < 0.5 and len(random_region['cities']) > 0:
                        company_records.loc[company_records['data_source_id'] == data_source_id, 'city'] = \
                        random_city[
                            'name']

        # Finally, we modify the 'last_modified' attribute of the modified records
        years_after_event = random.randint(1, 10)
        for data_source_id in data_source_ids:
            initial_timestamp = syn_records_dict['timestamp']
            event_timestamp = initial_timestamp
            while event_timestamp == initial_timestamp:
                try:
                    event_timestamp = initial_timestamp.replace(
                        year=initial_timestamp.year + years_after_event) + datetime.timedelta(
                        days=random.randint(0, 30),
                        hours=random.randint(0, 23),
                        minutes=random.randint(0, 59),
                        seconds=random.randint(0, 59))
                except:
                    # We can get a ValueError if the initial timestamp is a Feb 29th
                    event_timestamp = initial_timestamp.replace(year=initial_timestamp.year + years_after_event,
                                                                month=initial_timestamp.month
                                                                      + random.choice([-1, 1])) + datetime.timedelta(
                        days=random.randint(0, 30),
                        hours=random.randint(0, 23),
                        minutes=random.randint(0, 59),
                        seconds=random.randint(0, 59))

            company_records.loc[
                company_records['data_source_id'] == data_source_id, 'last_modified'] = random_close_date(
                event_timestamp)

        return syn_records_dict, self.name


class MultipleCountriesDataArtifactMultiCompany(MultiCompanyDataArtifact):

    def __init__(self, modified_attributes: str, prob: float):
        artifact_name = 'multiple_countries_multi_record_artifact'
        super().__init__(artifact_name, prob)
        self.modified_attribute = modified_attributes
        self.countries_states_cities = countries_states_cities_dict

    def apply_artifact(self, syn_records_dict: dict):
        """ Randomly chooses a new (country, region, city) tuple and assigns it to 1 or more records with varying
        degrees of  coverage. This simulates a change of company location outside their original country.
        """
        company_records = syn_records_dict['company_records']
        n_of_data_sources = random.randint(0,
                                           len(company_records) - 1)  # At least 1 record should keep the original location
        data_source_ids_list = list(company_records['data_source_id'])
        data_source_ids = random.sample(data_source_ids_list, n_of_data_sources)

        random_country = random.choice(self.countries_states_cities)
        # We first look for random country to assign to the modified records
        if len(random_country['states']) > 0:
            random_region = random.choice(random_country['states'])
            if len(random_region['cities']) > 0:
                random_city = random.choice(random_region['cities'])

        if 'random_region' in locals():

            for data_source_id in data_source_ids:
                company_records.loc[company_records['data_source_id'] == data_source_id, 'country_code'] = ''
                company_records.loc[company_records['data_source_id'] == data_source_id, 'region'] = ''
                company_records.loc[company_records['data_source_id'] == data_source_id, 'city'] = ''

                country_format = random.uniform(0, 1)
                if country_format < 1 / 3:
                    company_records.loc[company_records['data_source_id'] == data_source_id, 'country_code'] = \
                        random_country['name']
                elif country_format > 1 / 3 and country_format < 2 / 3:
                    company_records.loc[company_records['data_source_id'] == data_source_id, 'country_code'] = \
                        random_country['iso2']
                else:
                    company_records.loc[company_records['data_source_id'] == data_source_id, 'country_code'] = \
                        random_country['iso3']

                if random.uniform(0, 1) < 0.5:
                    company_records.loc[company_records['data_source_id'] == data_source_id, 'region'] = \
                    random_region[
                        'name']
                else:
                    company_records.loc[company_records['data_source_id'] == data_source_id, 'region'] = \
                    random_region[
                        'state_code']

                if random.uniform(0, 1) < 0.5 and 'random_city' in locals():
                    company_records.loc[company_records['data_source_id'] == data_source_id, 'city'] = random_city[
                        'name']

        # Finally, we modify the 'last_modified' attribute of the modified records
        years_after_event = random.randint(1, 10)
        for data_source_id in data_source_ids:
            initial_timestamp = syn_records_dict['timestamp']
            event_timestamp = initial_timestamp
            while event_timestamp == initial_timestamp:
                try:
                    event_timestamp = initial_timestamp.replace(
                        year=initial_timestamp.year + years_after_event) + datetime.timedelta(
                        days=random.randint(0, 30),
                        hours=random.randint(0, 23),
                        minutes=random.randint(0, 59),
                        seconds=random.randint(0, 59))
                except:
                    # We can get a ValueError if the initial timestamp is a Feb 29th
                    event_timestamp = initial_timestamp.replace(year=initial_timestamp.year + years_after_event,
                                                                month=initial_timestamp.month
                                                                      + random.choice([-1, 1])) + datetime.timedelta(
                        days=random.randint(0, 30),
                        hours=random.randint(0, 23),
                        minutes=random.randint(0, 59),
                        seconds=random.randint(0, 59))

            company_records.loc[
                company_records['data_source_id'] == data_source_id, 'last_modified'] = random_close_date(
                event_timestamp)

        return syn_records_dict, self.name


class DataDriftDataArtifact(ABC):
    """ Class that implements Data Artifacts caused by real-world corporate processes ( mergers,acquisitions, bankruptcies,
    IPOs etc.) taking place over time.

    """

    def __init__(self, artifact_name: str, prob: float):
        self.name = artifact_name
        self.prob = prob

    @abstractmethod
    def apply_artifact(self, syn_records_dict: dict, investors_df: pd.DataFrame, syn_records_dicts_list: list, external_ids:dict ):
        """Basic function of each DataDriftDataArtifact subclass. It takes a synthetic_records_dict with a
         pandas Dataframe of company records which contain the following attributes:
            -company_name
            -city
            -region
            -country code
            -short description
        and applies the corresponding data artifact.
        """
        raise NotImplementedError("Should be implemented in the respective subclasses.")


class CreateCorporateAcquisitionDataArtifact(DataDriftDataArtifact):

    def __init__(self, modified_attributes: str, prob: float):
        artifact_name = 'create_corporate_acquisition_artifact'
        super().__init__(artifact_name, prob)
        self.modified_attributes = modified_attributes

    def apply_artifact(self, syn_records_dict: dict, investors_df: pd.DataFrame, syn_records_dicts_list: list, external_ids:dict ):
        """" Overwrites a company record(s) of 1 or more SyntheticDataSources with the attributes of
         an acquiring company (taken from a set of investors) simulating a corporate acquisition process"""
        company_records = syn_records_dict['company_records']
        if len(investors_df) > 1:
            acquiring_company = investors_df.sample(n=1)
            investors_df = investors_df.drop(acquiring_company.index, axis=0)
            acquiring_company = acquiring_company.squeeze()

            if len(company_records) > 2:
                n_of_data_sources = random.randint(1,
                                                   len(company_records) - 1)  # At least 1 original record should remain
            elif len(company_records) == 2:
                n_of_data_sources = 1
            else:
                n_of_data_sources = 0

            data_source_ids_list = list(company_records['data_source_id'])
            # Not all data sources may properly record the acquisition, so we randomly choose which ones will
            data_sources_to_modify = random.sample(data_source_ids_list, n_of_data_sources)

            for data_source_id in data_sources_to_modify:
                syn_records_dict['data_drift_dicts'].append({
                    'affected_ids': [(data_source_id, syn_records_dict['gen_id'])],
                    'data_drift_type': 'recorded_acquisition_overwrite'})

                acquiring_company['data_source_id'] = data_source_id
                for unchanged_data_source_id in data_source_ids_list:
                    if unchanged_data_source_id not in data_sources_to_modify:
                        syn_records_dict['data_drift_dicts'].append({
                            'affected_ids': [(data_source_id, syn_records_dict['gen_id']),
                                             (unchanged_data_source_id, syn_records_dict['gen_id'])],
                            'data_drift_type': 'unrecorded_acquisition_overwrite'})

                for column, val in acquiring_company.iteritems():
                    # We always swap the name but only randomly swap the other attributes
                    if column == 'name':
                        company_records.loc[company_records['data_source_id'] == data_source_id, column] = \
                            acquiring_company[column]
                    # We don't reassign descriptions (otherwise the only records with descriptions in some
                    # data_sources will be those affected by this artifact)
                    elif column != 'short_description':
                        if random.uniform(0, 1) < 0.5:
                            company_records.loc[company_records['data_source_id'] == data_source_id, column] = \
                                acquiring_company[column]

            # Finally, we modify the 'last_modified' attribute of the modified records
            years_after_event = random.randint(1, 10)
            for data_source_id in data_sources_to_modify:
                initial_timestamp = syn_records_dict['timestamp']
                event_timestamp = initial_timestamp
                while event_timestamp == initial_timestamp:
                    try:
                        event_timestamp = initial_timestamp.replace(
                            year=initial_timestamp.year + years_after_event) + datetime.timedelta(
                            days=random.randint(0, 30),
                            hours=random.randint(0, 23),
                            minutes=random.randint(0, 59),
                            seconds=random.randint(0, 59))
                    except:
                        # We can get a ValueError if the initial timestamp is a Feb 29th
                        event_timestamp = initial_timestamp.replace(year=initial_timestamp.year + years_after_event,
                                                                    month=initial_timestamp.month
                                                                          + random.choice([-1, 1])) + datetime.timedelta(
                            days=random.randint(0, 30),
                            hours=random.randint(0, 23),
                            minutes=random.randint(0, 59),
                            seconds=random.randint(0, 59))

                company_records.loc[
                    company_records['data_source_id'] == data_source_id, 'last_modified'] = random_close_date(
                    event_timestamp)

        return syn_records_dict, investors_df, self.name, external_ids

class CreateCorporateMergerDataArtifact(DataDriftDataArtifact):

    def __init__(self, modified_attributes: str, prob: float):
        artifact_name = 'create_corporate_merger_artifact'
        super().__init__(artifact_name, prob)
        self.modified_attributes = modified_attributes

    def apply_artifact(self, syn_records_dict: dict, investors_df: pd.DataFrame, syn_records_dicts_list: list, external_ids:dict):
        """" Records the match with a merger company from a different syn_records_dict simulating a corporate
        merger process. When generating securities, their securities will share some id_attribute(s)."""
        # We randomly select the merger
        merger_dict = copy.deepcopy(random.choice(syn_records_dicts_list))
        merger_records = merger_dict['company_records']
        company_records = syn_records_dict['company_records']

        if len(company_records) > 2:
            n_of_data_sources = random.randint(1, len(company_records) - 1)
        elif len(company_records) == 2:
            n_of_data_sources = 1
        else:
            n_of_data_sources = 0

        data_source_ids_list = list(company_records['data_source_id'])

        # Not all data sources may properly record the merger, so we randomly choose which ones will
        data_source_ids_to_record = random.sample(data_source_ids_list, n_of_data_sources)

        for data_source_id in data_source_ids_to_record:

            merger_company_data_source_equiv = merger_records[
                (merger_dict['company_records'] == data_source_id)].squeeze()

            # The data_source chosen may not have a record for the merger company, in this case we do not record any ids
            if len(merger_company_data_source_equiv) > 0:
                syn_records_dict['data_drift_dicts'].append({
                    'affected_ids': [(data_source_id, syn_records_dict['gen_id']),
                                     (data_source_id, merger_dict['gen_id'])],
                    'data_drift_type': 'recorded_merger'})
                for unchanged_data_source_id in data_source_ids_list:
                    if unchanged_data_source_id not in data_source_ids_to_record:
                        syn_records_dict['data_drift_dicts'].append({
                            'affected_ids': [(data_source_id, syn_records_dict['gen_id']),
                                             (unchanged_data_source_id,
                                              merger_dict['gen_id'])],
                            'data_drift_type': 'unrecorded_merger'})

        # Finally, we modify the 'last_modified' attribute of the modified records
        years_after_event = random.randint(1, 10)
        for data_source_id in data_source_ids_to_record:
            initial_timestamp = syn_records_dict['timestamp']
            event_timestamp = initial_timestamp
            while event_timestamp == initial_timestamp:
                try:
                    event_timestamp = initial_timestamp.replace(
                        year=initial_timestamp.year + years_after_event) + datetime.timedelta(
                        days=random.randint(0, 30),
                        hours=random.randint(0, 23),
                        minutes=random.randint(0, 59),
                        seconds=random.randint(0, 59))
                except:
                    # We can get a ValueError if the initial timestamp is a Feb 29th
                    event_timestamp = initial_timestamp.replace(year=initial_timestamp.year + years_after_event,
                                                                month=initial_timestamp.month
                                                                      + random.choice([-1, 1])) + datetime.timedelta(
                        days=random.randint(0, 30),
                        hours=random.randint(0, 23),
                        minutes=random.randint(0, 59),
                        seconds=random.randint(0, 59))

            company_records.loc[
                company_records['data_source_id'] == data_source_id, 'last_modified'] = random_close_date(
                event_timestamp)

        return syn_records_dict, investors_df, self.name, external_ids


class BatchCompanyDataArtifact(ABC):

    def __init__(self, artifact_name: str, prob: float):
        self.name = artifact_name
        self.prob = prob

    @abstractmethod
    def apply_artifact(self, company_seed: pd.Series):
        """Basic function of each BatchCompanyDataArtifact subclass. It takes a list of synthetic record dicts
        and applies the corresponding batch data artifact.
        """
        raise NotImplementedError("Should be implemented in the respective subclasses.")


class BatchParaphraseAttributeArtifact(BatchCompanyDataArtifact):
    def __init__(self, modified_attribute: str, prob: float, gpu_seed: int, data_source_id):
        artifact_name = f'paraphrase_{modified_attribute}_artifact'
        super().__init__(artifact_name, prob)

        self.data_source_id = data_source_id
        self.modified_attribute = modified_attribute
        self.torch_device, _ = initialize_gpu_seed(
            seed=gpu_seed)
        model_name = "tuner007/pegasus_paraphrase"
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(self.torch_device)
        self.tokenizer = PegasusTokenizerFast.from_pretrained(model_name)
        self.batch_size = 32

    def _get_paraphrased_text(self, batch, num_return_sequences=1, num_beams=5):
        # tokenize text
        inputs = self.tokenizer(batch, truncation=True, padding="longest", return_tensors="pt").to(self.torch_device)

        # generate the paraphrased text
        outputs = self.model.generate(
            **inputs,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            max_length=self.tokenizer.max_len_single_sentence
        )

        if self.torch_device == "cuda":
            del inputs
            torch.cuda.empty_cache()

        decoded_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        if self.torch_device == "cuda":
            del outputs
            torch.cuda.empty_cache()

        return decoded_output

    def apply_artifact(self, syn_records_dicts_list):
        company_records_df = build_df_from_records(syn_records_dicts_list, 'company_records')
        inputs = company_records_df[company_records_df['data_source_id'] == self.data_source_id][
            'short_description'].tolist()

        batches = [inputs[i:i + self.batch_size] for i in range(0, len(inputs), self.batch_size)]

        outputs = []

        for batch in tqdm(batches, total=len(batches), leave=True, colour='green',
                          desc='Applying ParaphraseAttributeArtifactSingleCompany'):
            clean_batch = [input for input in batch if input != '']
            if len(clean_batch) > 0:
                output = self._get_paraphrased_text(clean_batch)
                outputs.append(output)

        flat_outputs = [item for sublist in outputs for item in sublist]
        company_records_df.loc[(company_records_df['data_source_id'] == self.data_source_id) &
                               (company_records_df['short_description'] != ''), 'short_description'] = flat_outputs

        for gen_id, company_records in company_records_df.groupby('gen_id'):
            syn_records_dicts_list[int(gen_id)]['company_records'][self.modified_attribute] = \
                company_records[self.modified_attribute]

        return syn_records_dicts_list
