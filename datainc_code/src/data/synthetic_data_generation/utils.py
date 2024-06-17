import pandas as pd
import time
import datetime
import random
import json
import string

from src.helpers.path_helper import *
from src.helpers.logging_helper import setup_logging

setup_logging()

# Load the countries, states & cities dict
f = open(dataset_raw_file_path(os.path.join('synthetic_data', 'countries_states_cities.json')),
         encoding="utf8")
countries_states_cities_dict = json.load(f)

def build_df_from_records(syn_records_list, records_type):

    syn_records_dfs = [synthetic_records_dict[records_type] for synthetic_records_dict in syn_records_list if records_type in synthetic_records_dict.keys()]

    df = pd.concat(syn_records_dfs)

    return df

def str_time_prop(start, end, time_format, prop):
    """Get a time at a proportion of a range of two formatted times.

    start and end should be strings specifying times formatted in the
    given format (strftime-style), giving an interval [start, end].
    prop specifies how a proportion of the interval to be taken after
    start.  The returned time will be in the specified format.
    """

    stime = time.mktime(time.strptime(start, time_format))
    etime = time.mktime(time.strptime(end, time_format))

    ptime = stime + prop * (etime - stime)

    return time.strftime(time_format, time.localtime(ptime))


def random_date(start, end, prop):
    return datetime.datetime.strptime(str_time_prop(start, end, "%d/%m/%Y %H:%M:%S", prop), "%d/%m/%Y %H:%M:%S")

def random_close_date(date):
    date = date + (-1 ** random.randint(1, 2)) * datetime.timedelta(
        days=random.randint(0, 6),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59))
    return date

def get_new_external_id(data_source_id, external_ids):

    if external_ids[data_source_id] is None:
        new_external_id = random.randint(1, 100000000)
        external_ids[data_source_id] = [random.randint(1, 100000000)]
    else:
        new_external_id = random.randint(1, 100000000)
        while new_external_id in external_ids[data_source_id]:
            new_external_id = random.randint(1, 100000000)

        external_ids[data_source_id].append(new_external_id)

    return str(new_external_id), external_ids

def insert_letters(string, index, letters):
    return string[:index] + letters + string[index:]

def generate_id_attribute(company_seed: pd.Series, id_attribute: str, security_seed: pd.Series, syn_records_ids: dict):
    """ Generate an id attribute for the synthetic securities issued by a group of company records.
    """

    """
    From Wikipedia:
    ---
    ISINs consist of two alphabetic characters, which are the ISO 3166-1 alpha-2 code for the issuing country,
    nine alpha-numeric characters [...] and one numerical check digit.  Since existing national numbering schemes
    administered by the various NNAs form the basis for ISINs, the methodology for assignment is not consistent
    across agencies globally.
    ---

    """
    if id_attribute == 'ISIN':
        country_code = company_seed['country_code']
        country_dict = next(
            (item for item in countries_states_cities_dict if item['iso3'] == country_code), None)
        if country_dict is not None:
            iso_code = country_dict['iso2']
        else:
            country_dict = random.choice(countries_states_cities_dict)
            iso_code = country_dict['iso2']

        number_of_letters = random.randint(0, 4)

        letters = ''.join(random.choice(string.ascii_uppercase) for _ in range(number_of_letters))

        random_alphanum_code = ''.join(random.choice(string.digits) for _ in range(10 - number_of_letters))

        letters_position = random.randint(1, len(random_alphanum_code))

        random_alphanum_code = insert_letters(random_alphanum_code, letters_position, letters)

        gen_id = iso_code + random_alphanum_code
    """
    From https://www.cusip.com/identifiers.html:
    ---
    Derived from the Committee on Uniform Security Identification Procedures, CUSIPs are 9-character identifiers
    that capture an issue’s important differentiating characteristics for issuers and their financial instruments
    in the U.S. and Canada.
    ---
    """
    if id_attribute == 'CUSIP':

        country_code = company_seed['country_code']

        if country_code in ['USA', 'CAN']:
            # We extract the CUSIP from the ISIN
            gen_id = security_seed['ISIN'][2:-1]
            if gen_id in syn_records_ids['CUSIP']:
                gen_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(9))
        else:
            gen_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(9))

    """

    From https://www.six-group.com/en/products-services/financial-information/data-standards.html
    ---
    SIX is the official securities numbering agency in Switzerland, Liechtenstein and Belgium [...]. The Valor is
    incorporated in the Swiss, Belgium and Liechtenstein ISIN.
    ---

    """
    if id_attribute == 'VALOR':

        country_code = company_seed['country_code']

        if country_code in ['CHE', 'LIE', 'BEL']:
            # We extract the VALOR from the ISIN
            gen_id = security_seed['ISIN'][2:-1]
            if gen_id in syn_records_ids['VALOR']:
                gen_id = str(100000000 + random.randint(0, 25000000))
        else:
            gen_id = str(100000000 + random.randint(0, 25000000))

    """
    From https://www.lseg.com/content/dam/lseg/en_us/documents/sedol/sedol-masterfile-faqs-170321.pdf
    ---
    SEDOL codes are seven-character identification codes, consisting of six alphanumeric characters and a
    trailing check digit.[...] SEDOLs are embedded within UK ISIN codes and therefore are an integral part of ISIN
    identifier data [...] SEDOL codes do not contain embedded intelligence. The codes consist of two parts,
    a six-place alphanumeric code, and a trailing check digit that is a weighted sum of the
    first six characters. The SEDOL code will be seven characters alphanumeric with the
    following structure
    """
    if id_attribute == 'SEDOL':

        country_code = company_seed['country_code']

        if country_code in ['GBR']:
            # We extract the SEDOL from the ISIN
            gen_id = security_seed['ISIN'][2:-3]
            if gen_id in syn_records_ids['SEDOL']:
                gen_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
        else:
            gen_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))

    return gen_id

def get_artifacts(artifacts_params_dict, artifact_class):

    artifacts_subclasses_dict = {}
    for artifact_subclass in artifact_class.__subclasses__():
        artifacts_subclasses_dict[
            artifact_subclass.__name__] = artifact_subclass


    artifacts = []
    for artifact_name, params in artifacts_params_dict.items():
        # params can either be a list or a single dict (one or many instances of the same SingleRecordDataArtifact)
        if isinstance(params, list):
            for artifact_params in params:
                artifacts.append(
                    artifacts_subclasses_dict[artifact_name](**artifact_params))
        else:
            artifacts.append(
                artifacts_subclasses_dict[artifact_name](**params))

    return artifacts
