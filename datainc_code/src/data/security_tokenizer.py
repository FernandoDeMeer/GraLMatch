from src.data.tokenizer import DataIncTokenizer

import itertools
from enum import Enum
import pandas as pd
from tqdm.auto import tqdm
import copy

# Ensure that the dataset is recreated when these tokens are adjusted.
# In addition, models have to be retrained for changes here as any change to the custom
# tokens is a *breaking* change!
#
# keys are the column name, values the start and end tokens
#
CUSTOM_ID_TOKENS = {
    'active_isin': ('[isin]', '[/isin]'),
    'active_cusip': ('[cusip]', '[/cusip]'),
    'active_valor': ('[valor]', '[/valor]'),
    'active_sedol': ('[sedol]', '[/sedol]'),
    'active_ric': ('[ric]', '[/ric]'),
    'external_id': ('[external_id]', '[/external_id]'),
    'issuer_id': ('[issuer_id]', '[/issuer_id]')
}

# Ensure that the dataset is recreated when these tokens are adjusted.
# In addition, models have to be retrained for changes here as any change to the custom
# tokens is a *breaking* change!
#
# keys are the start token of the column attribute
#
CUSTOM_HEUR_TOKENS = {
    '[isin]': '[identical_isin]',
    '[cusip]': '[identical_cusip]',
    '[valor]': '[identical_valor]',
    '[sedol]': '[identical_sedol]',
    '[ric]': '[identical_ric]',
    '[external_id]': '[identical_external_id]',
    '[issuer_id]': '[identical_issuer_id]'
}


# Ensure that the dataset is recreated when these tokens are adjusted.
# In addition, models have to be retrained for changes here as any change to the custom
# tokens is a *breaking* change!
#
class SecurityType(str, Enum):
    RIGHT = '[right]'
    EQUITY = '[equity]'
    UNIT = '[unit]'
    BOND = '[bond]'
    ETF = '[etf]'
    FUND = '[fund]'

SEC_TYPE_TOKENS = [
    SecurityType.RIGHT, SecurityType.EQUITY, SecurityType.UNIT,
    SecurityType.BOND, SecurityType.ETF, SecurityType.FUND
]

# the lists are sorted by len (desc)
SECURITY_TYPES = {
    '[right]': ['Equity Rights'],
    '[equity]': ['Shares/Units with shares/Particip. Cert.', 'Preferred Equity/Derivative Unit',
               'Equity Convertible Preference', 'Equity Depositary Receipts', 'Equity Depositary Receipt',
               'Equity Depository Receipt',
               'Equity/Preferred Unit', 'Equity Derivatives', 'Equity Preference', 'US Dep Rect (ADR)',
               'Preferred Equity', 'Preference Share', 'Preferred Issue', 'Preferred Stock', 'Ordinary Shares',
               'registered shs', 'Ordinary Share', 'Equity Shares', 'Common Shares', 'Common Equity',
               'Equity Issue', 'Common Stock', 'Ord Shs', 'Equity', 'Units'],
    '[unit]': ['Dept/Equity Composite Units', 'Investment trust unit/share', 'Equity/Derivative Unit',
             'Dept/Derivative Unit', 'Dept/Preferred Unit', 'Units'],
    '[bond]': ['Bond'],
    '[etf]': ['ETF'],
    '[fund]': ['Hedge Fund', 'Mutual Fund', 'FUND']
}


class SecurityTokenizer(DataIncTokenizer):
    def __init__(self, model_name, do_lower_case, max_seq_length):
        super().__init__(model_name, do_lower_case, max_seq_length)
        self.add_custom_tokens(self.tokenizer)

    def add_custom_tokens(self, tokenizer):
        tokenizer.add_tokens(
            list(itertools.chain(*CUSTOM_ID_TOKENS.values()))
            + list(CUSTOM_HEUR_TOKENS.values())
            + list(map(lambda x: x.value, SEC_TYPE_TOKENS))
        )

    def tokenize_df(self, df_original: pd.DataFrame) -> pd.DataFrame:
        df = df_original.copy()
        df = self.add_custom_identifier_tokens(df)
        df = self.add_security_type_tokens(df)

        return super().tokenize_df(df)

    def add_custom_identifier_tokens(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Adds custom delimiter tokens to ID attributes of the securities data
        '''
        df = df.copy()
        for key in tqdm(CUSTOM_ID_TOKENS.keys(), total=len(CUSTOM_ID_TOKENS.keys()), desc='Adding custom identifier tokens...'):
            try:
                # Add the delimiter tokens to the corresponding attribute:
                start_token, end_token = CUSTOM_ID_TOKENS[key]
                df[key.lower()] = ['{} {} {}'.format(start_token, element, end_token) for element in list(df[key.lower()])]
                # Replace nan values with an empty string so that we don't introduce un-informative tokens in the input
                # (int64 NaN are returned as <NA> to string)
                df.loc[df[key.lower()] == '{} {} {}'.format(start_token, 'nan', end_token), key.lower()] = ''
                df.loc[df[key.lower()] == '{} {} {}'.format(start_token, '<NA>', end_token), key.lower()] = ''
            except KeyError:
                print(f'\nKeyError: key {key} does not exist in dataframe, continuing...')
                continue

        return df

    def add_security_type_tokens(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Replaces certain words in given string with keywords (EQUITY, RIGHT, etc.)
        Returns:
            tuple: (changed_string, changed_word) if the string was changed else (original_string, None)
        '''
        def replace_strs(x: str):
            str_to_search = str(x).lower()
            for key, value in SECURITY_TYPES.items():
                for stopword in value:
                    res = str_to_search.find(stopword.lower())
                    if res != -1:
                        str_to_search = str_to_search.replace(stopword.lower(), key)
                        return str_to_search, key

            return x, None

        # go through all rows; change the type
        # if it changed: change the name and remove the added token (else change the name)
        #
        for i, row in tqdm(df.iterrows(), total=df.shape[0], desc='Adding security type tokens...'):
            type_replace = replace_strs(row['type'])
            name_replace = replace_strs(row['name'])
            if type_replace != row['type']:
                df.at[i, 'type'] = type_replace[0]
                df.at[i, 'name'] = name_replace[0].replace(name_replace[1], '') if name_replace[1] is not None else row[
                    'name']
            else:
                df.at[i, 'name'] = name_replace[0]

        return df

    def add_identical_identifier_tokens(self, l_txt, r_txt):
        custom_id_tokens = list(CUSTOM_ID_TOKENS.values())

        # Ensure that the given lists are not changed
        l_txt = copy.deepcopy(l_txt)
        r_txt = copy.deepcopy(r_txt)

        # Add the heuristic tokens based on matching id attributes
        for start_token, end_token in custom_id_tokens:
            l_id_present, r_id_present = True, True

            try:
                l_start_idx = l_txt.index(start_token)
                l_end_idx = l_txt.index(end_token)
            except ValueError:
                l_id_present = False

            try:
                r_start_idx = r_txt.index(start_token)
                r_end_idx = r_txt.index(end_token)
            except ValueError:
                r_id_present = False

            if l_id_present and r_id_present:
                # Both contain the tokens, check if they are the same and if so,
                # add the identical_<ident> token to both texts at the beginning
                # (to avoid truncation removing it)
                #
                l_identifier = l_txt[l_start_idx + 1:l_end_idx]
                r_identifier = r_txt[r_start_idx + 1:r_end_idx]

                if l_identifier == r_identifier:
                    identical_token = CUSTOM_HEUR_TOKENS[start_token]
                    l_txt.insert(0, identical_token)
                    r_txt.insert(0, identical_token)

            # Used to remove the "[ISIN] US123... [/ISIN]" part of the sequence,
            # since we currently are only interested in having [IDENTICAL_ISIN] tokens (or nothing at all)
            #
            if l_id_present:
                del l_txt[l_start_idx:l_end_idx + 1]

            if r_id_present:
                del r_txt[r_start_idx:r_end_idx + 1]

        return l_txt, r_txt

    def generate_sample(self, l_txt: str, r_txt: str, label: int = 0):
        l_txt, r_txt = self.add_identical_identifier_tokens(l_txt, r_txt)

        seq = super().build_sequence(l_txt, r_txt, label)
        return seq
