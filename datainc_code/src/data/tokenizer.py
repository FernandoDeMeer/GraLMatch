import os
import sys
sys.path.append(os.getcwd())

from transformers import AutoTokenizer
import torch

from src.helpers.logging_helper import setup_logging
import logging
from src.models.config import Config
import copy
import swifter


import pandas as pd
from abc import ABC, abstractmethod


setup_logging()


class DataIncTokenizer(ABC):
    def __init__(self, model_name, do_lower_case, max_seq_length):
        self.model_name = model_name
        self.do_lower_case = do_lower_case
        self.max_seq_length = max_seq_length
        self.tokenizer = self._setup_tokenizer()

    def __len__(self):
        return self.tokenizer.__len__()

    def tokenize(self, text: str):
        return self.tokenizer.tokenize(text)

    @abstractmethod
    def generate_sample(self, l_txt: str, r_txt: str, label: int = 0):
        '''
        Takes the tokenized left and right text inputs and generates a sequence for the transformer
        '''
        raise NotImplementedError("Should be implemented in the respective subclasses.")

    def _setup_tokenizer(self):
        '''
        Loads the corresponding tokenizer according to the configuration above
        '''
        pretrained_model = Config.MODELS[self.model_name].pretrained_model

        # Use the AutoTokenizer that defaults to a FastTokenizer instance
        # using a Rust-based (=much faster) implementation.
        # (see https://huggingface.co/course/chapter6/3 and
        # https://huggingface.co/docs/transformers/v4.26.0/en/model_doc/auto#transformers.AutoTokenizer)
        #
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model, do_lower_case=self.do_lower_case)

        return tokenizer


    def tokenize_df(self, df_original: pd.DataFrame) -> pd.DataFrame:
        df = df_original.copy()

        logging.info(f'Start tokenizing.')

        tokenized_df = df \
            .swifter \
            .allow_dask_on_strings(enable=True) \
            .progress_bar(desc='Concatenating rows...') \
            .apply(self._get_concat_fn(), axis=1) \
            .to_frame(name='concat_data')

        tokenized_df['tokenized'] = tokenized_df \
            .swifter \
            .progress_bar(desc='Tokenizing rows...') \
            .apply(lambda row: self.tokenizer.tokenize(row['concat_data']), axis=1)

        logging.info('Done tokenizing.')

        return tokenized_df, df

    def _truncate_sequences(self, tokens_a, tokens_b, max_length):
        '''
        This is a simple heuristic which will always truncate the longer sequence
        one token at a time. This makes more sense than truncating an equal percent
        of tokens from each, since if one sequence is very short then each token
        that's truncated likely contains more information than a longer sequence.
        '''
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def build_sequence(self, tokens_a: list, tokens_b: list, label,
                       cls_token_at_end=False,
                       cls_token='[CLS]',
                       cls_token_segment_id=0,
                       sep_token='[SEP]',
                       sep_token_extra=False,
                       pad_on_left=False,
                       pad_token=0,
                       pad_token_segment_id=0,
                       sequence_a_segment_id=0,
                       sequence_b_segment_id=1,
                       mask_padding_with_zero=True):
        '''
        Builds a sequence for the transformer

        Parameters
        ------------
        tokens_a: list
            1st list of tokens
        tokens_b: list
            2nd list of tokens
        label: any
            Label of the pair (l,r)
        cls_token_at_end: boolean
            Location of the CLS token (False: BERT/XLM, True: XLNET/GPT)
        cls_token: str
            What token to use for CLS
        cls_token_segment_id: int
            Segment ID for the CLS Token (Bert: 0, XLNET: 2)
        sep_token: str
            What token to use for SEP
        sep_token_extra: boolean
            Only used for RoBERTa
        pad_on_left: boolean
            Whether to pad left or right side
        pad_token: int
            What token to use for padding
        pad_token_segment_id: int
            Segment ID for padding
        sequence_a_segment_id: int
            Segment ID for left side
        sequence_b_segment_id: int
            Segment ID for right side
        mask_padding_with_zero: boolean
            Whether to mask padding as zeros or ones
        '''
        # Ensure that the given lists are not changed
        tokens_a = copy.deepcopy(tokens_a)
        tokens_b = copy.deepcopy(tokens_b)

        special_tokens_count = 4 if sep_token_extra else 3
        self._truncate_sequences(tokens_a, tokens_b, self.max_seq_length - special_tokens_count)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        tokens += tokens_b + [sep_token]
        segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        label_id = torch.tensor([label], dtype=torch.long)

        return (input_ids, input_mask, segment_ids, label_id)

    def _get_concat_fn(self):
        '''
        Provides the method for concatenating a single row in the dataframe
        '''
        def concat_fn(row):
            def string_conversion(cell):
                col_name, val = cell
                return '' if pd.isna(val) else str(val)

            return " ".join(map(string_conversion, row[1:].items())).strip()

        return concat_fn