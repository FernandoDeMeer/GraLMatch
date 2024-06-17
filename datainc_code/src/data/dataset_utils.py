import itertools
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

from enum import Enum, IntEnum

from tqdm.auto import tqdm


# Enum to keep track and choose splitting methods for the
# train/test split in the dataset
#
class SplitMethod(Enum):
    RANDOM = 0
    PRE_SPLIT = 1


class MatchingLabel(IntEnum):
    POSITIVE_LABEL = 1
    NEGATIVE_LABEL = 0



def add_random_nonmatches(df: pd.DataFrame, dataset_type: str, match_matrix: csr_matrix, nonmatch_ratio, name):
    df = df.copy(deep=True)
    num_pos_matches = df[df.label == MatchingLabel.POSITIVE_LABEL.value].shape[0]
    num_neg_matches = df[df.label == MatchingLabel.NEGATIVE_LABEL.value].shape[0]
    num_nonmatches_expected = int(max(1, nonmatch_ratio - 1) * num_pos_matches)

    # We expected to generate new non-matches, otherwise there might be an issue
    # (we *have* to subtract the existing non-matches, otherwise the ratios are off)
    
    num_nonmatches_to_generate = (num_nonmatches_expected - num_neg_matches)
    assert num_nonmatches_to_generate > 0, "Your ratio between pos/neg is off, you have more negatives than expected."

    result = list()
    pbar = tqdm(total=num_nonmatches_to_generate + 1,
                desc=f'[{name}] Building random nonmatches for {dataset_type}...')

    # Iterate over match_matrix and randomly select new nonmatches (=0) to add.
    #
    # We adjust the :match_matrix after each pass, so that the matrix
    # is indicative of already chosen elements in previous generation steps.
    #
    # However, this while loop is an expensive operation, in particular the access of match_matrix[row, col]
    #
    while len(result) < num_nonmatches_to_generate:
        row_id = np.random.randint(0, match_matrix.shape[0] - 1)
        col_id = np.random.randint(0, match_matrix.shape[1] - 1)

        if match_matrix[row_id, col_id] == 0:
            result.append((row_id, col_id))
            result.append((col_id, row_id))
            pbar.update(2)

    pbar.close()

    # Generate dataset from result tuples
    df_nonmatches = pd.DataFrame(result, columns=['lid', 'rid'])
    df_nonmatches['label'] = MatchingLabel.NEGATIVE_LABEL.value

    df_processed = pd.concat([df, df_nonmatches]).reset_index(drop=True)
    return df_processed


# Builds the matches matrix from the given :df. if a max_id is provided, use that
# instead of calculating the max_id from the given :df (for consistency)
#
def _generate_sparse_match_matrix(df: pd.DataFrame, max_id=None):
    if not max_id:
        max_id = np.max(df[['lid', 'rid']].max().to_numpy())
    data = np.ones(len(df), dtype=np.bool_)
    rows = df['lid']
    cols = df['rid']

    # Instantiated with both row-col and col-row, to include
    # both variants (A,B) and (B,A) as "already taken"
    #
    sparse_data = np.concatenate([data, data])
    sparse_row_ind = np.concatenate([rows, cols])
    sparse_col_ind = np.concatenate([cols, rows])
    sparse_matrix = csr_matrix((sparse_data, (sparse_row_ind, sparse_col_ind)), shape=(max_id + 1, max_id + 1))

    return sparse_matrix, max_id


def add_random_non_matches_train_val_test(train_given, test_given, val_given, nonmatch_ratio, name):
    full_df = pd.concat([train_given, test_given, val_given])
    match_matrix, max_id = _generate_sparse_match_matrix(full_df)
    test_df = add_random_nonmatches(test_given, 'test', match_matrix, nonmatch_ratio=nonmatch_ratio, name=name)

    # Add matches generated during test set generation to the match matrix, so that they
    # are not chosen during train set generation
    #
    # This is fine to do in two steps for test and train separately, as the generated matches
    # are logged in the match_matrix and are per-definition not taken again.
    #
    test_matches, max_id = _generate_sparse_match_matrix(test_df, max_id=max_id)
    match_matrix += test_matches
    train_df = add_random_nonmatches(train_given, 'train', match_matrix, nonmatch_ratio=nonmatch_ratio, name=name)

    val_df = pd.DataFrame()
    if not val_given.empty:
        train_matches, _ = _generate_sparse_match_matrix(train_df, max_id=max_id)
        match_matrix += train_matches
        val_df = add_random_nonmatches(val_given, 'val', match_matrix, nonmatch_ratio=nonmatch_ratio, name=name)
    return test_df, train_df, val_df
