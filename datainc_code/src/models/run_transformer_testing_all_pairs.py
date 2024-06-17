from torch.utils.data import DataLoader
from src.data.dataset import PytorchDataset
from src.models.pytorch_model import PyTorchModel
from src.helpers.seed_helper import initialize_gpu_seed
from src.models.config import read_arguments_test
from src.helpers.logging_helper import setup_logging
from src.helpers.path_helper import *
import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.getcwd())


setup_logging()


def main(args):
    initialize_gpu_seed(args.model_seed)

    checkpoint_suffix = '__epoch' + str(args.epoch)
    if args.epoch == 0:
        checkpoint_suffix += '__zeroshot'

    file_name = "".join([args.model_name, checkpoint_suffix, '.pt'])
    checkpoint_path = experiment_file_path(args.experiment_name, file_name)

    model = PyTorchModel.load_from_checkpoint(args, checkpoint_path)

    # We enumerate all the pairs in the dataset but making sure to semantically block all pairs with the same data_source_id
    raw_df = model.dataset.get_raw_df()

    # all_pairs = pd.DataFrame(columns=['lid','rid','label'])
    all_pairs = np.empty(shape=(0, 3))
    for entry in raw_df.index:
        non_matching_data_source_id_entries = raw_df.iloc[entry:][raw_df[entry:]
                                                                  ['data_source_id'] != raw_df.iloc[entry]['data_source_id']]
        array_to_append = np.zeros((non_matching_data_source_id_entries.shape[0], 3))
        array_to_append[:, 0] = entry
        array_to_append[:, 1] = non_matching_data_source_id_entries.index
        all_pairs = np.concatenate((all_pairs, array_to_append), axis=0)

    all_pairs = pd.DataFrame(data=all_pairs, columns=['lid', 'rid', 'label'])
    # We modify the test Dataloader of the model to go through the new set of pairs during the testing rather than the original.
    test_ds = PytorchDataset(model_name=model.dataset.name, idx_df=all_pairs, data_df=model.dataset.get_tokenized_data(),
                             tokenizer=model.dataset.tokenizer, max_seq_length=model.args.max_seq_length)

    test_dl = DataLoader(test_ds, shuffle=False, batch_size=model.args.batch_size)

    model.test_data_loader = test_dl
    model._reset_prediction_buffer()

    model.test(args.epoch)


if __name__ == '__main__':
    args = read_arguments_test()
    main(args)
