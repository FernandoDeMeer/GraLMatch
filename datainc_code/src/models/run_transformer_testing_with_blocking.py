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

    # We load the unblocked pairs from the csv created by PyJedAI
    unblocked_pairs = pd.read_csv('notebooks/blocking/small_securities_blocked_type.csv', delimiter=' ', names=['lid', 'rid'])
    unblocked_pairs['label'] = 0

    # We modify the test Dataloader of the model to go through the new set of pairs during the testing rather than the original.
    test_ds = PytorchDataset(model_name=model.dataset.name, idx_df=unblocked_pairs, data_df=model.dataset.get_tokenized_data(),
                             tokenizer=model.dataset.tokenizer, max_seq_length=model.args.max_seq_length)

    test_dl = DataLoader(test_ds, shuffle=False, batch_size=model.args.batch_size)

    model.test_data_loader = test_dl
    model._reset_prediction_buffer()

    model.test(args.epoch)


if __name__ == '__main__':
    args = read_arguments_test()
    main(args)
