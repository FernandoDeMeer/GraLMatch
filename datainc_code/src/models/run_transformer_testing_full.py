import os
import sys

sys.path.append(os.getcwd())
from src.models.pytorch_model import PyTorchModel
from src.helpers.seed_helper import initialize_gpu_seed
from src.models.config import read_arguments_test
from src.helpers.logging_helper import setup_logging
from src.helpers.path_helper import *
from src.data.full_data_utils import create_full_security_candidates_from_set, create_full_syn_security_candidates_from_set


setup_logging()


def main(args):
    initialize_gpu_seed(args.seed)

    checkpoint_suffix = '__epoch' + str(args.epoch)
    if args.epoch == 0:
        checkpoint_suffix += '__zeroshot'

    file_name = "".join([args.model_name, checkpoint_suffix, '.pt'])
    checkpoint_path = experiment_file_path(args.experiment_name, file_name)

    model = PyTorchModel.load_from_checkpoint(args, checkpoint_path)
    candidate_df = create_full_syn_security_candidates_from_set(model.dataset)



    # candidate_idx_df = candidate_df.drop(columns=['match_type'])
    # model.test_data_loader.dataset.idx_df = candidate_idx_df
    # model.test(args.epoch)


    # import code; code.interact(local=dict(globals(), **locals()))


if __name__ == '__main__':
    args = read_arguments_test()
    main(args)
