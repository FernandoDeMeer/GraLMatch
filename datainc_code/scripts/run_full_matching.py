import os
import sys
import argparse
import logging

sys.path.append(os.getcwd())
from src.models.pytorch_model import PyTorchModel
from src.models.config import update_args_with_config
from src.helpers.seed_helper import initialize_gpu_seed
from src.models.config import read_arguments_test
from src.helpers.logging_helper import setup_logging
from src.helpers.path_helper import *
from src.helpers.matcher_helper import matchers_dict

setup_logging()



def read_arguments_matching():
    parser = argparse.ArgumentParser(description='Test model with following arguments')
    # Arguments from read_arguments_test() of src.models.config
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--company_matching', type=str, required=False, default=None)
    parser.add_argument('--use_validation_set', action='store_true')
    # Arguments for the matcher:
    parser.add_argument('--matcher', type=str, required=True, choices=list(matchers_dict.keys()))
    # Argument for the threshold used in the matching
    parser.add_argument('--threshold', type=float, required=False, default=0.999)

    args = parser.parse_args()
    args = update_args_with_config(args.experiment_name, args)

    for argument in vars(args):
        logging.info("argument: {} =\t{}".format(str(argument).ljust(20), getattr(args, argument)))

    return args


def main(args):
    initialize_gpu_seed(args.seed)

    checkpoint_suffix = '__epoch' + str(args.epoch)
    if args.epoch == 0:
        checkpoint_suffix += '__zeroshot'

    file_name = "".join([args.model_name, checkpoint_suffix, '.pt'])
    checkpoint_path = experiment_file_path(args.experiment_name, file_name)

    model = PyTorchModel.load_from_checkpoint(args, checkpoint_path)

    matcher = matchers_dict[args.matcher](model=model)

    matcher.run_matching(args)

if __name__ == '__main__':
    args = read_arguments_matching()
    main(args)
