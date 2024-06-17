import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

sys.path.append(os.getcwd())

from src.helpers.path_helper import *
from src.models.config import read_config_from_file
from src.helpers.logging_helper import setup_logging

setup_logging()

from src.models.pytorch_model import PyTorchModel


def get_pairwise_scores_args():
    parser = argparse.ArgumentParser(description='Calculate the average acc/recall/F1 pairwise scores in the test set of a set of training instances')

    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--experiment_names_list', action ='append', required=True, help='List of train experiment names to calculate pairwise scores for')
    parser.add_argument('--epochs_list', action ='append', required=True, help='List of epochs evaluated for each training run')

    args = parser.parse_args()

    return args

def get_model_args(experiment_name, epoch):

    parser = argparse.ArgumentParser(description='Carry out records matching experiment')
    # Manually add arguments to the parser
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--use_validation_set', action='store_true')
    args = parser.parse_args(['--experiment_name', experiment_name, '--epoch', f'{epoch}'])

    if os.path.isfile(experiment_config_path(args.experiment_name)):
        args.__dict__.update(read_config_from_file(args.experiment_name))

        # Disable updates to the model
        args.wandb = False
        args.save_config = False
        args.save_model = False

        # If an old configuration is loaded without this value, add the default
        try:
            args.nonmatch_ratio
        except AttributeError:
            args.nonmatch_ratio = 5

    else:
        raise FileNotFoundError(f"Config could not be found at {experiment_config_path(args.experiment_name)}")

    for argument in vars(args):
        logging.info("argument: {} =\t{}".format(str(argument).ljust(20), getattr(args, argument)))

    return args

def get_pairwise_scores(dataset_name, experiment_names_list, epochs_list):

    acc_list = []
    prec_list = []
    recall_list = []
    f1_list = []

    for i, training_run in enumerate(experiment_names_list):

        epoch = epochs_list[i]
        pred_path = experiment_file_path(experiment_name= training_run, file_name='distilbert__prediction_log__ep{}.csv'.format(epoch))

        if not os.path.isfile(pred_path):
            # Load model and predict
            model_args = get_model_args(training_run, epoch)
            model = PyTorchModel.load_from_args(model_args)

            model.test(epoch=epoch, global_step=0)

    for i, training_run in enumerate(experiment_names_list):

        epoch = epochs_list[i]
        pred_path = experiment_file_path(experiment_name= training_run, file_name='distilbert__prediction_log__ep{}.csv'.format(epoch))

        test_df = pd.read_csv(pred_path)

        acc = (test_df['predictions'] == test_df['labels']).mean()
        precision = (test_df['predictions'] & test_df['labels']).sum() / test_df['predictions'].sum()
        recall = (test_df['predictions'] & test_df['labels']).sum() / test_df['labels'].sum()
        f1 = 2 * (precision * recall) / (precision + recall)

        acc_list.append(acc)
        prec_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    # Calculate mean and std of scores

    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)

    prec_mean = np.mean(prec_list)
    prec_std = np.std(prec_list)

    recall_mean = np.mean(recall_list)
    recall_std = np.std(recall_list)

    f1_mean = np.mean(f1_list)
    f1_std = np.std(f1_list)

    print('-' * 50)
    print('PAIRWISE SCORES FOR DATASET: {}'.format(dataset_name))
    print('-' * 50)

    print('Accuracy: {:.4f} +- {:.4f}'.format(acc_mean, acc_std))
    print('Precision: {:.4f} +- {:.4f}'.format(prec_mean, prec_std))
    print('Recall: {:.4f} +- {:.4f}'.format(recall_mean, recall_std))
    print('F1: {:.4f} +- {:.4f}'.format(f1_mean, f1_std))

if __name__ == '__main__':
    # Run with: python -m scripts.get_pairwise_scores --help
    parser = get_pairwise_scores_args()

    get_pairwise_scores(parser.dataset_name, parser.experiment_names_list, parser.epochs_list)
