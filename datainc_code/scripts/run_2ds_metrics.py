import os
import sys
import argparse

sys.path.append(os.getcwd())
from src.helpers.path_helper import *
from src.helpers.logging_helper import setup_logging

setup_logging()

from src.models.config import read_arguments_test
import pandas as pd


def main(args):
    file_path = experiment_file_path(experiment_name=args.experiment_name,
                                     file_name=f'distilbert__prediction_log__ep{args.epoch}.csv')

    pred_df = pd.read_csv(file_path)

    tp, fp, tn, fn = 0, 0, 0, 0

    for index, row in pred_df.iterrows():
        if row['predictions'] == 1:
            if row['labels'] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if row['labels'] == 0:
                tn += 1
            else:
                fn += 1

    precision = round(tp / (tp + fp), 4)
    recall = round(tp / (tp + fn), 4)
    f1 = round(2 * ((precision * recall) / (precision + recall)), 4)

    # print scores

    print('-' * 80)
    print(args.experiment_name)
    print('-' * 80)

    print('Number of True Positives: ' + str(tp))
    print('Number of False Positives: ' + str(fp))
    print('Number of False Negatives: ' + str(fn))
    print('Number of True Negatives: ' + str(tn))

    print('Precision : ' + str(precision))
    print('Recall: ' + str(recall))
    print('F1 score: ' + str(f1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calculates precision, recall and f1-score for a given prediction log')

    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--epoch', type=int, required=True)

    args = parser.parse_args()
    main(args)
