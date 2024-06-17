import argparse
import datetime
import json
import logging
import os
from dataclasses import dataclass

from src.helpers.logging_helper import setup_logging

setup_logging()

from pytorch_transformers import (BertConfig, BertForSequenceClassification,
                                  BertTokenizer, DistilBertConfig,
                                  DistilBertForSequenceClassification,
                                  DistilBertTokenizer)

from src.helpers.path_helper import (experiment_config_path,
                                     file_exists_or_create)

# XLNetTokenizer, \
# XLNetForSequenceClassification, XLNetConfig, XLMForSequenceClassification, XLMConfig, XLMTokenizer, \
# RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig, DistilBertConfig, \



DEFAULT_NONMATCH_RATIO = 5
DEFAULT_SEED = 44
DEFAULT_MODEL_SEED = 44
DEFAULT_SEQ_LENGTH = 128
DEFAULT_TRAIN_FRAC = 0.8


@dataclass
class LanguageModelConfig:
    model_class: object
    model_config: object
    pretrained_model: str
    tokenizer: object


class Config():
    MODELS = {
        'bert': LanguageModelConfig(BertForSequenceClassification, BertConfig, 'bert-base-uncased', BertTokenizer),
        # 'xlnet': LanguageModelConfig(XLNetConfig, XLNetConfig, 'xxx', XLNetTokenizer),
        # 'xlm': LanguageModelConfig(XLMConfig, XLMConfig, 'xxx', XLMTokenizer),
        # 'roberta': LanguageModelConfig(RobertaConfig, RobertaConfig, 'xxx', RobertaTokenizer),
        'distilbert': LanguageModelConfig(DistilBertForSequenceClassification, DistilBertConfig,
                                          'distilbert-base-uncased', DistilBertTokenizer),
    }

    DATASETS = {
        'synthetic_companies': os.path.join('synthetic_data', 'seed_0', 'synthetic_companies_dataset_seed_0_size_868254_sorted.csv'),
        'synthetic_companies_small': os.path.join('synthetic_data', 'seed_0', 'synthetic_companies_dataset_seed_0_size_868254_sorted.csv'),
        'synthetic_securities_small': os.path.join('synthetic_data', 'seed_0', 'synthetic_securities_dataset_seed_0_size_984942_sorted.csv'),
        'synthetic_securities': os.path.join('synthetic_data', 'seed_0', 'synthetic_securities_dataset_seed_0_size_984942_sorted.csv'),
        'wdc': os.path.join('wdc_80pair', 'wdc_80pair.csv'),
    }


def write_config_to_file(args):
    config_path = experiment_config_path(args.experiment_name)
    file_exists_or_create(config_path)

    with open(config_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    logging.info(f'\tSuccessfully saved configuration at {config_path}')


def read_config_from_file(experiment_name: str):
    config_path = experiment_config_path(experiment_name)

    with open(config_path, 'r') as f:
        args = json.load(f)

    logging.info(f'\tSuccessfully loaded configuration for {experiment_name}')
    return args


def read_arguments_train():
    parser = argparse.ArgumentParser(description='Run training with following arguments')

    parser.add_argument('--dataset_name', default='synthetic_companies',
                        choices=Config.DATASETS.keys(),
                        help='Choose a dataset to be processed.')
    parser.add_argument('--model_name_or_path', default='distilbert-base-uncased', type=str)
    parser.add_argument('--model_name', default='distilbert', choices=Config.MODELS.keys())
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--max_seq_length', default=DEFAULT_SEQ_LENGTH, type=int)
    parser.add_argument('--do_lower_case', action='store_true', default=True)
    parser.add_argument('--nonmatch_ratio', default=DEFAULT_NONMATCH_RATIO, type=int)
    parser.add_argument('--seed', default=DEFAULT_SEED, type=int)
    parser.add_argument('--model_seed', default=DEFAULT_MODEL_SEED, type=int)
    parser.add_argument('--use_validation_set', action='store_true')

    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--adam_eps', default=1e-8, type=float)
    parser.add_argument('--warmup_steps', default=0, type=int)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--use_softmax_layer', action='store_true', default=True)

    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--save_config', action='store_true')
    parser.add_argument('--wandb', action='store_true', default=True)
    parser.add_argument('--no_wandb', dest='wandb', action='store_false')

    args = parser.parse_args()

    if args.save_model:
        args.save_config = True

    # Provide default experiment name unless one is given
    if not args.experiment_name:
        model_name_parts = []
        model_name_parts.append(datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S'))
        # Add separation of company/security models by identifier
        if 'securities' in args.dataset_name:
            model_name_parts.append('securities')
        elif 'companies' in args.dataset_name:
            model_name_parts.append('companies')
        args.experiment_name = "__".join(model_name_parts)

    else:
        # Load from saved config if name matches
        if os.path.isfile(experiment_config_path(args.experiment_name)):
            found_config = read_config_from_file(args.experiment_name)

            # Compatibility for old models before
            # using a softmax layer in the transformer
            #
            try:
                found_config['use_softmax_layer']
            except AttributeError:
                found_config['use_softmax_layer'] = False

            # Use the value of the seed parameter for the
            # model seed, when it is not present
            # #
            try:
                found_config['model_seed']
            except AttributeError:
                found_config['model_seed'] = found_config['seed']

            args.__dict__.update(found_config)

    if args.save_config:
        write_config_to_file(args)

    logging.info("*** parsed configuration from command line and combine with constants ***")

    for argument in vars(args):
        logging.info("argument: {} =\t{}".format(str(argument).ljust(20), getattr(args, argument)))

    return args


def read_arguments_test():
    parser = argparse.ArgumentParser(description='Test model with following arguments')

    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--offset', type=int, required=False, default=0,
                        help="""In case a task gets stuck, crashes or is stopped, can be used to run the same task again but
                        continue from an advanced position. Depending on the task, is used as start_index = 0+offset,
                        though indices here are often the indices of the traversed dataframe (not necessarily 1,2,3,4
                        without gaps)
                        """)
    parser.add_argument('--use_validation_set', action='store_true')


    args = parser.parse_args()
    args = update_args_with_config(args.experiment_name, args)

    for argument in vars(args):
        logging.info("argument: {} =\t{}".format(str(argument).ljust(20), getattr(args, argument)))

    return args

def update_args_with_config(experiment_name: str, args: argparse.Namespace = argparse.Namespace()) -> argparse.Namespace:
    if os.path.isfile(experiment_config_path(experiment_name)):
        found_config = read_config_from_file(experiment_name)

        # Use the value of the seed parameter for the
        # model seed, when it is not present
        # #
        try:
            found_config['model_seed']
        except KeyError:
            found_config['model_seed'] = found_config['seed']

        args.__dict__.update(found_config)

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

    return args
