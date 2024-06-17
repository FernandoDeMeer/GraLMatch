from src.models.config import Config
from src.data.dataset import PytorchDataset
from src.models.pytorch_model import PyTorchModel
from src.helpers.seed_helper import initialize_gpu_seed
from src.models.config import read_arguments_test
from torch.utils.data import DataLoader
import torch
from src.helpers.logging_helper import setup_logging
import logging
from src.helpers.path_helper import *
import os
import sys
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

sys.path.append(os.getcwd())

setup_logging()


def generate_groups(model, batch_size=32):
    train, test = model.dataset.get_train_test()
    data_df = model.dataset.get_tokenized_data()

    idx = np.array(data_df.index)
    num_repetitions = len(idx)

    lidx = np.repeat(idx, num_repetitions)
    ridx = np.tile(idx, num_repetitions)

    df = pd.DataFrame({'lid': lidx, 'rid': ridx})
    df = df[df.lid >= df.rid]
    df['label'] = 0

    tokenizer_class = Config.MODELS[model.args.model_name].tokenizer
    model_class = Config.MODELS[model.args.model_name].pretrained_model
    tokenizer = tokenizer_class.from_pretrained(model_class, do_lower_case=model.args.do_lower_case)

    dataset = PytorchDataset(model_name=model.args.model_name, idx_df=df, data_df=data_df,
                             tokenizer=tokenizer, max_seq_length=model.args.max_seq_length)
    loader = DataLoader(dataset, shuffle=False, batch_size=batch_size)

    buffer = {
        'lid': np.zeros(len(lidx), dtype=int),
        'rid': np.zeros(len(lidx), dtype=int),
        'prediction': np.zeros(len(lidx), dtype=int)
    }
    buffer_step = 0
    buffer_length = len(buffer['lid'])

    for step, batch_tuple in tqdm(enumerate(loader), desc='Generating group predictions ...', total=len(loader)):
        model.network.eval()
        outputs, inputs, raw_inputs = model.predict(batch_tuple)

        # Calculate how many were correct
        predictions = torch.argmax(outputs[-1], axis=1)

        start_idx = step * batch_size
        end_idx = np.min([((step + 1) * batch_size), len(lidx)])

        try:
            buffer['prediction'][start_idx:end_idx] = predictions.cpu()
            buffer['lid'][start_idx:end_idx] = raw_inputs['lids'].cpu().reshape(-1)
            buffer['rid'][start_idx:end_idx] = raw_inputs['rids'].cpu().reshape(-1)
        except Exception as e:
            logging.info("Exception: " + str(e))

        buffer_step += 1

        if (step % 25000) == 0:
            pd.DataFrame(buffer).to_csv(
                f"{model.args.experiment_name}__ep{model.args.epoch}__all_pairs__intermediate.csv", index=False)

    pd.DataFrame(buffer).to_csv(f"{model.args.experiment_name}__ep{model.args.epoch}__all_pairs.csv", index=False)


def main(args):
    initialize_gpu_seed(args.model_seed)

    checkpoint_suffix = '__epoch' + str(args.epoch)
    if args.epoch == 0:
        checkpoint_suffix += '__zeroshot'

    file_name = "".join([args.model_name, checkpoint_suffix, '.pt'])
    checkpoint_path = experiment_file_path(args.experiment_name, file_name)

    model = PyTorchModel.load_from_checkpoint(args, checkpoint_path)
    generate_groups(model)
    # import code; code.interact(local=dict(globals(), **locals()))


if __name__ == '__main__':
    args = read_arguments_test()
    main(args)
