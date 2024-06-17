import os
import argparse
import json
import sys
import torch
import numpy as np
import pandas as pd
import random
import time

import sklearn.metrics as metrics
from tqdm.auto import tqdm
from torch.utils import data

sys.path.insert(0, "Snippext_public")

from ditto_light.dataset import DittoDataset
from ditto_light.summarize import Summarizer
from ditto_light.knowledge import *
from ditto_light.ditto import evaluate, DittoModel

import warnings
warnings.filterwarnings('ignore')



def evaluate_test_set(model, iterator, threshold):
    all_p = []
    all_y = []
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(iterator, desc='Running Graph Candidates...', total=len(iterator)):
            x, y = batch
            logits = model(x)
            probs = logits.softmax(dim=1)[:, 1]
            all_probs += probs.cpu().numpy().tolist()
            all_y += y.cpu().numpy().tolist()

    pred = [1 if p > threshold else 0 for p in all_probs]
    f1 = metrics.f1_score(all_y, pred)

    # Log probabilities of test set during training process

    df_proba = pd.DataFrame({
        'probabilities': all_probs,
        'true_label': all_y,
        'task': hp.task,
        'threshold': threshold
    })

    precision = metrics.precision_score(all_y, pred)
    recall = metrics.recall_score(all_y, pred)
    print(f"\tPrecision:{precision:.4f}\tRecall:{recall:.4f}\tF1:{f1:.4f}")

    return df_proba


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Structured/Beer")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--lm", type=str, default='distilbert')
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--da", type=str, default=None)
    parser.add_argument("--alpha_aug", type=float, default=0.8)
    parser.add_argument("--dk", type=str, default=None)
    parser.add_argument("--summarize", dest="summarize", action="store_true")
    parser.add_argument("--size", type=int, default=None)



    hp = parser.parse_args()

    # set seeds
    seed = hp.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # only a single task for baseline
    task = hp.task

    # create the tag of the run
    run_tag = '%s_lm=%s_da=%s_dk=%s_su=%s_size=%s_id=%d' % (task, hp.lm, hp.da,
            hp.dk, hp.summarize, str(hp.size), hp.run_id)
    run_tag = run_tag.replace('/', '_')

    # load task configuration
    configs = json.load(open('configs.json'))
    configs = {conf['name'] : conf for conf in configs}
    config = configs[task]

    if 'graph_matching' not in config.keys():
        raise ValueError(f"The task {hp.task} is missing the :graph_matching key in the config.json.")

    # Load Specific Test Set

    trainset = config['trainset']
    validset = config['validset']

    train_dataset = DittoDataset(trainset,
                                   lm=hp.lm,
                                   max_len=hp.max_len,
                                   size=hp.size,
                                   da=hp.da)
    valid_dataset = DittoDataset(validset, lm=hp.lm)
    test_candidate_dataset = DittoDataset(config['graph_matching'], lm=hp.lm)

    valid_iter = data.DataLoader(dataset=valid_dataset,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=train_dataset.pad)
    test_candidate_iter = data.DataLoader(dataset=test_candidate_dataset,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=train_dataset.pad)

    # Load Model Checkpoint
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DittoModel(device=device, lm=hp.lm, alpha_aug=hp.alpha_aug)
    ckpt = torch.load(f'{hp.logdir}/{hp.task}/model.pt')
    model.load_state_dict(ckpt['model'])
    model = model.cuda()

    _, threshold = evaluate(model, valid_iter, test=False, threshold=None)
    print(f"Threshold={threshold}")

    test_start = time.time()

    # Evaluate Specific Test Set
    df_proba = evaluate_test_set(model, test_candidate_iter, threshold)

    print(f"\tTesting took {time.time()-test_start:.2f} seconds.")

    save_dir = os.path.join(hp.logdir, hp.task)
    save_path = os.path.join(save_dir, f'test_candidate_probabilities.csv')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    df_proba.to_csv(save_path, index=False)

