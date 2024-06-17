import getpass
import os

import wandb

from src.helpers.logging_helper import setup_logging

setup_logging()


def initialize_wandb(args, job_type='train'):
    wandb.init(
        project="PLEASE_REPLACE_ME_WITH_YOUR_WANDB_PROJECT_NAME",
        entity="PLEASE_REPLACE_ME_WITH_YOUR_WANDB_ENTITY",
        config=args.__dict__,
        job_type=job_type,
        group=args.dataset_name,
        # mode='disabled',
        name=args.experiment_name,
        tags=[
            f"user__{getpass.getuser()}",
            f"host__{os.uname()[1]}",
            f"dataset__{args.dataset_name}",
            f"model__{args.model_name}",
            f"seed__{args.seed}",
            f"model_seed__{args.model_seed}"
        ]
    )
