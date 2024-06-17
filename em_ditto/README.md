> This repository has been taken from [Ditto](https://github.com/megagonlabs/ditto/). Please refer to the original repository and cite their [publication](https://arxiv.org/abs/2004.00584).

> Also refer to the [original README.md](original_README.md).

# DITTO

## Installing Dependencies

While the authors claim different versions for `torch`, `spacy` and `transformers`, we have found them to be incompatible and replaced them with the ones noted in the [requirements.txt](requirements.txt).
If `conda` is not installed, you can install it using the instructions [here](https://docs.conda.io/projects/miniconda/en/latest/).

In particular, those changes are:
- `torch`: from `1.9.0+cu111` to `1.13.1+cu116`
- `spacy`: from `3.1` to `3.6.0`
- `transformers`: from `4.9.2` to `4.33.2`
- `sentencepiece`: from `0.1.85` to `0.1.99`

```bash
conda create -y -n "ditto" python=3.10

conda activate ditto

conda install -c conda-forge nvidia-apex

pip install -r requirements.txt

python -m spacy download en_core_web_lg
```


In case the dependencies cannot be installed because Rust is not present, install Rust using:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```


Maybe install Apex via:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install
```

## Running Code



### Running Ditto's Blocking

DITTO provides its own version of blocking, which we **do not use** for our experiments. Instead we apply the blocking described in our manuscript, which is done in the `datainc_code` part of the code.
In case you want to learn more about DITTO's blocking, refer to the original author's paper and take a look at [the blocking README](blocking/README.md).



### Running Ditto

Make sure that your dataset is added to the [configs.json](configs.json) file in order to select it through the `--task` flag. A large number of benchmark datasets is already provided. You can take the settings used by the Ditto authors from their [publication](https://arxiv.org/abs/2004.00584), i.e. which task used which `--lm` etc.

Please make sure that you generate the DITTO-encoded variants of the synthetic datasets using the notebooks outlined in the `datainc_code/notebooks/ditto_processing/` directory.

Example function call to run Ditto on the `synthetic` datasets:

```bash
conda activate ditto

# Training different DITTO models
python train_ditto.py --task finance/synthetic_companies --run_id 0 --batch_size 32 --max_len 128 --n_epochs 5 --finetuning --save_model --lm distilbert --logdir checkpoints/synthetic_companies128/
python train_ditto.py --task finance/synthetic_companies --run_id 0 --batch_size 32 --max_len 256 --n_epochs 5 --finetuning --save_model --lm distilbert --logdir checkpoints/synthetic_companies256/

python train_ditto.py --task finance/synthetic_securities --run_id 0 --batch_size 32 --max_len 128 --n_epochs 5 --finetuning --save_model --lm distilbert --logdir checkpoints/synthetic_securities128/
python train_ditto.py --task finance/synthetic_securities --run_id 0 --batch_size 32 --max_len 256 --n_epochs 5 --finetuning --save_model --lm distilbert --logdir checkpoints/synthetic_securities256/

# Running the blocked test set
python test_graph_matching_ditto.py --task finance/synthetic_companies --run_id 0 --batch_size 32 --max_len 128 --n_epochs 5 --finetuning --save_model --lm distilbert --logdir checkpoints/synthetic_companies128
python test_graph_matching_ditto.py --task finance/synthetic_companies --run_id 0 --batch_size 32 --max_len 256 --n_epochs 5 --finetuning --save_model --lm distilbert --logdir checkpoints/synthetic_companies256

python test_graph_matching_ditto.py --task finance/synthetic_securities --run_id 0 --batch_size 32 --max_len 128 --n_epochs 5 --finetuning --save_model --lm distilbert --logdir checkpoints/synthetic_securities128
python test_graph_matching_ditto.py --task finance/synthetic_securities --run_id 0 --batch_size 32 --max_len 256 --n_epochs 5 --finetuning --save_model --lm distilbert --logdir checkpoints/synthetic_securities256
```

For the GraLMatch graph cleanup, check the script under `<repository_location>/datainc_code/scripts/ditto/run_graph_cleanup_on_predictions.py` as such (make sure to `conda deactivate` the conda environment and activate the environment of the `datainc_code` directory):

```bash
python scripts/ditto/run_graph_cleanup_on_predictions.py --task finance/synthetic_companies --results_file_path ../em_ditto/checkpoints/synthetic_companies128/finance/synthetic_companies --processed_folder_path data/processed/synthetic_companies
python scripts/ditto/run_graph_cleanup_on_predictions.py --task finance/synthetic_companies --results_file_path ../em_ditto/checkpoints/synthetic_companies256/finance/synthetic_companies --processed_folder_path data/processed/synthetic_companies

python scripts/ditto/run_graph_cleanup_on_predictions.py --task finance/synthetic_securities --results_file_path ../em_ditto/checkpoints/synthetic_securities128/finance/synthetic_securities --processed_folder_path data/processed/synthetic_securities
python scripts/ditto/run_graph_cleanup_on_predictions.py --task finance/synthetic_securities --results_file_path ../em_ditto/checkpoints/synthetic_securities256/finance/synthetic_securities --processed_folder_path data/processed/synthetic_securities
```
