import os
import json
import datetime
import pandas as pd

class TrainedModels:
    def __init__(self):
        self._build_available_models()

    def _build_available_models(self):
        checkpoint_folder = os.path.join('models')
        
        models = []

        # Iterate over /models folder to gather all potential checkpoints
        #
        for folder in os.listdir(checkpoint_folder):
            checkpoint_folder_path = os.path.join(checkpoint_folder, folder)
            if os.path.isdir(checkpoint_folder_path):
                config_path = os.path.join(checkpoint_folder_path, 'config.cfg')

                # Only add folder if the config.cfg is present
                #
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        data = json.load(f)

                    # Read the relevant information from the config and
                    # check how many checkpoints are available (added as full path)
                    #
                    models.append({
                        'dataset_name': data['dataset_name'],
                        'model_name': data['model_name'],
                        'experiment_name': data['experiment_name'],
                        'timestamp': datetime.datetime.strptime(data['experiment_name'][:17], '%Y-%m-%d_%H%M%S'),
                        'checkpoints': [os.path.join(checkpoint_folder_path, x) for x in os.listdir(checkpoint_folder_path) 
                                        if x.endswith('.pt')]
                    })

        self.df = pd.DataFrame(models) \
                    .sort_values(['timestamp'], ascending=False) \
                    .reset_index(drop=True)

    # Returns the checkpoint of the newest trained model for a given dataset_name,
    # an optional epoch can be requested and will be confirmed if it is available
    #
    # returns experiment_name, epoch
    #
    def checkpoint_by_dataset(self, dataset_name, epoch=None):
        row = self.df.iloc[self.df[self.df.dataset_name == dataset_name].index[0]]

        epochs = []
        for cp in row['checkpoints']:
            epochs.append(int(cp.split('.pt')[0].split('__zeroshot')[0].split('epoch')[1]))

        if not epoch:
            return row['experiment_name'], epochs[0]
        else:
            if epoch in epochs:
                return row['experiment_name'], epoch
            else:
                raise FileNotFoundError(
                    f"Experiment {row['experiment_name']} checkpoint for epoch {epoch} not found, only found {epochs}.")
