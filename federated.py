import torch
from flamby.utils import evaluate_model_on_tests
from flamby.datasets.fed_tcga_brca import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    metric,
    NUM_CLIENTS,
    get_nb_max_rounds
)
from flamby.datasets.fed_tcga_brca import FedTcgaBrca
from flamby.strategies.fed_opt import FedAdam as strat

# Training
# Create the training data loaders
train_dataloaders = [
            torch.utils.data.DataLoader(
                FedTcgaBrca(center = i),
                batch_size = BATCH_SIZE,
                shuffle = True,
                num_workers = 0
            )
            for i in range(NUM_CLIENTS)
        ]

lossfunc = BaselineLoss()
m = Baseline()
args = {
            "training_dataloaders": train_dataloaders,
            "model": m,
            "loss": lossfunc,
            "optimizer_class": torch.optim.SGD,
            "learning_rate": LR / 10.0,
            "num_updates": 100,
            "nrounds": get_nb_max_rounds(100),
        }
s = strat(**args)
m = s.run()[0]

