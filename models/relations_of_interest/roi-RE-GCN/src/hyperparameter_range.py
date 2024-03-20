"""
 Hyperparameter range specification.
"""


hp_range = {
    "n_hidden": [100, 200],
    "n_layers": [1, 2],
    "dropout": [0.2, 0.4],
    "n_bases": [100],
    "train_history_len": [1, 3, 7, 10] #added julia
}


