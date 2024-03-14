# static model
Implementation of DistMult in temporal training setting

## Installation
install ```requirements.txt```

## Preparation
To run this code for the given GTA_ACLED Use Case copy the data from the data folder to this folder /data/crisis2023


## Running
```
python src/main.py 
```

## Run for selected relations only
set MASK in ```src/engine/engine.py``` to ```True``` s.t. the model is only trained and evaluated on the relations of interest (specified in mask).

## Hyperparameters
specified in conf.yaml

hyperparameters as desribed in our paper set in conf.yaml

If you want to do hyperparameter runing, run 
```
python src/main.py --multirun
```
Thsi will automatically run hyperparameter tuning, and store all results (settings, and valid_mrr) in ```"all_results_"+dataset_name+".csv"```

Note: for this, optuna hydra package is needed. (https://hydra.cc/docs/plugins/optuna_sweeper/) It will log the validation mrrs in validation_mrrs.txt.

## Evaluation
This will create a pkl file with predictions for each test triple and store it to the folder 'results'. copy this file to the 'evaluation' to create a unified evaluation
