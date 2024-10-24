# TITer

This is the official and modified code release of the following paper:
EMNLP 2021 paper **TimeTraveler: Reinforcement Learning for Temporal Knowledge Graph Forecasting**

![TITer](./img/main.jpg)

https://github.com/JHL-HUST/TITer

### GTA_ACLED

To run this code for the given GTA_ACLED Use Case copy the data from the data folder to this folder ```/data/crisis2023```

install the dependencies from ```requirements.txt```

#### Data preprocessing
```
python3 preprocess_data.py --data_dir data/crisis2023
```
#### Dirichlet parameter estimation

If you use the reward shaping module, you need to do this step.

```
python3 mle_dirichlet.py --data_dir data/crisis2023 
```

#### Train and test:
```
python3 main.py --data_path data/crisis2023 --do_train --do_test 
```


#### Evaluate:

This will create a pkl file with predictions for each test triple and store it to the folder 'results'. copy this file to the 'evaluation/Timetraveler' and run the evaluation script to create a unified evaluation

####  Hyperparameter Range GTA_ACLED
Please find the hyperparameter range in conf2.yaml. 
We set the hyperparameter default values to the values reported in our paper.
If you want to do hyperparameter tuning, set an additional multirun, like this:
```
python3 main.py --data_path data/crisis2023 --do_train --multirun
```
Note: for this, optuna hydra package is needed. (https://hydra.cc/docs/plugins/optuna_sweeper/)
It will log the validation mrrs in validation_mrrs.txt. 

## Original paper README:
### Qucik Start

#### Data preprocessing

This is not necessary, but can greatly shorten the experiment time.

```
python3 preprocess_data.py --data_dir data/ICEWS14
```

#### Dirichlet parameter estimation

If you use the reward shaping module, you need to do this step.

```
python3 mle_dirichlet.py --data_dir data/ --time_span 24
```

#### Train
you can run as following:
```
python3 main.py --data_path data/ICEWS14 --cuda --do_train --reward_shaping --time_span 24
```

#### Test
you can run as following:
```
python3 main.py --data_path data/ICEWS14 --cuda --do_test --IM --load_model_path xxxxx
```

### Acknowledgments
model/dirichlet.py is from https://github.com/ericsuh/dirichlet

### Cite

```
@inproceedings{Haohai2021TITer,
	title={TimeTraveler: Reinforcement Learning for Temporal Knowledge Graph Forecasting},
	author={Haohai Sun, Jialun Zhong, Yunpu Ma, Zhen Han, Kun He.},
	booktitle={EMNLP},
	year={2021}
}
```
