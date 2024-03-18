## Evaluation of models based on created .pkl files

## 1. Requirements
Install:
* torch
* numpy
* os
* enum
* pickle
* json


## 2. Prerequisites
* In the folder results create a folder for each model that should be evaluated
* ['TLogic', 'RE-GCN', 'static_baseline', "Timetraveler"]
* New models have to be added in run_evaluation, dir_names
* Copy all .pkl files of interest to the respective models folder


## 3. Run Evaluation
* Run run_evaluation.py
* This creates a .json file, with one entry per .pkl file, containing scores of interest (MRR, MR, Hits@K, MRR over snapshots, per relation)
* Run parser.py to create an excel-table from this json file, with one sheet per setting (raw, static, time-aware). For us, time-aware setting is of importance.

