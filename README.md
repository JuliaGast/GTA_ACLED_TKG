
# GTA_ACLED_TKG

This is the code to the paper Dynamic Representations of Global Crises: A Temporal Knowledge Graph For Conflicts, Trade and Value Networks
We well add the authors after review period has passed.

In the following we describe the steps needed to reproduce our results. It is splot in two parts, 1. Dataset Preprocessing and 2. TKG Forecasting.
It is not required to re-run the Dataset Preprocessing Steps. We provide the output of Dataset Preprocessing in ```/data/crisis2023```. These files can be used for TKG Forecasting.

## 1. Dataset Preprocessing 
### Sparql Queries:
* Please see README.md in folder ```queries```

### Timestep Assignment:
* Run ```python3 ./data_preprocessing/ts_assignment_gta_star.py``` and ```python3 /data_preprocessing/ts_assignment_acled.py``` to read the .nt files and assign timesteps
* This requires the rdflib package, that can be downloaded here https://github.com/XuguangSong98/rdflib and put into the data_preprocessing folder. Processing data with this package very slow and can take hours to days.
* The output are csv files that can be found in ```/data/acled``` and ```/data/gta``` respectively

### Merge the two datasets:
* Run ```python3 ./data_preprocessing/merke-tkg-from-gta-acled.py``` to merge both subsets and create train, valid, test.txt
* What it does:
 * Specify timerange of interest. In our case this is 2023-01-01 – 2023-12-31
 * Split dataset based on timesteps. Specify train/valid/test split. In our case it is 80/10/10
 * Automatically stores the resulting files in ```/data/crisis2023```
* It produces various files:
  * ```train.txt```, ```valid.txt```, ```test.txt```: one line per quadruple, quadruples as ```subject_id, relation_id, object_id, timestamp``` (from 0 to num_timesteps), ```original_dataset_id``` (0: gta, 1: acled)
  * ```train_names.txt```, ```valid_names.txt```, ```test_names.txt```: one line per quadruple with string description for each node and relation; ```subject_string, relation_string, object_string, original_dataset_id, timestamp``` (from 0 to num_timesteps)
  * ```id_to_node.json``` and ```id_to_rel.json```: contains dicts with mappings from ```"node_id"``` to ```node_string```, and ```"relation_id"``` to ```relation string```.
  * ```node_to_id.json``` and ```rel_to_id.json```: contains dicts with mappings from ```node_string``` to ```"node_id"```, and ```relation string``` to "relation_id" .
  * ```stat.txt```: two entries, number of nodes, number of distinct relations


## 2. TKG Forecasting:
All models for TKG Forecasting are in the folder ```models```. Follow the instructions in the respective ```README.md```.


## 3. Result Evaluation:
The code for evaluating the results for TKG Forecasting are in the folder ```result_evaluation.py```. Follow the instructions in the respective ```README.md```.
