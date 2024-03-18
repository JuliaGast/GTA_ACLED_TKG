
# GTA_ACLED_TKG

## 1. Dataset Preprocessing 
### Sparql Queries:
*TODO Lorenz?  - see readme in folder queries

### Timestep Assignment:
* Run ```python3 ./data_preprocessing/ts_assignment_gta_star.py``` and ```python3 /data_preprocessing/ts_assignment_acled.py``` to read the .nt files and assign timesteps
* This requires the rdflib package, that can be downloaded here https://github.com/XuguangSong98/rdflib and put into the data_preprocessing folder. Processing data with this package very slow and can take hours to days.
* The output are csv files that can be found in ```/data/acled``` and ```/data/gta``` respectively

### Merge the two datasets:
* Run ```python3 ./data_preprocessing/merke-tkg-from-gta-acled.py``` to merge both subsets and create train, valid, test.txt
* What it does:
 * Specify timerange of interest. In our case this is 2023-01-01 – 2023-12-31
 * Split dataset based on timesteps. Specify train/valid/test split. In our case it is 80/10/10
 * Automatically stores the resulting files in ```/data/crisis_merged2023-01-012023-12-31```
* It produces various files:
  * ```train.txt```, ```valid.txt```, ```test.txt```: one line per quadruple, quadruples as ```subject_id, relation_id, object_id, timestamp``` (from 0 to num_timesteps), ```original_dataset_id``` (0: gta, 1: acled)
  * ```train_names.txt```, ```valid_names.txt```, ```test_names.txt```: one line per quadruple with string description for each node and relation; ```subject_string, relation_string, object_string, original_dataset_id, timestamp``` (from 0 to num_timesteps)
  * ```id_to_node.json``` and ```id_to_rel.json```: contains dicts with mappings from ```"node_id"``` to ```node_string```, and ```"relation_id"``` to ```relation string```.
  * ```node_to_id.json``` and ```rel_to_id.json```: contains dicts with mappings from ```node_string``` to ```"node_id"```, and ```relation string``` to "relation_id" .


## 2. TKG Forecasting:
All models for TKG Forecasting are in the folder ```models```. Follow the instructions in the respective ```README.md```.
