import pandas as pd
import numpy as np
from utils import df_to_rdfgraph, map_ids_to_string, map_ids_to_time, create_dicts_from_df, store_ids, store_graph
from copy import copy
import os
from tqdm import tqdm

# 0) read graph_gta.csv  and graph_acled.csv & create dicts

acled_name= 'acled_all'
gta_name = 'gta_2023' #gta_2023

if gta_name == 'gta':
    datasetid_gta = './data/gta/gta_newquads.csv'
elif gta_name == 'gta_2023':
    datasetid_gta = './data/gta/gta_2023.csv'
elif gta_name == 'gta_aggregated':
    datasetid_gta = './data/gta/graph_gta_aggregated.csv'

if acled_name == 'acled_subset':
    datasetid_acled = './data/acled/graph_acled_subset.csv'
    folder_name = 'crisis'
elif acled_name == 'acled_subset_merged':
    datasetid_acled = './data/acled/graph_acled_subset_merged.csv'    
    folder_name = 'crisis_merged'
elif acled_name == 'acled_event':
    datasetid_acled = './data/acled/acled_2023_event_types.csv'    
    folder_name = 'crisis_merged'
elif acled_name == 'acled_all':
    datasetid_acled = './data/acled/acled_2023_all.csv'    
    folder_name = 'crisis_merged'



task_name = 'crisis2023'
data_loc = './data'
added_dict = create_dicts_from_df(datasetid_gta, datasetid_acled) 
# added_dict: key: date. all triples that have announcement date (gta) or general date (acled) at that timestep. 

# 1) create a list  with all possible timesteps, and timesteps of interest
start_range="2023-01-01"
end_range="2023-12-31"
folder_name +=start_range
folder_name +=end_range
if not os.path.exists(data_loc + '/' + folder_name):
    os.mkdir(data_loc + '/' + folder_name)
print('make sure the out-files do not exist yet, results will be appended!, foldername: ', data_loc + '/' + folder_name)
timesteps_range = pd.date_range(start_range,end_range) # only timesteps added after 2020-01-01 and removed before 2022
timesteps_range_all = pd.date_range(start="1900-01-01",end="2080-12-31")  

# 1a) train/valid/test percentage
train_per = 80
valid_per = 10
num_timesteps = len(timesteps_range)
timesteps_range_train_id = int(train_per/100*num_timesteps)
timesteps_range_valid_id = int((valid_per+train_per)/100*num_timesteps)
timesteps_range_test_id= int((100)/100*num_timesteps)-1



# 2) string to id mapping (for nodes and relations); date to id mapping
node_to_id, rel_to_id = map_ids_to_string(added_dict, timesteps_range, folder_name)
time_to_id, id_to_time, time_to_id_ourinterval, id_to_time_ourinterval = map_ids_to_time(timesteps_range_all, timesteps_range)

# 3) for each possible timesteps: create graph snapshot with all events that happen in that timestep  & write quadruples to txt file-> 
# add all triples that are added at this timestep and all remove triples that are removed at it
timestamps = []
graph_timeseries ={}
triples_dict = {}
current_graph = pd.DataFrame(columns =[0, 1, 2, 'ts', 3 ])
tminus = None

for t_all in tqdm(timesteps_range): 
    train_val_test = 'test'
    if t_all < timesteps_range[timesteps_range_valid_id]:
        train_val_test= 'valid'
    if t_all < timesteps_range[timesteps_range_train_id]:
        train_val_test= 'train'
    
    t = str(t_all.date())

    if t in added_dict.keys(): # add all the triples that were added in this ts
        triples_added_t_df = pd.DataFrame(added_dict[t])
        current_graph = triples_added_t_df.drop_duplicates(keep='first') 
        

    triples_dict[str(t_all.date())] = current_graph
    if t_all >= timesteps_range[0]:
        graph = df_to_rdfgraph(current_graph)       

        store_graph(current_graph,  time_to_id_ourinterval, t,  data_type=train_val_test, location=data_loc,task_name=folder_name)
        store_ids(current_graph, node_to_id, rel_to_id, time_to_id_ourinterval, t, data_type=train_val_test, location=data_loc, task_name=folder_name)


    tminus = copy(t)

print("done")



