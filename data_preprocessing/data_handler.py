'''
 TKG-ACLED-GTA-Dataset


'''
import numpy as np
import os
import pathlib

from itertools import groupby
from operator import itemgetter


def load(dataset_name: str) -> tuple: 
    """ load txt files with the graph quadruples """
    if 'merged' in dataset_name:
        name = 'acled_subset_merged_gta_ids_graph.txt'
        if 'aggregated' in dataset_name:
            name = 'acled_subset_merged_gta_aggregated_ids_graph.txt'
    else:
        name = 'acled_subset_gta_ids_graph.txt'
        if 'aggregated' in dataset_name:
            name = 'acled_subset_gta_aggregated_ids_graph.txt'
    if '2023' in dataset_name:
            name = dataset_name
    

    
    root = os.path.join(pathlib.Path().resolve(), 'data')

    data= _load_file(root, name)
   
    return data

def group_by(data: np.array, key_idx: int) -> dict:
    data_grouped = groupby(data, key=itemgetter(key_idx))
    data_dict = {key: np.delete(np.array(list(values)), key_idx, 1) for key, values in data_grouped}

    return data_dict

def _load_file( path: str, filename: str) -> np.array:
    data = np.loadtxt(os.path.join(path, filename), dtype=int) 
    print(data.shape)

    return data
