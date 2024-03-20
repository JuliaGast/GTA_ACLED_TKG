import numpy as np
import os
import pathlib

from itertools import groupby
from operator import itemgetter

# import utils

def load(dataset_name: str) -> np.array:
    # root = os.path.join(pathlib.Path().resolve(), 'data', dataset_name)
    root = os.path.join(pathlib.Path('/home/jgastinger/coypu/model'), 'data', dataset_name)
    stat = np.loadtxt(os.path.join(root, 'stat.txt'))
    train = _load_file(dataset_name, root, 'train.txt')
    valid = _load_file(dataset_name, root, 'valid.txt')
    test = _load_file(dataset_name, root, 'test.txt')
    
    return train, valid, test, stat

def group_by(data: np.array, key_idx: int) -> dict:
    data_grouped = groupby(data, key=itemgetter(key_idx))
    data_dict = {key: np.delete(np.array(list(values)), key_idx, 1) for key, values in data_grouped}

    return data_dict

def _load_file(identifier: str, path: str, filename: str) -> np.array:
    data = np.loadtxt(os.path.join(path, filename), dtype=int) 
    # dataset-specific preprocessing
    if identifier == 'ICEWS18':
        data = _icews18_parser(data)
    if identifier =='ICEWS14':
        data = _icews18_parser(data)
    # debug
    print(data.shape)

    return data


def _icews18_parser(data: np.array) -> np.array:
    data = data.astype(np.int32)
    return np.delete(data, 4, 1)


# def compute_relations_dict(quads_incl_inverse):
#     """ compute a dict which has keys: timesteps, values {node_id: [relation_ids]}"""

#     ts_dict = group_by(quads_incl_inverse, 3) # dict with keys: timesteps, values: other entries
#     ts_node_rel_dict = {}
#     for ts, triples in ts_dict.items():
#         ts_node_rel_dict[ts] = {}
#         for entry in triples:
#             a, b, c = entry
#             if a not in ts_node_rel_dict[ts]:
#                 ts_node_rel_dict[ts][a] = [b]
#             else:
#                 ts_node_rel_dict[ts][a].append(b)
#     return ts_node_rel_dict


def compute_relations_dict(ts_dict, num_relations):
    """ compute a dict which has keys: timesteps, values {node_id: [relation_ids]}"""

    ts_node_rel_dict = {}
    for ts, triples in ts_dict.items():
        triples_incl_inv= utils.add_inverse_triples(triples, num_relations)
        ts_node_rel_dict[ts] = {}
        for entry in triples_incl_inv:
            a, b, c = entry
            if a not in ts_node_rel_dict[ts]:
                ts_node_rel_dict[ts][a] = [b]
            else:
                ts_node_rel_dict[ts][a].append(b)
    return ts_node_rel_dict