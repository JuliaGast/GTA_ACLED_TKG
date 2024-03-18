# """/*
#  *    Dynamic Representations of Global Crises: A Temporal Knowledge Graph For Conflicts, Trade and Value Networks
#  *
#  *        File: utils.py
#  *
#  *     Authors: Deleted for purposes of anonymity 
#  *
#  *     Proprietor: Deleted for purposes of anonymity --- PROPRIETARY INFORMATION
#  * 
#  * The software and its source code contain valuable trade secrets and shall be maintained in
#  * confidence and treated as confidential information. The software may only be used for 
#  * evaluation and/or testing purposes, unless otherwise explicitly stated in the terms of a
#  * license agreement or nondisclosure agreement with the proprietor of the software. 
#  * Any unauthorized publication, transfer to third parties, or duplication of the object or
#  * source code---either totally or in part---is strictly prohibited.
#  *
#  *     Copyright (c) 2021 Proprietor: Deleted for purposes of anonymity
#  *     All Rights Reserved.
#  *
#  * THE PROPRIETOR DISCLAIMS ALL WARRANTIES, EITHER EXPRESS OR 
#  * IMPLIED, INCLUDING BUT NOT LIMITED TO IMPLIED WARRANTIES OF MERCHANTABILITY 
#  * AND FITNESS FOR A PARTICULAR PURPOSE AND THE WARRANTY AGAINST LATENT 
#  * DEFECTS, WITH RESPECT TO THE PROGRAM AND ANY ACCOMPANYING DOCUMENTATION. 
#  * 
#  * NO LIABILITY FOR CONSEQUENTIAL DAMAGES:
#  * IN NO EVENT SHALL THE PROPRIETOR OR ANY OF ITS SUBSIDIARIES BE 
#  * LIABLE FOR ANY DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES
#  * FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF INFORMATION, OR
#  * OTHER PECUNIARY LOSS AND INDIRECT, CONSEQUENTIAL, INCIDENTAL,
#  * ECONOMIC OR PUNITIVE DAMAGES) ARISING OUT OF THE USE OF OR INABILITY
#  * TO USE THIS PROGRAM, EVEN IF the proprietor HAS BEEN ADVISED OF
#  * THE POSSIBILITY OF SUCH DAMAGES.
#  * 
#  * For purposes of anonymity, the identity of the proprietor is not given herewith. 
#  * The identity of the proprietor will be given once the review of the 
#  * conference submission is completed. 
#  *
#  * THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#  */"""



import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from rdflib import Graph
import rdflib
import json
from copy import copy



def store_graph(current_graph,  time_to_id_ourinterval, t, data_type='train', location="./data", task_name='crisis_merged'):
    snap = copy(current_graph)
    snap['ts'] = time_to_id_ourinterval[t]*np.ones(len(current_graph[0])).astype(int)
    file_path = location + '/' + task_name +  '/' + data_type+ '_names.txt'
    snap.to_csv(file_path, sep='\t', mode='a', header=False, index=False)

def store_ids(current_graph, node_to_id, rel_to_id, time_to_id_ourinterval, t, data_type='train', location="./data", task_name='/crisis_merged/'):
    # with open(location + '/' + task_name +  '/' + data_type+'.txt', 'a') as id_file: #write current graph (ids) to txt file
    obs = []
    subs = []
    preds = []
    dataset_origs = []
    for sub, pred, ob, dataset_orig in zip(current_graph[0], current_graph[1], current_graph[2], current_graph[3]): 
        subs.append(node_to_id[sub])
        preds.append(rel_to_id[pred])
        obs.append(node_to_id[ob])
        dataset_origs.append(dataset_orig)
    frame = {'subs': pd.Series(subs), 'preds': pd.Series(preds), 'obs': pd.Series(obs) , 
                'ts': pd.Series(time_to_id_ourinterval[t]*np.ones(len(subs)).astype(int)), 
                'dataset_origin': pd.Series(dataset_origs)}
    dframe =pd.DataFrame(frame)
    # frame_str = dframe.to_string(header=False, index=False)
    file_path = location + '/' + task_name +  '/' + data_type+'.txt'
    dframe.to_csv(file_path, sep='\t', mode='a', header=False, index=False)
        # id_file.write(frame_str)
        # id_file.write("\n")
        
def df_to_rdfgraph(graph_df):
    g = Graph()
    triples = graph_df.to_numpy()

    for triple in triples:
        if 'http' in triple[0]: s = rdflib.term.URIRef(triple[0]) 
        else: s = rdflib.term.BNode(triple[0])
        if 'http' in triple[2]: o = rdflib.term.URIRef(triple[2]) 
        else: o = rdflib.term.BNode(triple[2])       
        g.add((s,rdflib.term.URIRef(triple[1]),o))
    return g

def map_ids_to_time(timesteps_range_all, timesteps_range):
    # create dicts that map the timesteps to numeric id, and other way round. 
    i = 0
    time_to_id = {}
    id_to_time = {}
    for t in timesteps_range_all:
        t = str(t.date())
        time_to_id[t] = i #lookup dict
        id_to_time[i] = t
        i+=1

    i = 0
    time_to_id_ourinterval = {}
    id_to_time_ourinterval = {}
    for t in timesteps_range:
        t = str(t.date())
        time_to_id_ourinterval[t] = i #lookup dict
        id_to_time_ourinterval[i] = t
        i+=1
    return time_to_id, id_to_time, time_to_id_ourinterval, id_to_time_ourinterval

def map_ids_to_string(added_dict, timesteps_range, task_name):
    # create dicts that map the node strings to numeric id, and other way round. same for relations
    # dump those dicts as json files
    # create stat.txt with number of nodes and number of relations

    all_nodes_set = set()
    all_relations_set = set()
    for t_all in timesteps_range: 
        t = str(t_all.date())
        if t in added_dict.keys():
            nodes =  [[i[0] , i[2]] for i in added_dict[t]]
            nodes_1d = [item for sublist in nodes for item in sublist]

            relations = [i[1] for i in added_dict[t]]

            all_nodes_set.update(nodes_1d)
            all_relations_set.update(relations)
    nodes =  [] #[[i[0] , i[2]] for i in static_list]
    nodes_1d = [item for sublist in nodes for item in sublist]

    relations = [] #[i[1] for i in static_list]

    all_nodes_set.update(nodes_1d)
    all_relations_set.update(relations)

    i = 0
    node_to_id = {}
    id_to_node = {}
    for node in all_nodes_set:
        node_to_id[node] = i #lookup dict
        id_to_node[i] = node
        i+=1

    rel_to_id = {}
    id_to_rel = {}
    i = 0
    for rel in all_relations_set:
        rel_to_id[rel] = i #lookup dict
        id_to_rel[i] = rel
        i+=1

    with open('./data/' + task_name+'/node_to_id.json', 'a') as file:
        json.dump(node_to_id, file)
    with open('./data/' + task_name+'/id_to_node.json', 'a') as file:
        json.dump(id_to_node, file)
    with open('./data/' + task_name+'/rel_to_id.json', 'a') as file:
        json.dump(rel_to_id, file)
    with open('./data/' + task_name+'/id_to_rel.json', 'a') as file:
        json.dump(id_to_rel, file)

    num_nodes = len(all_nodes_set)
    num_rels = len(all_relations_set)
    with open('./data/' + task_name+ '/stat.txt', 'a') as id_file:
            frame = {'num_nodes': pd.Series(num_nodes), 'num_rels': pd.Series(num_rels)}
            dframe =pd.DataFrame(frame)
            frame_str = dframe.to_string(header=False, index=False)
            id_file.write(frame_str)

    return node_to_id, rel_to_id

def create_dicts_from_df(datasetid_gta, datasetid_acled=None):
    # create a dict with key: timestamp, values: triples at that timestamp for both, acled and gta plus dataset identifier
    # read the specified csv files, use the columns of interest
    dataset_df_gta = pd.read_csv(datasetid_gta, keep_default_na=False)  # because it contains labels "NA" for Namibia, 
                                                                        # which will otherwise be interpreeted as nan
    dataset_df_acled = pd.read_csv(datasetid_acled, keep_default_na=False)

    # gta
    added_dict = {start_d: df[["0","1","2","6"]].values for start_d,df in dataset_df_gta.groupby("3")}
    # 6: gta string
    if '1000-03-03' in added_dict.keys():
        del added_dict['1000-03-03'] #remove the entries which have no announcement date

    # acled
    acled_ts_dict = {impl_d: df[["0","1","2","6"]].values for impl_d,df in dataset_df_acled.groupby("3")}
    # 6: acled string

    # combine both
    merged_dict = {}
    for key, val1 in added_dict.items():
        if key in acled_ts_dict.keys():
            merged_dict[key] = list(val1) + list(acled_ts_dict[key])
        else:
            merged_dict[key] = list(val1)
    for key2, val2 in acled_ts_dict.items(): # add the acled triples for not existing gta timestamps
        if key2 not in added_dict.keys():
            merged_dict[key2] = list(val2)    
    
    return merged_dict


