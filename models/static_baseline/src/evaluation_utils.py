"""*
 *     Static Baseline
 *
 *        File: evaluation_utils.py
 *
 *     Authors: Deleted for purposes of anonymity 
 *
 *     Proprietor: Deleted for purposes of anonymity --- PROPRIETARY INFORMATION
 * 
 * The software and its source code contain valuable trade secrets and shall be maintained in
 * confidence and treated as confidential information. The software may only be used for 
 * evaluation and/or testing purposes, unless otherwise explicitly stated in the terms of a
 * license agreement or nondisclosure agreement with the proprietor of the software. 
 * Any unauthorized publication, transfer to third parties, or duplication of the object or
 * source code---either totally or in part---is strictly prohibited.
 *
 *     Copyright (c) 2021 Proprietor: Deleted for purposes of anonymity
 *     All Rights Reserved.
 *
 * THE PROPRIETOR DISCLAIMS ALL WARRANTIES, EITHER EXPRESS OR 
 * IMPLIED, INCLUDING BUT NOT LIMITED TO IMPLIED WARRANTIES OF MERCHANTABILITY 
 * AND FITNESS FOR A PARTICULAR PURPOSE AND THE WARRANTY AGAINST LATENT 
 * DEFECTS, WITH RESPECT TO THE PROGRAM AND ANY ACCOMPANYING DOCUMENTATION. 
 * 
 * NO LIABILITY FOR CONSEQUENTIAL DAMAGES:
 * IN NO EVENT SHALL THE PROPRIETOR OR ANY OF ITS SUBSIDIARIES BE 
 * LIABLE FOR ANY DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES
 * FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF INFORMATION, OR
 * OTHER PECUNIARY LOSS AND INDIRECT, CONSEQUENTIAL, INCIDENTAL,
 * ECONOMIC OR PUNITIVE DAMAGES) ARISING OUT OF THE USE OF OR INABILITY
 * TO USE THIS PROGRAM, EVEN IF the proprietor HAS BEEN ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGES.
 * 
 * For purposes of anonymity, the identity of the proprietor is not given herewith. 
 * The identity of the proprietor will be given once the review of the 
 * conference submission is completed. 
 *
 * THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
 *"""


import torch
import numpy as np
from copy import copy
def store_scores(directory:str, method_name:str, query_name:str, dataset_name:str, ground_truth, predictions):
    """   THIS IS OUTDATED AND NO LONGER USED. Istead we store all queries in one dct
    for given queries: store the scores for each prediction in a file.
    query_id: e.g. sid_rid_xxx_ts, sid_xxx_oid_ts, sid_rid_xxx_ts, sid_rid_oid_xxx
    store dictionary, with query_id = name    
    dict with keys: 'predictions' 'ground_truth' and values: tensors
    :param directory: [str] directory, usually e.g. '/home/jgastinger/tempg/Baseline Evaluation'
    :param method_name: [str] e.g. renet
    :param query_name: [str] e.g. "xxx_1_235_24" -> the xxx is the element in question; order: subid_relid_obid_timestep
    :param ground truth: tensor e.g. tensor(4759, device='cuda:0') tensor with the id of the ground truth node
    :param predictions: tensor with predicted scores, one per node; e.g. tensor([ 5.3042,  6....='cuda:0') torch.Size([23033])
    """
    method_name =method_name
    query_name = query_name
    ground_truth =ground_truth
    predictions = predictions
    name = method_name + '_' + query_name +'.pt'
    dir_results = directory + '/resultscores' + '/' + dataset_name
    location = dir_results + '/' + name
    
    torch.save({"ground_truth": ground_truth, "predictions":predictions}, location)
    #https://stackoverflow.com/questions/62932368/best-way-to-save-many-tensors-of-different-shapes


def create_scores_tensor(predictions_dict, num_nodes, device=None):
    """ for given dict with key: node id, and value: score -> create a tensor with num_nodes entries, where the score 
    from dict is enetered at respective place, and all others are zeros.

    :returns: predictions  tensor with predicted scores, one per node; e.g. tensor([ 5.3042,  6....='cuda:0') torch.Size([23033])
    """
    predictions = torch.zeros(num_nodes, device=device)
    for node_id in predictions_dict.keys():
        predictions[node_id] = predictions_dict[node_id]
    return predictions

def query_name_from_quadruple_cygnet(quad, ob_pred=True):
    """ get the query namefrom the given quadruple. 
    :param quad: numpy array, len 4: [sub, rel, ob, ts]; 
    :param ob_pred: [bool] true: the object is predicted, false: the subject is predicted
    :return: 
    query_name [str]: name of the query, with xxx showing the entity of interest. e.g.'30_13_xxx_334' for 
        object prediction or 'xxx_13_18_334' for subject prediction
    test_query_ids [np array]: sub, rel, ob, ts (original rel id)
    """
    rel = quad[1]
    ts = quad[3]
    sub = quad[0]
    ob = quad[2]
    
    if ob_pred == True:
        query_name = str(sub) + '_' + str(rel) + '_' + 'xxx'+ str(ob) +'_' + str(ts)
    else:
        query_name = 'xxx'+ str(sub)+ '_' + str(rel) + '_' + str(ob) + '_'  + str(ts)

    test_query_ids = np.array([sub, rel, ob, ts])

    return query_name, test_query_ids

def query_name_from_quadruple(quad, num_rels, plus_one_flag=False):
    """ get the query namefrom the given quadruple. if they do reverse prediction with nr*rel+rel_id then we undo it here
    :param quad: numpy array, len 4: [sub, rel, ob, ts]; if rel>num_rels-1: this means inverse prediction
    :param num_rels: [int] number of relations
    :param plus_one_flag: [Bool] if the number of relations for inverse predictions is one higher than expected - the case for timetraveler:self.quadruples.append([ex[2], ex[1]+num_r+1, ex[0], ex[3]]
    :return: 
    query_name [str]: name of the query, with xxx showing the entity of interest. e.g.'30_13_xxx_334' for 
        object prediction or 'xxx_13_18_334' for subject prediction
    test_query_ids [np array]: sub, rel, ob, ts (original rel id)
    """
    rel = quad[1]
    ts = quad[3]
    if rel > (num_rels-1): #FALSCH RUM
        
        ob_pred = False
        if plus_one_flag == False:
            rel = rel - (num_rels) 
        else:
            rel = rel - (num_rels) -1 
        sub = quad[2]
        ob = quad[0]
    else:
        ob_pred = True
        sub = quad[0]
        ob = quad[2]      
    
    if ob_pred == True:
        query_name = str(sub) + '_' + str(rel) + '_' + 'xxx'+ str(ob) +'_' + str(ts)
    else:
        query_name = 'xxx'+ str(sub)+ '_' + str(rel) + '_' + str(ob) + '_'  + str(ts)
    
    test_query_ids = np.array([sub, rel, ob, ts])
    return query_name, test_query_ids

def get_total_number(inPath, fileName):
    """ return number of nodes and number of relations
    from renet utils.py
    """
    import os
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])

def update_scores(query, results, scores_dict={}):
    
    pass

def load_scores(directory:str, method_name:str, dataset:str, query_name:str, device:str):
    
    """ load scores from pt file from folder resultscores
    scores_dict is a dict with keys: 'predictions' 'ground_truth' and values: tensors
    for the predictions it is a tensor with a score for each node -> e.g. 23000 entries for ICEWS18
    for the ground truth it is one entry with the ground truth node
    :param directory: [str] directory, usually e.g. '/home/jgastinger/tempg/Baseline Evaluation'
    :param method_name: [str] e.g. renet
    :param dataset: [str] e.g. ICEWS18
    :param query_name: [str] e.g. "xxx_1_235_24" -> the xxx is the element in question; order: subid_relid_obid_timestep
    :returns: scores: tensor with predicted scores, one per node; gt: tensor with the id of the ground truth node
    """
    print("HI")
    dir_results = directory + '/resultscores'+ '/' + dataset
    name = method_name  + query_name +'.pt'
    location = dir_results + '/' + name
    scores_dict = torch.load(location, map_location=torch.device(device))
    scores = scores_dict['predictions']
    gt = scores_dict['ground_truth']
    return scores, gt

def compute_ranks():
    #for the different filter settings
    pass

def load_test_set():
    pass

def plot_results():
    pass


"""/*
 *    Utils for Testing for TKG Forecasting
 *
    Subset of utils function from RE-GCN source code (only keeping the relevant parts)
    https://github.com/Lee-zix/RE-GCN/blob/master/rgcn/utils.py
    Zixuan Li, Xiaolong Jin, Wei Li, Saiping Guan, Jiafeng Guo, Huawei Shen, Yuanzhuo Wang and Xueq
 *
 *     
"""

"""
Utility functions for link prediction
Most code is adapted from authors' implementation of RGCN link prediction:
https://github.com/MichSchli/RelationPrediction

"""
import os.path
import pandas as pd

import numpy as np
import torch
import logging
import sys
sys.path.append("..")
import rgcn.knowledge_graph as knwlgrh
import json

#######################################################################
#
# Utility function for building training and testing graphs
#
#######################################################################


def load_data(dataset, directory, bfs_level=3, relabel=False):
    if dataset in ['ICEWS18', 'ICEWS14', "GDELT", "ICEWS14", "ICEWS05-15","YAGO",
                     "WIKI", 'crisis2022']:
        # path = os.path.join(os.getcwd(), 'data', directory)
        path = './data' #, directory)
        return knwlgrh.load_from_local(path, dataset)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))


def load_all_answers_for_filter(total_data, num_rel, rel_p=False):
    # store subjects for all (rel, object) queries and
    # objects for all (subject, rel) queries
    def add_relation(e1, e2, r, d):
        if not e1 in d:
            d[e1] = {}
        if not e2 in d[e1]:
            d[e1][e2] = set()
        d[e1][e2].add(r)

    all_ans = {}
    for line in total_data:
        s, r, o = line[: 3]
        if rel_p:
            add_relation(s, o, r, all_ans)
            add_relation(o, s, r + num_rel, all_ans)
        else:
            add_subject(s, o, r, all_ans, num_rel=num_rel)
            add_object(s, o, r, all_ans, num_rel=0)
    return all_ans


def load_all_answers_for_time_filter(total_data, num_rels, num_nodes, rel_p=False):
    all_ans_list = []
    all_snap = split_by_time(total_data)
    for snap in all_snap:
        all_ans_t = load_all_answers_for_filter(snap, num_rels, rel_p)
        all_ans_list.append(all_ans_t)

    return all_ans_list

def add_subject(e1, e2, r, d, num_rel):
    if not e2 in d:
        d[e2] = {}
    if not r+num_rel in d[e2]:
        d[e2][r+num_rel] = set()
    d[e2][r+num_rel].add(e1)


def add_object(e1, e2, r, d, num_rel):
    if not e1 in d:
        d[e1] = {}
    if not r in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)


def split_by_time(data):
    snapshot_list = []
    snapshot = []
    snapshots_num = 0
    latest_t = 0
    for i in range(len(data)):
        t = data[i][3]
        train = data[i]
        if latest_t != t:  
            # show snapshot
            latest_t = t
            if len(snapshot):  # appends in the list lazily i.e. when new timestamp is observed
                # load the previous batch and empty the cache
                snapshot_list.append(np.array(snapshot).copy())
                snapshots_num += 1
            snapshot = []
        snapshot.append(train[:3])

    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
        snapshots_num += 1

    union_num = [1]
    nodes = []
    rels = []
    # Loops only for sanity check
    for snapshot in snapshot_list:
        uniq_v, edges = np.unique((snapshot[:,0], snapshot[:,2]), return_inverse=True)  # edges are indices of combined (src, dst)
        uniq_r = np.unique(snapshot[:, 1])
        edges = np.reshape(edges, (2, -1))  #FIXME: unused just like in RENET
        nodes.append(len(uniq_v))
        rels.append(len(uniq_r)*2)
    print("# Sanity Check:  ave node num : {:04f}, ave rel num : {:04f}, snapshots num: {:04d}, max edges num: {:04d}, min edges num: {:04d}, max union rate: {:.4f}, min union rate: {:.4f}"
          .format(np.average(np.array(nodes)), np.average(np.array(rels)), len(snapshot_list), max([len(_) for _ in snapshot_list]), min([len(_) for _ in snapshot_list]), max(union_num), min(union_num)))
    return snapshot_list

def stat_ranks(rank_list, method, mode, mrr_snapshot_list):
    hits = [1, 3, 10]
    total_rank = torch.cat(rank_list)
    mr = torch.mean(total_rank.float())
    mrr = torch.mean(1.0 / total_rank.float())
    print("MR ({}): {:.6f}".format(method, mr.item()))
    print("MRR ({}): {:.6f}".format(method, mrr.item()))

    if mode == 'test':
        logging.debug("MR ({}): {:.6f}".format(method, mr.item()))
        logging.debug("MRR ({}): {:.6f}".format(method, mrr.item()))
        # logging.debug("MRR over time ({}): {:.6f}".format(method, mrr_snapshot_list))
    hit_scores = []
    for hit in hits:
        avg_count = torch.mean((total_rank <= hit).float())
        print("Hits ({}) @ {}: {:.6f}".format(method, hit, avg_count.item()))
        if mode == 'test':
            logging.debug("Hits ({}) @ {}: {:.6f}".format(method, hit, avg_count.item()))
            hit_scores.append(avg_count.item())
    return (mr.item(), mrr.item(), hit_scores, mrr_snapshot_list)


def flatten_list(input_list):
    flat_list = []
    for item in input_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        elif isinstance(item, torch.Tensor):
            flat_list.extend(item.view(-1).tolist())
        else:
            flat_list.append(item)
    return flat_list

def store_ranks(rank_list, directory):
    rank_list = flatten_list(rank_list)

    with open(directory+'list_data_3.json', 'w') as file:
        json.dump(rank_list, file) 



def get_total_rank(test_triples, score, all_ans, all_ans_static, eval_bz, rel_predict=0, ts=0, df_for_results=None):
    '''

    :param test_triples: triples with inverse relationship.
    :param score:
    :param all_ans: dict with [s,o]:rel:[o,s] or [s,o]:[o,s]:rel per timestamp.
    :param all_ans_static: dict with [s,o]:rel:[o,s] or [s,o]:[o,s]:rel, timestep independent
    :param eval_bz: evaluation batch size
    :param rel_predict: if 1 predicts relations/link prediction otherwise entity prediction.
    :return:
    '''
    triples_and_scores_df = {}
     
    num_triples = len(test_triples)
    n_batch = (num_triples + eval_bz - 1) // eval_bz
    rank = []
    filter_t_rank = []
    filter_s_rank = []
    for idx in range(n_batch):
        batch_start = idx * eval_bz
        batch_end = min(num_triples, (idx + 1) * eval_bz)
        triples_batch = test_triples[batch_start:batch_end, :]
        score_batch = score[batch_start:batch_end, :]
        # print(rel_predict)
        if rel_predict == 1:
            target = test_triples[batch_start:batch_end, 1]
        elif rel_predict == 2:
            target = test_triples[batch_start:batch_end, 0]
        else:
            target = test_triples[batch_start:batch_end, 2]
        # raw:
        batch_r_rank, _ = sort_and_rank(score_batch, target)
        rank.append(batch_r_rank)

        # time aware filter
        if rel_predict == 1:
            filter_score_batch_t = filter_score_r(triples_batch, score_batch, all_ans)
        else:
            filter_score_batch_t = filter_score(triples_batch, score_batch, all_ans)
            c = copy(filter_score_batch_t)
        batch_t_rank, sorted_scores_t = sort_and_rank(filter_score_batch_t, target)
        # batch_t_rank = sort_and_rank(filter_score_batch_t, target)
        filter_t_rank.append(batch_t_rank)

        

        # static filter
        if rel_predict:  # if rel_predict == 1
            filter_score_batch_s = filter_score_r(triples_batch, score_batch, all_ans_static)
        else:
            filter_score_batch_s = filter_score(triples_batch, score_batch, all_ans_static)
        batch_s_rank, _ =  sort_and_rank(filter_score_batch_s, target)
        filter_s_rank.append(batch_s_rank)

        for index in range(len(triples_batch)):
            rplus = batch_t_rank[index]+1
            mrr = (1.0/rplus.float()).item()
            # b = copy(filter_score_batch_t[index])
            # sorted_preds = b.sort(descending=True)[1][:10].numpy() #the ten predictions with the highest scores, after filtering
            sorted_preds = sorted_scores_t[index][:10].numpy()
            # line = {'s': triples_batch[index][0].item(), 'r': triples_batch[index][1].item(), 'o': triples_batch[index][2].item(), 't': ts, 'rank':mrr, 'pred': torch.argmax(filter_score_batch_t[index]).item() }
            line = {'s': triples_batch[index][0].item(), 'r': triples_batch[index][1].item(), 'o': triples_batch[index][2].item(), 't': ts, 'rank':mrr, 'pred0': int(sorted_preds[0]),
                    'pred1': int(sorted_preds[1]), 'pred2': int(sorted_preds[2]), 'pred3': int(sorted_preds[3]), 'pred4': int(sorted_preds[4]), 'pred5': int(sorted_preds[5]), 'pred6': int(sorted_preds[6]),
                    'pred7': int(sorted_preds[7]),
                    'pred8': int(sorted_preds[8]), 'pred9': int(sorted_preds[9]) }
            new_df = pd.DataFrame([line])
            df_for_results = pd.concat([df_for_results, new_df], ignore_index=True)
            # key = str(triples_batch[index][0])+'_'+str(triples_batch[index][0])+'_'+str(triples_batch[index][0])+'_'+str(ts)
            # triples_and_scores_df[triples_batch[index]] = {'rank': batch_t_rank[index], 'pred': torch.argmax(filter_score_batch_t[index]), 'triple': triples_batch[index], 'ts': ts}
    rank = torch.cat(rank)
    filter_t_rank = torch.cat(filter_t_rank)
    filter_s_rank = torch.cat(filter_s_rank)

    rank += 1 # change to 1-indexed
    filter_t_rank += 1
    filter_s_rank += 1

    mrr = torch.mean(1.0 / rank.float())
    filter_t_mrr = torch.mean(1.0 / filter_t_rank.float())
    filter_s_mrr = torch.mean(1.0 / filter_s_rank.float())

    return filter_s_mrr.item(), filter_t_mrr.item(), mrr.item(), rank, filter_t_rank, filter_s_rank, df_for_results


def filter_score(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t = triple
        # try:
        ans = list(all_ans[h.item()][r.item()])
        ans.remove(t.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #
        # except:
        #     print('KeyError in all_ans')

    return score

def filter_score_r(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t = triple
        ans = list(all_ans[h.item()][t.item()])
        ans.remove(r.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #
    return score

def sort_and_rank(score, target):
    sorted, indices = torch.sort(score, dim=1, descending=True) # with default: stable=False; pytorch docu: "If stable is True then the sorting routine becomes stable, preserving the order of equivalent elements."
    indices_new = torch.nonzero(indices == target.view(-1, 1), as_tuple=False)
    indices_new = indices_new[:, 1].view(-1)
    return indices_new, indices



