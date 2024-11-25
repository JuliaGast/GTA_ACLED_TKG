"""*
 *     Static Baseline
 *
 *        File: evaluation_utils.py
 *
Authors: Julia Gastinger (julia.gastinger@uni-mannheim.de), Timo Sztyler (timo.sztyler@neclab.eu), 
Nils Steinert, Sabine Gruender-Fahrer, Michael Martin, Anett Schuelke, Heiner Stuckenschmidt 
  
NEC Laboratories Europe GmbH, Copyright (c) 2024, All rights reserved.  

THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
 
PROPRIETARY INFORMATION ---  

SOFTWARE LICENSE AGREEMENT

ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution
or non-profit organization or self (called "Licensee" or "You" in this
Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
Agreement).  All rights not specifically granted to you in this Agreement
are reserved for Licensor. 

RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
ownership of any copy of the Software (as defined below) licensed under this
Agreement and hereby grants to Licensee a personal, non-exclusive,
non-transferable license to use the Software for noncommercial research
purposes, without the right to sublicense, pursuant to the terms and
conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
Agreement, the term "Software" means (i) the actual copy of all or any
portion of code for program routines made accessible to Licensee by Licensor
pursuant to this Agreement, inclusive of backups, updates, and/or merged
copies permitted hereunder or subsequently supplied by Licensor,  including
all or any file structures, programming instructions, user interfaces and
screen formats and sequences as well as any and all documentation and
instructions related to it, and (ii) all or any derivatives and/or
modifications created or made by You to any of the items specified in (i).

CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
proprietary to Licensor, and as such, Licensee agrees to receive all such
materials and to use the Software only in accordance with the terms of this
Agreement.  Licensee agrees to use reasonable effort to protect the Software
from unauthorized use, reproduction, distribution, or publication. All
publication materials mentioning features or use of this software must
explicitly include an acknowledgement the software was developed by NEC
Laboratories Europe GmbH.

COPYRIGHT: The Software is owned by Licensor.  

PERMITTED USES:  The Software may be used for your own noncommercial
internal research purposes. You understand and agree that Licensor is not
obligated to implement any suggestions and/or feedback you might provide
regarding the Software, but to the extent Licensor does so, you are not
entitled to any compensation related thereto.

DERIVATIVES: You may create derivatives of or make modifications to the
Software, however, You agree that all and any such derivatives and
modifications will be owned by Licensor and become a part of the Software
licensed to You under this Agreement.  You may only use such derivatives and
modifications for your own noncommercial internal research purposes, and you
may not otherwise use, distribute or copy such derivatives and modifications
in violation of this Agreement.

BACKUPS:  If Licensee is an organization, it may make that number of copies
of the Software necessary for internal noncommercial use at a single site
within its organization provided that all information appearing in or on the
original labels, including the copyright and trademark notices are copied
onto the labels of the copies.

USES NOT PERMITTED:  You may not distribute, copy or use the Software except
as explicitly permitted herein. Licensee has not been granted any trademark
license as part of this Agreement.  Neither the name of NEC Laboratories
Europe GmbH nor the names of its contributors may be used to endorse or
promote products derived from this Software without specific prior written
permission.

You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
whole or in part, or provide third parties access to prior or present
versions (or any parts thereof) of the Software.

ASSIGNMENT: You may not assign this Agreement or your rights hereunder
without the prior written consent of Licensor. Any attempted assignment
without such consent shall be null and void.

TERM: The term of the license granted by this Agreement is from Licensee's
acceptance of this Agreement by downloading the Software or by using the
Software until terminated as provided below.  

The Agreement automatically terminates without notice if you fail to comply
with any provision of this Agreement.  Licensee may terminate this Agreement
by ceasing using the Software.  Upon any termination of this Agreement,
Licensee will delete any and all copies of the Software. You agree that all
provisions which operate to protect the proprietary rights of Licensor shall
remain in force should breach occur and that the obligation of
confidentiality described in this Agreement is binding in perpetuity and, as
such, survives the term of the Agreement.

FEE: Provided Licensee abides completely by the terms and conditions of this
Agreement, there is no fee due to Licensor for Licensee's use of the
Software in accordance with this Agreement.

DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
RELATED MATERIALS.

SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
provided as part of this Agreement.  

EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
permitted under applicable law, Licensor shall not be liable for direct,
indirect, special, incidental, or consequential damages or lost profits
related to Licensee's use of and/or inability to use the Software, even if
Licensor is advised of the possibility of such damage.

EXPORT REGULATION: Licensee agrees to comply with any and all applicable
export control laws, regulations, and/or other laws related to embargoes and
sanction programs administered by law.

SEVERABILITY: If any provision(s) of this Agreement shall be held to be
invalid, illegal, or unenforceable by a court or other tribunal of competent
jurisdiction, the validity, legality and enforceability of the remaining
provisions shall not in any way be affected or impaired thereby.

NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
or remedy under this Agreement shall be construed as a waiver of any future
or other exercise of such right or remedy by Licensor.

GOVERNING LAW: This Agreement shall be construed and enforced in accordance
with the laws of Germany without reference to conflict of laws principles.
You consent to the personal jurisdiction of the courts of this country and
waive their rights to venue outside of Germany.

ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
entire agreement between Licensee and Licensor as to the matter set forth
herein and supersedes any previous agreements, understandings, and
arrangements between the parties relating hereto.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.


 */"""



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



