"""*
 *     Static Baseline
 *
 *        File: utils.py
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


import numpy as np
import torch
import logging
def add_inverse_triples(triples: np.array, num_rels:int) -> np.array:
    inverse_triples = triples[:, [2, 1, 0]]
    inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels  # we also need inverse triples
    all_triples = np.concatenate((triples[:,0:3], inverse_triples))

    return all_triples

def add_inverse_quadruples(triples: np.array, num_rels:int) -> np.array:
    inverse_triples = triples[:, [2, 1, 0, 3]]
    inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels  # we also need inverse triples
    all_triples = np.concatenate((triples, inverse_triples))

    return all_triples



def has_nonempty_list(lst) -> bool: 
    # to check if we have any path >0
    # for a given list lst, which contains lists: check if any of the lists has length >0. if yes: return true.
    for sublist in lst:
        if len(sublist) > 0:
            return True
    return False


def reverse_dict_keys(orig_dict):
    reverse_dict = {}
    for key_a, inner_dict in orig_dict.items():
        for key_b, list_c in inner_dict.items():
            if key_b not in reverse_dict:
                reverse_dict[key_b] = {}
            reverse_dict[key_b][key_a] = list_c
    return reverse_dict



def get_smaller_ints(sorted_list, target):
    """
    return all ints that are smaller than a certain target int in a new list
    """
    sorted_arr = np.array(sorted_list)
    smaller_arr = sorted_arr[sorted_arr < target]
    return smaller_arr.tolist()




#### compute MRRs
## all methods below taken or modified from https://github.com/Lee-zix/RE-GCN
# Zixuan Li, Xiaolong Jin, Wei Li, Saiping Guan, Jiafeng Guo, Huawei Shen, Yuanzhuo Wang and Xueqi Cheng. 
# Temporal Knowledge Graph Reasoning Based on Evolutional Representation Learning. SIGIR 2021.

def compute_timefilter_hits(scores_dict, timesteps_test, test_data, num_nodes, num_relations, mask_flag):
    all_ans_list_test, test_data_snaps = load_all_answers_for_time_filter(test_data, num_relations, num_nodes, False)
    scores_t_filter, scores_raw = compute_testscores(timesteps_test, test_data_snaps, scores_dict, all_ans_list_test, mask_flag)

    return scores_t_filter, scores_raw

def load_all_answers_for_filter(total_data, num_rel, rel_p=False):
    # from RE-GCN
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

def compute_testscores(timesteps_test, test_data, final_scores, all_ans_list_test, mask_flag):
    ranks_raw, ranks_t_filter, mrr_raw_list, mrr_t_filter_list= [], [], [], []
    assert len(timesteps_test) == len(test_data) == len(final_scores) == len(all_ans_list_test)

    timesteps = list(range(len(timesteps_test)))  # rename to match the standard of all_and_list_test
    for time_idx, test_triple, final_score in zip(timesteps, test_data, final_scores):
        if mask_flag:
            mask = (test_triple[:, 1] == 4)  | (test_triple[:, 1] == 8) | (test_triple[:, 1] == 23) | (test_triple[:, 1] == 27) 
            test_triple = test_triple[mask] #only triples with relation of interest
        if len(test_triple) > 0:
            mrr_t_filter_snap, mrr_snap, rank_raw, rank_t_filter = get_total_rank(
                test_triple, final_score,
                all_ans_list_test[time_idx],
                eval_bz=300, #1000,
                rel_predict=0)
            # used to global statistic
            ranks_raw.append(rank_raw)
            ranks_t_filter.append(rank_t_filter)

            # used to show slide results
            mrr_raw_list.append(mrr_snap)
            mrr_t_filter_list.append(mrr_t_filter_snap)


    mode = 'test'
    scores_raw = stat_ranks(ranks_raw, "Entity Prediction Raw", mode, mrr_raw_list) 
    #(mr.item(), mrr.item(), hit_scores, mrr_snapshot_list)
    scores_t_filter = stat_ranks(ranks_t_filter, "Entity TimeAware Prediction Filter", mode, mrr_t_filter_list) 
    #(mr.item(), mrr.item(), hit_scores, mrr_snapshot_list)
   
    return scores_t_filter, scores_raw

def load_all_answers_for_time_filter(total_data, num_rels, num_nodes, rel_p=False):
    all_ans_list = []
    all_snap = split_by_time(total_data, num_rels)
    for snap in all_snap:
        all_ans_t = load_all_answers_for_filter(snap, num_rels, rel_p)
        all_ans_list.append(all_ans_t)

    return all_ans_list, all_snap

def split_by_time(data, num_rels):
    # modified to take as input a dictionary
    snapshot_list = []
    snapshot = []
    snapshots_num = 0
    latest_t = 0
    for i in data.keys():
        t = i
        train = data[i]
        if latest_t != t:  
            # show snapshot
            latest_t = t
            if len(snapshot):  # appends in the list lazily i.e. when new timestamp is observed
                # load the previous batch and empty the cache
                snapshot_list.append(np.array(snapshot).copy().squeeze())
                snapshots_num += 1
            snapshot = []
        reverse = train[:, np.argsort([2,1,0])]
        reverse[:,1] = reverse[:,1]+num_rels
        back_forth = np.concatenate((train[:,0:3], reverse), axis=0)
        snapshot.append(back_forth)


    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy().squeeze())
        snapshots_num += 1

    union_num = [1]
    nodes = []
    rels = []
    # Loops only for sanity check
    for snapshot in snapshot_list:
        uniq_v, edges = np.unique((snapshot[:,0], snapshot[:,2]), return_inverse=True)  
        # edges are indices of combined (src, dst)
        uniq_r = np.unique(snapshot[:, 1])
        edges = np.reshape(edges, (2, -1))  #FIXME: unused just like in RENET
        nodes.append(len(uniq_v))
        rels.append(len(uniq_r)*2)
    print("# Sanity Check:  ave node num : {:04f}, ave rel num : {:04f}, snapshots num: {:04d}, max edges num: {:04d}, min edges num: {:04d}, max union rate: {:.4f}, min union rate: {:.4f}"
            .format(np.average(np.array(nodes)), np.average(np.array(rels)), len(snapshot_list), max([len(_) for _ in 
            snapshot_list]), min([len(_) for _ in snapshot_list]), max(union_num), min(union_num)))
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

def get_total_rank(test_triples, score, all_ans, eval_bz=1000, rel_predict=0):
    '''
    :param test_triples: triples with inverse relationship.
    :param score:
    :param all_ans: dict with [s,o]:rel:[o,s] or [s,o]:[o,s]:rel per timestamp.
    :param all_ans_static: dict with [s,o]:rel:[o,s] or [s,o]:[o,s]:rel, timestep independent
    :param eval_bz: evaluation batch size
    :param rel_predict: if 1 predicts relations/link prediction otherwise entity prediction.
    :return:
    '''
    num_triples = len(test_triples)
    n_batch = (num_triples + eval_bz - 1) // eval_bz
    rank = []
    filter_t_rank = []
    
    for idx in range(n_batch):
        batch_start = idx * eval_bz
        batch_end = min(num_triples, (idx + 1) * eval_bz)
        triples_batch = torch.tensor(test_triples[batch_start:batch_end, :], device = score.device)
        score_batch = score[batch_start:batch_end, :]
        # print(rel_predict)
        if rel_predict == 1:
            target = test_triples[batch_start:batch_end, 1]
        elif rel_predict == 2:
            target = test_triples[batch_start:batch_end, 0]
        else:
            target = test_triples[batch_start:batch_end, 2]
        # raw:
        rank.append(sort_and_rank(score_batch, target))

        # time aware filter
        if rel_predict == 1:
            filter_score_batch_t = filter_score_r(triples_batch, score_batch, all_ans)
        else:
            filter_score_batch_t = filter_score(triples_batch, score_batch, all_ans)
        filter_t_rank.append(sort_and_rank(filter_score_batch_t, target))


    rank = torch.cat(rank)
    filter_t_rank = torch.cat(filter_t_rank)
    

    rank += 1 # change to 1-indexed
    filter_t_rank += 1
   

    mrr = torch.mean(1.0 / rank.float())
    filter_t_mrr = torch.mean(1.0 / filter_t_rank.float())
    

    return filter_t_mrr.item(), mrr.item(), rank, filter_t_rank


def sort_and_rank(score, target):    
    _, indices = torch.sort(score, dim=1, descending=True) # with default: stable=False; pytorch docu: 
    # "If stable is True then the sorting routine becomes stable, preserving the order of equivalent elements."
    target = torch.tensor(target).to(indices.device)
    indices = torch.nonzero(indices == target.view(-1, 1), as_tuple=False)
    indices = indices[:, 1].view(-1)
    return indices

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

