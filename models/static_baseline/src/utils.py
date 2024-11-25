"""*
 *     Static Baseline
 *
 *        File: utils.py
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

