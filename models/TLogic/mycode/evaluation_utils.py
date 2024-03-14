import torch
import numpy as np
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