"""*
 *     Static Baseline
 *
 *        File: main.py
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


import hydra 
import os
from omegaconf import DictConfig
import torch
from time import gmtime, strftime
import utils
import numpy as np

import data.data_handler as data_handler

from engine.engine import Engine
from utils import add_inverse_triples

from time import gmtime, strftime, ctime
import numpy as np
# import time
import json
import export_results_local

@hydra.main(version_base=None, config_path='', config_name="conf")
def main(config:DictConfig):
    
    device = torch.device('cuda:' + str(config.background.device) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) #SET DEFAULT DEVICE FOR TORCH

    time_format = strftime("%a_%d_%b_%Y_%H_%M_%S", gmtime())
    result_file_name = config.background.result_file_name+time_format

    dataset_name = config.dataset.name
    dataset = (dataset_name, 3) # identifier, timestamp_column_idx

    

    ## load dataset & group by timestamp
    train_quads, valid_quads, test_quads, stat = data_handler.load(dataset[0])
    train_dict, valid_dict, test_dict = [data_handler.group_by(entry, dataset[1]) for entry in [train_quads, valid_quads, test_quads]] 

    ## Preprocess. create or load anchors and paths 
    timestep_node_rel_dict_train = data_handler.compute_relations_dict(train_dict, stat[1])
    timestep_node_rel_dict_valid = data_handler.compute_relations_dict(valid_dict, stat[1])
    timestep_node_rel_dict_valid[np.max(list(timestep_node_rel_dict_train.keys()))] = timestep_node_rel_dict_train[np.max(list(timestep_node_rel_dict_train.keys()))]
    timestep_node_rel_dict_test = data_handler.compute_relations_dict(test_dict, stat[1])
    timestep_node_rel_dict_test[np.max(list(timestep_node_rel_dict_valid.keys()))] = timestep_node_rel_dict_valid[np.max(list(timestep_node_rel_dict_valid.keys()))]

    ## Init engine 
    engine = Engine(config, num_nodes=int(stat[0]), num_relations=int(stat[1]), device=device, datasetid=dataset_name) # init engine & model

    ## Train or Load pretrained Model
    if config.background.train_flag:
        logging_dict, name, H_t = engine.train(train_dict, valid_dict, timestep_node_rel_dict_train, timestep_node_rel_dict_valid, device, result_file_name)
        ## load best model from file
        logging_dict, result_file_name =  engine.load_model_from_file(device, name)
    if config.background.load_flag:
        ## load best model from file
        name = config.background.load_model_name
        logging_dict, result_file_name =  engine.load_model_from_file(device, name)
        print("TODO: implement loading the pretrained model")
        logging_dict, name, H_t = engine.train_nobackprob(train_dict, valid_dict, timestep_node_rel_dict_train, timestep_node_rel_dict_valid, device, result_file_name)
        name = config.background.load_model_name



    ## Test
    engine.test(train_dict, valid_dict, test_dict,timestep_node_rel_dict_test, H_t, result_file_name)


    # print("finish to test at: ", ctime() )

    ## write results
    logging_dict['finish_time'] = str(ctime())

    with open(result_file_name+'.json', 'w') as fp: # We also dump this after each validation epoch
        json.dump(logging_dict, fp)

    export_results_local.write(config, result_file_name)
    print("wrote results from ", result_file_name)
    print("time:" , strftime("%a_%d_%b_%Y_%H_%M_%S", gmtime()))

    return logging_dict['best_valid_mrr']

    print('Done.')

 














if __name__ == "__main__":
    main()
