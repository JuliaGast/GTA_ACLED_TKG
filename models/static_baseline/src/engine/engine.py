"""*
 *     Static Baseline
 *
 *        File: engine.py
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


from engine.baseline_model import Simple_Model

from torch.utils.tensorboard import SummaryWriter
import random
from tqdm import tqdm
import time
import torch
import json
import os
import pathlib
import pickle

from time import gmtime, strftime

import utils 
import numpy as np
from copy import copy

MASK = False

class Engine():
    """ The training and testing process manager """ 
    def __init__(self, config,  num_nodes:int, num_relations:int, device:str, datasetid:str):

        
        self.logging_dict = {}
        self._num_nodes = num_nodes
        self._num_relations = num_relations

        self.dump_results_pkl = config.model.dump_results_flag

        self.model = Simple_Model(num_nodes, num_relations, config.model.embedding_dim, device, config.model)
            # dataset_name:str, experts:list, num_nodes,                 



        self._dataset_identifier = datasetid
        self._tests_file_pth = os.path.join(pathlib.Path().resolve(), 'tests', self._dataset_identifier)
        if not os.path.exists(self._tests_file_pth):
            os.makedirs(self._tests_file_pth)
        self.model_file_dir = os.path.join(self._tests_file_pth, 'models')
        if not os.path.exists(self.model_file_dir):
            os.makedirs(self.model_file_dir)
        


        time_format = strftime("%a_%d_%b_%Y_%H_%M_%S", gmtime())
        self.name = datasetid + '_'  + '_' + time_format

        self.model_file_pth = os.path.join(self.model_file_dir, self.name)

        self.device = device

        self.num_epochs = config.training.epochs
        self.evaluate_every = config.training.evaluate_every

        # loss
        loss_type = config.training.loss_type
        if loss_type == 'cross_entropy':
            self.loss = torch.nn.CrossEntropyLoss()  # torch.nn.BCELoss()
        else:
            raise Exception('Invalid Loss type')

        # optimizer
        if config.training.algorithm == 'adam':
            optimizer = torch.optim.Adam
        elif config.training.algorithm == 'adamw':
            optimizer = torch.optim.AdamW
        elif config.training.algorithm == 'adagrad':
            optimizer = torch.optim.Adagrad
        elif config.training.algorithm == 'sgd':
            optimizer = torch.optim.SGD
        else:
            raise Exception('Invalid Optimizer')

        wd = float(config.training.weight_decay)
        lr = float(config.training.learn_rate)
        self.optimizer = optimizer(self.model.parameters(), weight_decay= wd, lr=lr)

        # evaluation
        self.early_stopping = config.evaluation['early_stopping']
        self.logging_dict = {'start_time': str(time.ctime()),
                             'device': str(self.device),
                             'mean_train_loss': None,
                             'mean_valid_loss': None,
                             'mean_train_loss_e': None,
                             'mean_valid_loss_e': None,
                             'mean_train_loss_r': None,
                             'mean_valid_loss_r': None,
                             'highest_epoch': 0,
                             'best_epoch': 0,
                             'best_valid_mrr': 0,
                             'best_valid_hits': 0,
                             'best_test_mrr': 0,
                             'best_test_hits': 0,
                             'model_name': self.name,
                             'finish_time': None,
                             'tensorboard_name': None}


        self.h_zero_matrix= torch.zeros((1, self._num_nodes, self.model.embedding_dim), dtype=torch.float32, device=device)




    def train(self,  train_data: dict, valid_data: dict,  timestep_node_rel_dict_train:dict, 
                timestep_node_rel_dict_valid:dict, device: str, result_file_name:str):
        """
        training process of model. includes validation
        :param train_data: [dict]: keys: timesteps, values: [array]  triples for this timestep; train
        :param valid_data: [dict]:  keys: timesteps, values: [array] triples for this timestep; valid
        :param device: [str]: device (e.g. device(type='cuda', index=2))
        """

        timesteps_train = list(train_data.keys())
        timesteps_valid = list(valid_data.keys())

        valid_losses_mean = []
        valid_losses = {}
        best_epoch = 0
        best_valid_mrr = 0
        best_valid_hits = 0

        writer = SummaryWriter(comment=self.name)  # setup the tensorboard writer path
        self.logging_dict['tensorboard_name'] = list(writer.all_writers.keys())[0]  # name of the tensorboard run

        bar_format = '{l_bar}{bar:30}{r_bar}{bar:-10b}'  # for writing progress
        best_Ht_valid = copy(self.h_zero_matrix)

        for epoch in range(1, self.num_epochs + 1):
            bar_name = '> Epoch ' + str(epoch)  # for writing progress
            self.model.train()
            losses = []

            timesteps_train_one = [i for i in timesteps_train if i>np.min(timesteps_train)]
            # random.shuffle(timesteps_shuffled)
            
            H_tminus = copy(self.h_zero_matrix)
            
            nantimesteps =[]
            with tqdm(timesteps_train_one, bar_name, bar_format=bar_format, unit='batch') as timestep_epoch:
                
                for t in timestep_epoch:
                    time_start = time.time()
                    self.optimizer.zero_grad()  # in each train step we want to compute new gradients -

                    # 0) data prep. including inverse triples in out graph
                    out_timestep, in_timesteps, out_graph, most_recent_in_graph_all = self.data_prep_t(t, 
                                                                                                       timesteps_train, 
                                                                                                       train_data)

                    # 1,2,3) model forward; compute scores for timestep t plus for all triples in out_graph for all possible nodes
                    if MASK:
                        mask = (out_graph[:, 1] == 4)  | (out_graph[:, 1] == 8) | (out_graph[:, 1] == 4+self._num_relations) | (out_graph[:, 1] == 8+self._num_relations) 
                        out_graph = out_graph[mask]
                    
                    out_queries = out_graph[:,0:2]  # without ground truth
                    rels_per_node = timestep_node_rel_dict_train[in_timesteps[-1]]
                    scores_tplus, H_t = self.model.forward(out_timestep, in_timesteps, rels_per_node, most_recent_in_graph_all, out_queries, H_tminus)

                    # 4) compute loss for t+1; optional: compute train mrr
                    loss = self.loss(scores_tplus,out_graph[:, 2].type(torch.long)) #.to('cpu'))  
             


                    # 5) backprop
                    if torch.isnan(loss) == False:
                        losses.append(loss.item())
                        timestep_epoch.set_postfix(loss=loss.item())
                        loss.backward() 
                        self.optimizer.step()
                    else:
                        nantimesteps.append(t)
                    H_tminus = copy(H_t)

            # # 6) logging and writing
            writer.add_scalar('Mean training loss', np.mean(losses), epoch)
            self.logging_dict['mean_train_loss'] = np.mean(losses)
            self.logging_dict['highest_epoch'] = epoch

            # 7) validation
            if epoch % self.evaluate_every == 0:
                self.model.eval()
                valid_losses_e, valid_hits, H_t = self.validate(timesteps_train, train_data, timesteps_valid, 
                                                           valid_data, timestep_node_rel_dict_valid, H_tminus, epoch)
                valid_loss = np.mean(valid_losses_e)  # self.w_e*np.mean(valid_losses_e) + self.w_r*np.mean(valid_losses_r)
                valid_losses_mean.append(valid_loss)

                valid_losses[epoch] = valid_loss
                writer.add_scalar('Mean validation loss', np.mean(valid_losses_mean), epoch)
                writer.add_scalar('Validation mrr', valid_hits[1], epoch)

                tmp2 = self.check_best_epoch(result_file_name, epoch, loss, losses, valid_loss, 
                                                valid_losses_mean, valid_hits, best_epoch, best_valid_mrr, 
                                                best_valid_hits, H_tminus, best_Ht_valid)
                best_epoch, best_valid_mrr, best_valid_hits, early_stopping_flag, best_Ht_valid = tmp2
                if early_stopping_flag:
                    # stop training loop because early stopping criterion applies
                    break

        writer.close()
        print("finished training, the latest loss was: ", loss.item())
        return self.logging_dict, self.name, best_Ht_valid 

    def train_nobackprob(self,  train_data: dict, valid_data: dict,  timestep_node_rel_dict_train:dict, 
                timestep_node_rel_dict_valid:dict, device: str, result_file_name:str):
        """
        to still get the H_t for the first testing step.
        training process of model. includes validation
        :param train_data: [dict]: keys: timesteps, values: [array]  triples for this timestep; train
        :param valid_data: [dict]:  keys: timesteps, values: [array] triples for this timestep; valid
        :param device: [str]: device (e.g. device(type='cuda', index=2))
        """

        timesteps_train = list(train_data.keys())
        timesteps_valid = list(valid_data.keys())


        writer = SummaryWriter(comment=self.name)  # setup the tensorboard writer path
        self.logging_dict['tensorboard_name'] = list(writer.all_writers.keys())[0]  # name of the tensorboard run

        bar_format = '{l_bar}{bar:30}{r_bar}{bar:-10b}'  # for writing progress
        best_Ht_valid = copy(self.h_zero_matrix)

        for epoch in range(1, 2):
            bar_name = '> Epoch ' + str(epoch)  # for writing progress
            self.model.train()
            losses = []

            timesteps_train_one = [i for i in timesteps_train if i>0]
            # random.shuffle(timesteps_shuffled)
            
            H_tminus = copy(self.h_zero_matrix)
            
            nantimesteps =[]
            with tqdm(timesteps_train_one, bar_name, bar_format=bar_format, unit='batch') as timestep_epoch:
                
                for t in timestep_epoch:
                    time_start = time.time()
                    self.optimizer.zero_grad()  # in each train step we want to compute new gradients -

                    # 0) data prep. including inverse triples in out graph
                    out_timestep, in_timesteps, out_graph, most_recent_in_graph_all = self.data_prep_t(t, 
                                                                                                       timesteps_train, 
                                                                                                       train_data)

                    # 1,2,3) model forward; compute scores for timestep t plus for all triples in out_graph for all possible nodes
                    if MASK:
                        mask = (out_graph[:, 1] == 4)  | (out_graph[:, 1] == 8) | (out_graph[:, 1] == 4+self._num_relations) | (out_graph[:, 1] == 8+self._num_relations) 
                        out_graph = out_graph[mask]
                    out_queries = out_graph[:,0:2]  # without ground truth
                    rels_per_node = timestep_node_rel_dict_train[in_timesteps[-1]]
                    scores_tplus, H_t = self.model.forward(out_timestep, in_timesteps, rels_per_node, most_recent_in_graph_all, out_queries, H_tminus)

                    # 4) compute loss for t+1; optional: compute train mrr
                    loss = self.loss(scores_tplus,out_graph[:, 2].type(torch.long)) #.to('cpu'))  

                    # 5) backprop
                    if torch.isnan(loss) == False:
                        losses.append(loss.item())
                        timestep_epoch.set_postfix(loss=loss.item())
                        loss.backward() 
                        self.optimizer.step()
                    else:
                        nantimesteps.append(t)
                    H_tminus = copy(H_t)


            # 7) validation
            # if epoch % self.evaluate_every == 0:
            self.model.eval()
            valid_losses_e, valid_hits, H_t = self.validate(timesteps_train, train_data, timesteps_valid, 
                                                        valid_data, timestep_node_rel_dict_valid, H_tminus, epoch)
            valid_loss = np.mean(valid_losses_e)  # self.w_e*np.mean(valid_losses_e) + self.w_r*np.mean(valid_losses_r)
            best_Ht_valid =  H_t 



        writer.close()
        print("finished training, the latest loss was: ", loss.item())
        return self.logging_dict, self.name, best_Ht_valid 
    
    def validate(self, timesteps_train, train_data, timesteps_valid, valid_data, timestep_node_rel_dict_valid, H_tminus, epoch):
        """
        validation

        for each timestep from validation set:
        data prep
        model forward
        scores computation
        loss computation (entity and relation? prediction)
        mrr and hits computations

        :param timesteps_train: list w. int. len: num training timesteps. list with all timesteps from train set
                                e.g. [0,24,48,..]
        :param train_data:  dict. with keys: timesteps (as in timesteps_train). values: np.array w. all triples for this
                            timestep. [sub, rel, ob, modified_indexer]
        :param timesteps_valid: list w. int. len: num valid timesteps. list with all timesteps from valid set
                                e.g. [0,24,48,..]
        :param valid_data: dict. with keys: timesteps (as in timesteps_valid). values: np.array w. all triples for this
                            timestep. [sub, rel, ob, modified_indexer]

        :returns losses: [list], num_validation_timesteps entries: the loss for each timestep computed with self.loss()
        :returns ranks_timeaware_valid: [tuple], timeaware results across all validation timesteps/triples: [0]: MR,
        [1]: MRR, [2]: hits (list with [Hits@1, Hits@3, Hits@10]), [3]: MRR per timesteps, (list with
        num_validation_timesteps entries)
        """
        losses = []
        valid_scores_tensor = []
        timesteps_trainvalid = timesteps_train + timesteps_valid
        trainvalid_data = train_data.copy()
        trainvalid_data.update(valid_data)

        nantimesteps = []

        for t_index in range(len(timesteps_train), len(timesteps_trainvalid)):  # all valid indeces
            timestep = timesteps_trainvalid[t_index]
            
            tmp = self.data_prep_t(timestep, timesteps_trainvalid, trainvalid_data)
            out_timestep, in_timesteps, out_graph, most_recent_in_graph_all = tmp
            rels_per_node = timestep_node_rel_dict_valid[in_timesteps[-1]]
            if MASK:
                mask = (out_graph[:, 1] == 4)  | (out_graph[:, 1] == 8) | (out_graph[:, 1] == 4+self._num_relations) | (out_graph[:, 1] == 8+self._num_relations) 
                out_graph = out_graph[mask]
            out_queries = out_graph[:,0:2]  # without ground truth
            out_queries = out_graph[:,0:2]  # without ground truth
            scores_tplus, H_tminus = self.model.forward(out_timestep, in_timesteps, rels_per_node, most_recent_in_graph_all, 
                                              out_queries, H_tminus) #include inverse triple

            loss = self.loss(scores_tplus, out_graph[:, 2].type(torch.long).to(self.device))  
            if torch.isnan(loss) == False:
                losses.append(loss.item())

            else:
                nantimesteps.append(t_index)

            scores = scores_tplus.detach().clone()         
            valid_scores_tensor.append(scores.cpu().detach())   


        ranks_timeaware_valid, ranks_raw_valid = utils.compute_timefilter_hits(valid_scores_tensor, 
                                                                               timesteps_valid,
                                                                                 valid_data, self._num_nodes,
                                                                                 self._num_relations, MASK)

        return losses, ranks_timeaware_valid, H_tminus

    def test(self, train_data, valid_data, test_data, timestep_node_rel_dict_test, H_tminus, result_file_name):
        """
        test the embedding model on triples from test_data
        :param train_data: [dict]: keys: timesteps, values: [array]  triples for this timestep; train
        :param valid_data: [dict]:  keys: timesteps, values: [array] triples for this timestep; valid
        :param test_data: [dict]: keys: timesteps, values: [array] triples for this timestep; test
        :param result_file_name: [string] - where should the logging_dict (with test mrrs and so on) be 
                                stored, e.g. 'results_logThu_25_May_2023_07_21_30'
        :returns self.logging_dict: [dict]: dict with result type (keys) and results (values), e.g.,'best_test_mrr'
        """
        losses = []
        timesteps_train = list(train_data.keys())
        timesteps_valid = list(valid_data.keys())
        timesteps_test = list(test_data.keys())

        timesteps_trainvalid = timesteps_train + timesteps_valid
        timesteps_trainvalidtest = timesteps_train + timesteps_valid + timesteps_test
        trainvalidtest_data = train_data
        trainvalidtest_data.update(valid_data)
        trainvalidtest_data.update(test_data)


        scores_dict = {}
        test_scores_tensor = []
        for t_index in range(len(timesteps_trainvalid), len(timesteps_trainvalidtest)): 
            timestep = timesteps_trainvalidtest[t_index]

            self.model.eval()
            # 0) data prep for this timestep (input and output timesteps, input graphs for all input timesteps,
            # output graph with positive and neg triples)
            tmp = self.data_prep_t(timestep, timesteps_trainvalidtest, trainvalidtest_data)
            out_timestep, in_timesteps, out_graph, most_recent_in_graph_all = tmp

            # 1)-3) Model Forward (including embedding and decoder)

            rels_per_node = timestep_node_rel_dict_test[in_timesteps[-1]]
            if MASK:
                mask = (out_graph[:, 1] == 4)  | (out_graph[:, 1] == 8) | (out_graph[:, 1] == 4+self._num_relations) | (out_graph[:, 1] == 8+self._num_relations) 
                out_graph = out_graph[mask]
            out_queries = out_graph[:,0:2]  # without ground truth
            out_queries = out_graph[:,0:2]  # without ground truth
            scores_tplus, H_tminus = self.model.forward(out_timestep, in_timesteps, rels_per_node, most_recent_in_graph_all, 
                                              out_queries, H_tminus) #include inverse triple

            # 4) compute loss for t+1   
            loss = self.loss(scores_tplus, out_graph[:, 2].type(torch.long).to(self.device))  
            losses.append(loss.item())

            scores = scores_tplus.detach().clone()         
            test_scores_tensor.append(scores.cpu().detach())   

            if self.dump_results_pkl:
                scores_dict = self.predicted_to_dict(out_timestep, scores, out_graph, scores_dict)

        # 5) compute the mrr and hits for all timesteps in test set
        ranks_timeaware, ranks_raw = utils.compute_timefilter_hits(test_scores_tensor, timesteps_test, test_data,
                                                                     self._num_nodes,
                                                                     self._num_relations, MASK)
        self.logging_dict['test_mrr'] = ranks_timeaware[1]
        self.logging_dict['test_hits'] = ranks_timeaware[2]

        # 6) dump json file with results
        with open(result_file_name + '.json', 'w', encoding='utf-8') as file_p:
            json.dump(self.logging_dict, file_p)

        # dump the scores_dict to pkl file:
        if self.dump_results_pkl:  # do not dump it by default as this is a large file
            # print('dump results pkl')
            result_folder_path = os.path.join(pathlib.Path().resolve(), 'results')
            
            if not os.path.exists(result_folder_path):
                os.makedirs(result_folder_path)
            model_file_path = os.path.join(result_folder_path, self.name + '_4_8_23_27' '.pkl')
            print(f'dump results pkl to {model_file_path}')
            with open(model_file_path, 'wb') as model_file_obj:
                pickle.dump(scores_dict, model_file_obj, protocol=4)
                model_file_obj.close()


        return self.logging_dict



    def predicted_to_dict(self, out_timestep, scores_predicted, out_graph, scores_dict):
        """
        select predicted scores for future timestep for given triples
        and store in dict with key: query, value: predicted scores for this query (object prediction, incl inverse triples)
        """
        index = 0
        for triple in out_graph:
            scores = scores_predicted[index]
            trip = triple.detach().cpu().numpy()
            query = str(trip[0]) + '_' + str(trip[1]) + '_xxx' + str(trip[2]) + '_' + str(out_timestep)
            gt = [trip[0], trip[1], trip[2], out_timestep]
            scores_dict[query] = [scores.cpu().detach().numpy(), gt]
            index = index +1


        return scores_dict

    def data_prep_t(self, t, timesteps_available, data_dict):
        """ prepare the data that we need for each timestep

        return out_timestep: [int] the timestep to predict for
        return in_timesteps: [list][int]: a list of all timesteps that can be used as input data (all < out_timestep)
        return out_graph_all: [torch.tensor](?) all triples in the output snapshot, including inverse triples         
        most_recent_in_graph_all: [torch.tensor](?) all triples in the last input snapshot, including inverse triples   
        """

        out_timestep = t

        in_timesteps = sorted(utils.get_smaller_ints(timesteps_available, out_timestep))

        out_graph = data_dict[out_timestep][:, 0:3] 

        # inverse triples
        inverse_out_graph = out_graph[:, [2, 1, 0]]
        inverse_out_graph[:, 1] = inverse_out_graph[:, 1] + self._num_relations
        out_graph_all = np.concatenate((out_graph, inverse_out_graph))
        out_graph_all =  torch.tensor(np.array(out_graph_all), dtype=torch.long, device=self.device)

        most_recent_in_graph = data_dict[in_timesteps[-1]][:, 0:3]
        # inverse triples
        inverse_most_recent_in_graph = most_recent_in_graph[:, [2, 1, 0]]
        inverse_most_recent_in_graph[:, 1] = inverse_most_recent_in_graph[:, 1] + self._num_relations
        most_recent_in_graph_all = np.concatenate((most_recent_in_graph, inverse_most_recent_in_graph))
        most_recent_in_graph_all =  torch.tensor(np.array(most_recent_in_graph_all), dtype=torch.long, device=self.device)

        return out_timestep, in_timesteps, out_graph_all, most_recent_in_graph_all


    def check_best_epoch(self, result_file_name, epoch, train_loss, train_losses, valid_loss, valid_losses_mean, 
                            valid_hits, best_epoch, best_valid_mrr, best_valid_hits, H_t, best_Htvalid):
        """ check if we have a new best model, if yes: store the new best model and update results logging dict
        also check if the early stopping condition is fullfilled, if yes. return early_stopping_flag: True
        """
        early_stopping_flag = False
        # load and check if better model
        if os.path.exists(self.model_file_pth):
            best_model = torch.load(self.model_file_pth, map_location=torch.device(self.device))
            if valid_hits[1] > best_model['val_mrr']:
                print('** Updating relation prediction model...')
                torch.save({'state_dict': self.model.state_dict(),
                            # 'gnn_state_dict': [gnn.state_dict() for gnn in self.model.gnns if gnn != None],
                            'train_loss': train_loss,
                            'mean_train_loss': np.mean(train_losses),
                            'mean_val_loss': np.mean(valid_losses_mean),
                            'val_loss': valid_loss,
                            'val_mrr': valid_hits[1],  # mrr
                            'best_epoch': epoch,
                            # 'h_all_all': self.model.h_all_all,
                            # 'seq_len': self.sequence_length, 
                            'result_file_name': result_file_name}, self.model_file_pth)

                best_epoch = epoch
                best_valid_mrr = valid_hits[1]
                best_valid_hits = valid_hits[2]
                best_Htvalid = H_t
                # model = best_model.copy()
            else:
                if (epoch - best_epoch) > self.early_stopping:
                    print('stopping because we do not improve the valid mmr since 50 epochs; at epoch: ', epoch)
                    with open(result_file_name + '.json', 'w', encoding='utf-8') as file_p: #dump and break
                        json.dump(self.logging_dict, file_p)
                    early_stopping_flag = True
                print(f'Best epoch: {best_epoch}')
                best_Htvalid = H_t
                

        else:  # first model
            torch.save({'state_dict': self.model.state_dict(),
                        # 'gnn_state_dict': [gnn.state_dict() for gnn in self.model.gnns if gnn != None],
                        'train_loss': train_loss,
                        'mean_train_loss': np.mean(train_losses),
                        'mean_val_loss': np.mean(valid_losses_mean),
                        'val_loss': valid_loss,
                        'val_mrr': valid_hits[1],
                        'best_epoch': epoch,
                        # 'h_all_all': self.model.h_all_all,
                        # 'seq_len': self.sequence_length,
                        'result_file_name': result_file_name}, self.model_file_pth)
            best_epoch = epoch
            best_valid_mrr = valid_hits[1]
            best_valid_hits = valid_hits[2]
            print('** Saved relation prediction model for the first time **')
        self.logging_dict['mean_valid_loss'] = np.mean(valid_losses_mean)
        self.logging_dict['best_epoch'] = best_epoch
        self.logging_dict['best_valid_mrr'] = best_valid_mrr
        self.logging_dict['best_valid_hits'] = best_valid_hits      

        return best_epoch, best_valid_mrr, best_valid_hits, early_stopping_flag, best_Htvalid
    

    def load_model_from_file(self, device, model_name):
        """
        load the pretrained model from a file, that is located in self.model_file_pth, to device, in self.mode;
        :param device: [str]: device (e.g. device(type='cuda', index=2))
        :param model_name: [str] name of the model to be loaded
        :returns logging_dict: logging dict with results from the loaded model, e.g. results for valid_mrr
        """
        self.name = model_name  # which model should be loaded
        self.model_file_pth = os.path.join(self.model_file_dir, self.name)

        model_dict = torch.load(self.model_file_pth, map_location=torch.device(device))
        self.model.load_state_dict(model_dict['state_dict'])

        self.model = self.model.to(device)

        self.logging_dict['mean_train_loss'] = model_dict['mean_train_loss']
        self.logging_dict['mean_valid_loss'] = model_dict['mean_val_loss']
        self.logging_dict['best_epoch'] = model_dict['best_epoch']
        self.logging_dict['best_valid_mrr'] = model_dict['val_mrr']
        result_file_name = model_dict['result_file_name']
        # self.logging_dict['best_valid_hits'] = model_dict'best_valid_hits']

        return self.logging_dict, result_file_name
