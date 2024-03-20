"""*
 *     Static Baseline
 *
 *        File: baseline_model.py
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
from torch.nn import Sequential as Seq, Linear, ReLU

import math
import numpy as np

import torch
from torch.nn.modules.module import Module
from torch.nn import Parameter
from torch.nn import functional as F



class Simple_Model(torch.nn.Module):
    def __init__(self, num_nodes, num_relations, embedding_dim, device, model_config) -> None:
        super(Simple_Model, self).__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.num_relations_incl_inverse = 2*num_relations

        self.device = device
        self.embedding_dim = embedding_dim
        self.real_embedding_dim = self.embedding_dim // 2  # RotatE interaction assumes vectors conists of two parts: real and imaginary

        self.decoder_type = model_config.decoder # 'rotate'
        if self.decoder_type =='distmult':
            self.decoder = DistMult()
        elif self.decoder_type =='convtranse':
            self.decoder = ConvTransE(self.num_nodes, self.embedding_dim, self.device)


        self.relation_embeddings = torch.nn.Embedding(self.num_relations_incl_inverse+1, embedding_dim=self.embedding_dim, 
                                                device=self.device) 
        self.node_embeddings = torch.nn.Embedding(self.num_nodes+1, embedding_dim=self.embedding_dim, 
                                                device=self.device) 





    def forward(self, t, timesteps_train, rels_per_node, most_recent_in_graph_all, out_queries, H_tminus):
        # 1) Encoder:
        # for each node: use the relations at the last known timestep

        if self.decoder_type == 'rotate':
            
            s = self.node_embeddings(out_queries[:,0]).view(-1, 1, self.real_embedding_dim, 2) #encode heads
            r = self.relation_embeddings(out_queries[:, 1]).view(-1, 1, self.real_embedding_dim, 2) #encode relations
            o = self.node_embeddings( torch.arange(self.num_nodes, device=self.device) ).view(1, -1, self.real_embedding_dim, 2)
            H_t = H_tminus
        else:
            s = self.node_embeddings(out_queries[:,0]) #encode heads
            r = self.relation_embeddings(out_queries[:, 1]) #.view(-1, 1, self.real_embedding_dim, 2) #encode relations
            o = self.node_embeddings( torch.arange(self.num_nodes, device=self.device) )# self.node_embeddings  #.view(1, -1, self.real_embedding_dim, 2)                
            H_t = H_tminus

        # 2) Decoder
        # Compute scores
        scores_tplus = self.decode(head=s, rel=r, tail=o)

        return scores_tplus, H_t


    def decode(self, head: torch.FloatTensor, rel: torch.FloatTensor, tail: torch.FloatTensor)-> torch.FloatTensor:
        if self.decoder_type == 'rotate':
            return self.rotate_decode(head, rel, tail) 

        elif self.decoder_type =='rotate_notworking':
            # return self.decoder.forward(head, rel, tail)  
            head_re = head[0]
            head_im = head[1]
            tail_re = tail[0]
            tail_im = tail[1]
            

            return self.decoder.forward(head_re, head_im, tail_re, tail_im, rel)
        elif self.decoder_type =='convtranse':
            return self.decoder.forward(head, rel, tail)    
        elif self.decoder_type =='distmult':
            return self.decoder.forward(head, rel, tail)  
        else:
            print(f"{self.decoder_type} not implemented. using rotate_decode instead")
            return self.rotate_decode(head, rel, tail) 


    ### c) decoder 
    @staticmethod 
    def rotate_decode( #original: def interaction_function()
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        #TODO: I think here we could also use the rotate implementation from pytorch geometric 
        # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/kge/rotate.html#RotatE.forward

        # Decompose into real and imaginary part
        h_re = h[..., 0] # returns the first element along the last axis of the array h. #The ellipsis (...) is a slicing 
                         # notation that represents a full slice along all dimensions of the array.
        h_im = h[..., 1]
        r_re = r[..., 0]
        r_im = r[..., 1]

        # Rotate (=Hadamard product in complex space).
        rot_h = torch.stack(
            [
                h_re * r_re - h_im * r_im,
                h_re * r_im + h_im * r_re,
            ],
            dim=-1,
        )
        # Workaround until https://github.com/pytorch/pytorch/issues/30704 is fixed
        diff = rot_h - t
        # scores = -torch.norm(diff.view(diff.shape[:-2] + (-1,)), dim=-1)

        scores =  - torch.linalg.vector_norm(diff.view(diff.shape[:-2] + (-1,)), dim=(-1))

        del diff, rot_h, h_re, h_im, r_re, r_im
        torch.cuda.empty_cache()
        return scores
    




class DistMult(Module):
    """ DistMult scoring function (from https://arxiv.org/pdf/1412.6575.pdf); copied from:
    https://github.com/thiviyanT/torch-rgcn/blob/267faffd09a441d902c483a8c130410c72910e90/torch_rgcn/layers.py#L20
    """
    def __init__(self):
        super(DistMult, self).__init__()


        # self.device = device
        # self.num_nodes = num_nodes
        # self.num_rel = num_rel
    def forward(self, sub, rel, ob):
        """ Score candidate triples 
        :param node_embeddings: tensor. shape [num_nodes, embd_dim]. embedding for each node as computed by encoder
        :param relation_embeddings: tensor. shape [2*num_rel, embd_dim]. relation embeddings as computed by encoder. 
            separate embedding for in or outgoing relation. only used if self.separate_emb=False
        :param triples: tenspr. shape [num_triples, 3]. contains inverse triples. triples to compute the scores for (we 
            compute scores for the nodes being the objects in the triples)
        :param all_triples_flag: boolean. if True: compute score for all nodes to be object for the triples. if 
                False: compute score only for node of interest. (this is used if we already provide all possible combis 
                in triples)
        :returns scores: tensor, shape [num_triples, num_nodes]. score for each node to be the object in each triple.
        """
        scores = torch.mm(sub * rel, ob.transpose(1, 0)) 
        # scores = (sub * rel * ob).sum(dim=-1)
        return scores






class ConvTransE(torch.nn.Module):
    """ copied from:
    https://github.com/Lee-zix/RE-GCN/blob/master/src/decoder.py
    ConvTransE scoring function Chao Shang, Yun Tang, Jing Huang, Jinbo Bi, Xiaodong He, and Bowen Zhou.
        2019. End-to-end structure-aware convolutional networks for knowledge base
        completion. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 33.
        3060–3067
    """
    def __init__(self, num_entities, embedding_dim, device='cpu', input_dropout=0, hidden_dropout=0, 
        feature_map_dropout=0, channels=50, kernel_size=3, use_bias=True):

        super(ConvTransE, self).__init__()
        # 初始化relation embeddings
        # self.emb_rel = torch.nn.Embedding(num_relations, embedding_dim, padding_idx=0)
        self.device = device
        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv1d(2, channels, kernel_size, stride=1,
                               padding=int(math.floor(kernel_size / 2))).to(self.device)  # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(2).to(self.device)
        self.bn1 = torch.nn.BatchNorm1d(channels).to(self.device)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim).to(self.device)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim).to(self.device)
        self.bn3 = torch.nn.BatchNorm1d(embedding_dim).to(self.device)
        # self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(embedding_dim).to(self.device)

  #  (h_t_k, output_graph, all_triples_flag)
    def forward(self, sub, rel, ob, all_triples_flag=True, train_flag=True): 
        sub_emb = F.tanh(sub).unsqueeze(1)
        rel_emb = rel.unsqueeze(1) #.unsqueeze(1)
        ob_emb = ob

        # e1_embedded_all = F.tanh(embedding)
        batch_size = len(ob)
        # e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        # rel_embedded = emb_rel(triplets[:, 1]).unsqueeze(1)
        stacked_inputs = torch.cat([sub_emb, rel_emb], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        if train_flag: #dropout only during training
            x = self.inp_drop(stacked_inputs)
        else:
            x = stacked_inputs
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        if train_flag:
            x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        if train_flag:
            x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        if all_triples_flag:
            x = torch.mm(x, ob_emb.transpose(1, 0))
        else:
            ob_embedded = ob_emb
            x = (x*ob_embedded).sum(dim=-1)
        return x
