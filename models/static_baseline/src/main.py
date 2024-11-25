"""*
 *     Static Baseline
 *
 *        File: main.py
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
