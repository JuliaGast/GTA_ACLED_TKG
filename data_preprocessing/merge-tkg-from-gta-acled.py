"""/*
 *    Dynamic Representations of Global Crises: A Temporal Knowledge Graph For Conflicts, Trade and Value Networks
 *
 *        File: merge-tkg-from-gta-acled.py
 *
Authors: Julia Gastinger (julia.gastinger (at) uni-mannheim.de), Timo Sztyler (timo.sztyler (at) neclab.eu), 
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



import pandas as pd
import numpy as np
from utils import df_to_rdfgraph, map_ids_to_string, map_ids_to_time, create_dicts_from_df, store_ids, store_graph
from copy import copy
import os
from tqdm import tqdm

# 0) read graph_gta.csv  and graph_acled.csv & create dicts

acled_name= 'acled_all'
gta_name = 'gta_2023' #gta_2023


datasetid_gta = './data/gta/gta_2023.csv'

if acled_name == 'acled_event': #only events
    datasetid_acled = './data/acled/acled_2023_event_types.csv'    
    folder_name = 'crisis_merged'
elif acled_name == 'acled_all':
    datasetid_acled = './data/acled/acled_2023_all.csv'    
    folder_name = 'crisis_merged'



task_name = 'crisis2023'
data_loc = './data'
added_dict = create_dicts_from_df(datasetid_gta, datasetid_acled) 
# added_dict: key: date. all triples that have announcement date (gta) or general date (acled) at that timestep. 

# 1) create a list  with all possible timesteps, and timesteps of interest
start_range="2023-01-01"
end_range="2023-12-31"
folder_name += '2023' #start_range
# folder_name +=end_range
if not os.path.exists(data_loc + '/' + folder_name):
    os.mkdir(data_loc + '/' + folder_name)
print('make sure the out-files do not exist yet, results will be appended!, foldername: ', data_loc + '/' + folder_name)
timesteps_range = pd.date_range(start_range,end_range) # only timesteps added after starte_range and removed before end_range
timesteps_range_all = pd.date_range(start="1900-01-01",end="2080-12-31")  

# 1a) train/valid/test percentage
train_per = 80 
valid_per = 10
num_timesteps = len(timesteps_range)
timesteps_range_train_id = int(train_per/100*num_timesteps)
timesteps_range_valid_id = int((valid_per+train_per)/100*num_timesteps)
timesteps_range_test_id= int((100)/100*num_timesteps)-1



# 2) string to id mapping (for nodes and relations); date to id mapping
node_to_id, rel_to_id = map_ids_to_string(added_dict, timesteps_range, folder_name)
time_to_id, id_to_time, time_to_id_ourinterval, id_to_time_ourinterval = map_ids_to_time(timesteps_range_all, timesteps_range)

# 3) for each possible timesteps: create graph snapshot with all events that happen in that timestep  & write quadruples to txt file-> 
# add all triples that are added at this timestep and all remove triples that are removed at it
timestamps = []
graph_timeseries ={}
triples_dict = {}
current_graph = pd.DataFrame(columns =[0, 1, 2, 'ts', 3 ])
tminus = None

for t_all in tqdm(timesteps_range): 
    train_val_test = 'test'
    if t_all < timesteps_range[timesteps_range_valid_id]:
        train_val_test= 'valid'
    if t_all < timesteps_range[timesteps_range_train_id]:
        train_val_test= 'train'
    
    t = str(t_all.date())

    if t in added_dict.keys(): # add all the triples that were added in this ts
        triples_added_t_df = pd.DataFrame(added_dict[t])
        current_graph = triples_added_t_df.drop_duplicates(keep='first') 
        

    triples_dict[str(t_all.date())] = current_graph
    if t_all >= timesteps_range[0]:
        graph = df_to_rdfgraph(current_graph)       

        store_graph(current_graph,  time_to_id_ourinterval, t,  data_type=train_val_test, location=data_loc,task_name=folder_name)
        store_ids(current_graph, node_to_id, rel_to_id, time_to_id_ourinterval, t, data_type=train_val_test, location=data_loc, task_name=folder_name)


    tminus = copy(t)

print("done")



