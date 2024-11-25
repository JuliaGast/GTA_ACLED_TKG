"""/*
#  *    Dynamic Representations of Global Crises: A Temporal Knowledge Graph For Conflicts, Trade and Value Networks
#  *
#  *        File: tkg_analysis.py
#  *
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




import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from collections import Counter


def t2s(s,r,o):
    return str(s) + " " + str(r) + " " + str(o)

class TripleSet:

    def __init__(self):
        self.sub_rel_2_obj = {}
        self.obj_rel_2_sub = {}
        self.triples = []
        
        self.timestep_lists = []
        self.counter_s = 0
        self.max_ts = 0
        self.min_ts = 1000
        self.num_timesteps = 0
        self.num_triples = 0

    def add_from_file(self, filepath):
        file = open(filepath, 'r')
        lines = file.readlines()
        count = 0
        t_min = None
        t_max = None

        counter_s = 0

        for line in lines:
            token = line.split("\t")
            s = int(token[0])
            r = int(token[1])
            o = int(token[2])
            t = int(token[3])/timdif[dataset]

            self.triples.append([s,r,o,t]) 

            self.index_triple(self.sub_rel_2_obj, s,r,o,t, counter_s)
            self.index_triple(self.obj_rel_2_sub, o,r,s,t, 0)
            
            if t_min == None:
                t_min = t
                t_max = t
            if t < t_min: t_min = t
            if t > t_max: t_max = t
            count += 1

        # print(counter_s)
        print(">>> read " + str(count) + " triples from time " + str(t_max) + " to " + str(t_min))
        if t_min < self.min_ts:
            self.min_ts = t_min
        if t_max > self.max_ts:
            self.max_ts = t_max
        

    def compute_stat(self):
        self.timestep_lists = self.create_timestep_lists(self.sub_rel_2_obj)
        self.num_timesteps = 1+ self.max_ts - self.min_ts
        self.num_triples = len(self.triples)

    def create_timestep_lists(self, x_y_2_z):
        timestep_lists = list(self.flatten_dict(x_y_2_z))
        return timestep_lists

    def flatten_dict(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                yield from self.flatten_dict(value)
            else:
                yield value

    def index_triple(self, x_y_2_z, x, y, z, t, counter):
        if x not in x_y_2_z:
            x_y_2_z[x] = {}
        if y not in x_y_2_z[x]:
            x_y_2_z[x][y] = {}
        if z not in x_y_2_z[x][y]:
            x_y_2_z[x][y][z] = []
            # counter +=1
        if t not in x_y_2_z[x][y][z]:
            x_y_2_z[x][y][z].append(t)
        # return counter

    def get_latest_ts(self, s, r, o, t):
        closest = -1
        if s in self.sub_rel_2_obj:
            if r in self.sub_rel_2_obj[s]:
                if o in self.sub_rel_2_obj[s][r]:
                    ts = self.sub_rel_2_obj[s][r][o]
                    for k in ts:
                        if k < t and k > closest:
                            closest = k
        return closest

    def count(self, e):
        return len(list(filter(lambda x : x[0] == e or x[2] == e, self.triples)))

    def show(self, num=100):
        count = 0
        len_sum = 0
        for s in self.sub_rel_2_obj:
            for r in self.sub_rel_2_obj[s]:
                for o in self.sub_rel_2_obj[s][r]:
                    ts = self.sub_rel_2_obj[s][r][o]
                    print(t2s(s,r,o) + ": " + str(len(ts)))
                    len_sum += len(ts)
                    count +=1
                    if count > num:
                        return
        # print("mean length: " + str(len_sum / count))

def max_consecutive_numbers(lst):
    max_count = 0
    current_count = 1
    
    for i in range(1, len(lst)):
        if lst[i] == lst[i-1] + 1:
            current_count += 1
        else:
            max_count = max(max_count, current_count)
            current_count = 1
    
    return max(max_count, current_count)

datasets = ['GTA_ACLED']#if you want to add more datasets you can e.g. download them here:(https://github.com/Lee-zix/RE-GCN)
# ['GDELT',  'ICEWS18', 'ICEWS14', 'GTA_ACLED', 'YAGO','WIKI'] # 'GDELT',  gdelt, ice18, ice14, gta, yago, wiki
timdif = {}
timdif['ICEWS14'] = 24
timdif['ICEWS18'] = 24
timdif['YAGO'] = 1
timdif['WIKI'] = 1
timdif['GTA_ACLED'] = 1
timdif['GDELT'] = 15
x_values_per_ds = {}
y_values_per_ds = {}
barplot_dict = {}

for dataset in datasets:

    print('DATASET: ', dataset)
    ts_all = TripleSet()
    ts_all.add_from_file("data/" + dataset + "/train.txt")
    ts_all.add_from_file("data/" + dataset + "/valid.txt")
    ts_all.add_from_file("data/" + dataset + "/test.txt")

    ts_all.compute_stat()

    ts_test = TripleSet()
    ts_test.add_from_file("data/" + dataset + "/test.txt")

    ts_test.compute_stat()

    lens = []
    for timesteps in ts_all.timestep_lists:
        lens.append(len(timesteps))
        if len(timesteps) > ts_all.num_timesteps:
            print('hi')



    count_previous = 0
    count_sometime = 0
    count_all = 0
    for qtriple in ts_test.triples:    
        (s,r,o,t) = qtriple
        # if not r == 2: continue
        # print("=> " + str(max_index) + "   " + str(max_score))
        k = ts_all.get_latest_ts(s,r,o, t)
        # print(str(s) + "   " + str(r) + "   " + str(o) + "  @" + str(t) + " found at " + str(k))
        count_all += 1
        if k + 1 == t: count_previous += 1
        if k > -1 and k < t: count_sometime += 1

    print("DATATSET:  " +  dataset)
    print("all:       " +  str(count_all))
    print("previous:  " +  str(count_previous))
    print("sometime:  " +  str(count_sometime))
    print("f-direct (DRec):   " +  str(count_previous / count_all))
    print("f-sometime (Rec): " +  str(count_sometime / count_all))


    print(f"the mean number of timesteps that a triple appears in is {np.mean(lens)}")
    print(f"the median number of timesteps that a triple appears in is {np.median(lens)}")
    print(f"the maximum number of timesteps that a triple appears in is {np.max(lens)}")


    # Compute max consecutive timesteps per triple
    results = [max_consecutive_numbers(inner_list) for inner_list in ts_all.timestep_lists]
    print(f"number of timesteps is {ts_all.num_timesteps}")
    print(f"number of total triples is {ts_all.num_triples}")
    print(f"number of distinct triples is {len(ts_all.timestep_lists)}")
    print(f"the mean max number of 100*consecutive timesteps/number of timesteps that a triple appears in is {100*np.mean(results)/ts_all.num_timesteps}")
    print(f"the median max number of 100*consecutive timesteps/number of timesteps that a triple appears in is {100*np.median(results)/ts_all.num_timesteps}")
    print(f"the maximum max number of 100*consecutive timesteps/number of timesteps that a triple appears in is {100*np.max(results)/ts_all.num_timesteps}")
    print(f"the mean max number of consecutive timesteps that a triple appears in is {np.mean(results)}")
    print(f"the median max number of consecutive timesteps that a triple appears in is {np.median(results)}")
    print(f"the maximum max number of consecutive timesteps that a triple appears in is {np.max(results)}")
    print(f"the std for max number of consecutive timesteps that a triple appears in is {np.std(results)}")


    # Compute frequency of each unique value
    value_counts = Counter(results)

    x_values = np.array(list(value_counts.keys())) #/ts_all.num_timesteps
    y_values = 100*np.array(list(value_counts.values()))/ts_all.num_triples
    x_values_per_ds[dataset] = x_values
    y_values_per_ds[dataset] = y_values
    # Create plot
    plt.figure()

    plt.grid(zorder=0)
    plt.gca().set_axisbelow(True)
    plt.scatter(x_values, y_values, s=10)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Maximum Number of Consecutive Timesteps')
    plt.ylabel('Number of Occurences (%)')

    plt.savefig(dataset+'consec.pdf')

    # Create histogram

    bins=[1, 2, 5, 10, 50, 100, 500]
    keys = value_counts.keys()
    values = 100*np.array(list(value_counts.values()))/(np.sum(list(value_counts.values()))) #ts_all.num_triples
    plt.figure()
    b2 = plt.hist(keys, bins, weights=values, edgecolor='black')
    bin_names = ['<'+str(b) for b in b2[1][1:]]
    plt.figure()

    plt.bar(bin_names, b2[0])
    plt.yscale('log')
    plt.savefig('test4.png')
    bin_names = ['<'+str(b) for b in b2[1][1:]]
    plt.figure()
    plt.bar(bin_names, b2[0])
    plt.savefig(dataset+'histogram.png')
    barplot_dict[dataset] = [bin_names, b2[0]]



    print('--------------------------------------------------------------------------')





# one fig for all datasets
plt.figure()
#marker_list = ['1', '.', 'x', '3', '2', 'X']
marker_list = ['o', 'v', '^', '<', '>', 'h']
plt.grid(zorder=0)
plt.gca().set_axisbelow(True)
i =0
for dataset in datasets:
    plt.scatter(x_values_per_ds[dataset], y_values_per_ds[dataset], s=15,alpha=0.7,label = dataset, marker = marker_list[i])
    i+=1
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Maximum Number of Consecutive Timesteps')
plt.ylabel('Number of Occurences (%)')
plt.savefig('all_consec.pdf')
plt.savefig('all_consec.png')



N = 6
# Position of bars on x-axis
ind = np.arange(N)

# Width of a bar 
width = 0.15       
plt.figure(figsize=(10,5))
# Plotting
i = 0
for dataset in datasets:
    plt.bar(ind+i*width , barplot_dict[dataset][1], width, label=dataset)
    i +=1

labels = ['[1,2)', '[2,5)', '[5,10)', '[10,50)', '[50,100)', '[100,500)']
plt.xticks(ind + 5*width/2  , labels)
# Finding the best position for legends and putting it
plt.legend(loc='best')

plt.ylabel('Number of Occurence ([%] of all triples)')
plt.xlabel('Consecutiveness Value (Bins)')

plt.savefig('histogram_all.png')
plt.savefig('histogram_all.pdf')


plt.close('all')

