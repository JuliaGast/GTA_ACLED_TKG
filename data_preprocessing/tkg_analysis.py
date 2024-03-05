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

