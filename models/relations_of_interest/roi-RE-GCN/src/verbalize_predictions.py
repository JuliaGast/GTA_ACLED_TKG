import os
import pickle
import numpy as np

def enttotext():
    file_path = "../data/crisis2022/relation2id.txt"

    # Initialize an empty dictionary
    rel2txt = {}

    # Read the text file and populate the dictionary
    with open(file_path, "r") as txt_file:
        for line in txt_file:
            # Split each line into key and value using a colon as the delimiter
            value, key = line.strip().split("\t")
            # Remove leading and trailing whitespace from the key and value
            # key = key.strip()
            key = int(key)
            # Add the key-value pair to the dictionary
            rel2txt[key] = value


    # Now, data_dict contains the data from the text file as a dictionary
    # print(data_dict)


    file_path = "../data/crisis2022/entity2id.txt"

    # Initialize an empty dictionary
    ent2txt = {}

    # Read the text file and populate the dictionary
    with open(file_path, "r") as txt_file:
        for line in txt_file:
            # Split each line into key and value using a colon as the delimiter
            value, key = line.strip().split("\t")
            # Remove leading and trailing whitespace from the key and value
            # key = key.strip()
            key = int(key)
            # Add the key-value pair to the dictionary
            ent2txt[key] = value


    # Now, data_dict contains the data from the text file as a dictionary
    # print(data_dict)
    return ent2txt, rel2txt

directory = './results'

pickle_file_1 = 'regcn-crisis2022-0-singlestep-raw.pkl' #ICEWS14_temporal_nodepiece_Thu_31_Aug_2023_07_29_38.pkl'
# pickle_file_2 = 'ICEWS14_temporal_nodepiece_Thu_31_Aug_2023_07_30_04.pkl'

print(f'Loading: {pickle_file_1}')
with open(os.path.join(directory, pickle_file_1), 'rb') as file:
    pickle_file1 = pickle.load(file)


# print(f'Loading: {pickle_file_2}')
# with open(os.path.join(directory, pickle_file_2), 'rb') as file:
#     pickle_file2 = pickle.load(file)

ent2txt, rel2txt = enttotext()

scores_combined = {}
predictions = []
for p1_key, p1_value in pickle_file1.items(): # pickle_file2):
    scores_p1 = p1_value[0]
    predicted_node = np.argmax(scores_p1)
    predicted_node_txt = ent2txt[predicted_node]

    triple_txt = []
    # for val in p1_value[1][0:3]:
    val = p1_value[1]
    triple_txt.append(ent2txt[val[0]])
    if val[1] > 7:
        val[1] = val[1]-7
        triple_txt.append(rel2txt[val[1]]+'_inverse')
    else:
        triple_txt.append(rel2txt[val[1]])
    
    triple_txt.append(ent2txt[val[2]])

    triple_txt.append(predicted_node_txt)
    triple_txt.append(val[3])
    predictions.append(triple_txt)

    # triple_txt = [ent2txt[(val for val in p1_value[1])]]
    # scores_p2 = pickle_file2[p1_key][0]
    # scores_sum = scores_p1 + scores_p2
    # scores_combined[p1_key]  = []
    # scores_combined[p1_key].append(scores_sum)
    # scores_combined[p1_key].append(p1_value[1])

# Specify the path to your text file
file_path = "./results/predictions3.csv"

with open(file_path , mode='wt', encoding='utf-8') as myfile:

    myfile.write('sub;rel;ob_gt;prediction;timestep;')
    myfile.write('\n')
    for line in predictions:
        for entry in line:
            myfile.write(str(entry)+"; ")  
        myfile.write('\n')
    # myfile.write('\n'.join(predictions))

# with open(file_path, "r") as txt_file:
#     for line in predictions:
#         # Split each line into key and value using a colon as the delimiter
#         value, key = line.strip().split("\t")

# to_pkl = scores_combined
                # timesteps, test_triples, final_scores, all_ans_list_test, all_ans_static = \
                #     setup(dataset_name, pickle_file)
                
# with open('./results/nodepiece/mean_combineThu_31_Aug_2023_07_29_38Thu_31_Aug_2023_07_30_04.pkl', 'wb') as handle:
#     pickle.dump(scores_combined, handle, protocol=pickle.HIGHEST_PROTOCOL)




