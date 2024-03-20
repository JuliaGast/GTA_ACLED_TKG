# """/*
#  *    Dynamic Representations of Global Crises: A Temporal Knowledge Graph For Conflicts, Trade and Value Networks
#  *
#  *        File: data_handler.py
#  *
#  *     Authors: Deleted for purposes of anonymity 
#  *
#  *     Proprietor: Deleted for purposes of anonymity --- PROPRIETARY INFORMATION
#  * 
#  * The software and its source code contain valuable trade secrets and shall be maintained in
#  * confidence and treated as confidential information. The software may only be used for 
#  * evaluation and/or testing purposes, unless otherwise explicitly stated in the terms of a
#  * license agreement or nondisclosure agreement with the proprietor of the software. 
#  * Any unauthorized publication, transfer to third parties, or duplication of the object or
#  * source code---either totally or in part---is strictly prohibited.
#  *
#  *     Copyright (c) 2021 Proprietor: Deleted for purposes of anonymity
#  *     All Rights Reserved.
#  *
#  * THE PROPRIETOR DISCLAIMS ALL WARRANTIES, EITHER EXPRESS OR 
#  * IMPLIED, INCLUDING BUT NOT LIMITED TO IMPLIED WARRANTIES OF MERCHANTABILITY 
#  * AND FITNESS FOR A PARTICULAR PURPOSE AND THE WARRANTY AGAINST LATENT 
#  * DEFECTS, WITH RESPECT TO THE PROGRAM AND ANY ACCOMPANYING DOCUMENTATION. 
#  * 
#  * NO LIABILITY FOR CONSEQUENTIAL DAMAGES:
#  * IN NO EVENT SHALL THE PROPRIETOR OR ANY OF ITS SUBSIDIARIES BE 
#  * LIABLE FOR ANY DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES
#  * FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF INFORMATION, OR
#  * OTHER PECUNIARY LOSS AND INDIRECT, CONSEQUENTIAL, INCIDENTAL,
#  * ECONOMIC OR PUNITIVE DAMAGES) ARISING OUT OF THE USE OF OR INABILITY
#  * TO USE THIS PROGRAM, EVEN IF the proprietor HAS BEEN ADVISED OF
#  * THE POSSIBILITY OF SUCH DAMAGES.
#  * 
#  * For purposes of anonymity, the identity of the proprietor is not given herewith. 
#  * The identity of the proprietor will be given once the review of the 
#  * conference submission is completed. 
#  *
#  * THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#  */"""

'''
 TKG-ACLED-GTA-Dataset


'''
import numpy as np
import os
import pathlib

from itertools import groupby
from operator import itemgetter


def load(dataset_name: str) -> tuple: 
    """ load txt files with the graph quadruples """
    if 'merged' in dataset_name:
        name = 'acled_subset_merged_gta_ids_graph.txt'
        if 'aggregated' in dataset_name:
            name = 'acled_subset_merged_gta_aggregated_ids_graph.txt'
    else:
        name = 'acled_subset_gta_ids_graph.txt'
        if 'aggregated' in dataset_name:
            name = 'acled_subset_gta_aggregated_ids_graph.txt'
    if '2023' in dataset_name:
            name = dataset_name
    

    
    root = os.path.join(pathlib.Path().resolve(), 'data')

    data= _load_file(root, name)
   
    return data

def group_by(data: np.array, key_idx: int) -> dict:
    data_grouped = groupby(data, key=itemgetter(key_idx))
    data_dict = {key: np.delete(np.array(list(values)), key_idx, 1) for key, values in data_grouped}

    return data_dict

def _load_file( path: str, filename: str) -> np.array:
    data = np.loadtxt(os.path.join(path, filename), dtype=int) 
    print(data.shape)

    return data
