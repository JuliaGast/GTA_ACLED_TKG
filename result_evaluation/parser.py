"""/*
 *    Dynamic Representations of Global Crises: A Temporal Knowledge Graph For Conflicts, Trade and Value Networks
 *
 *        File: parser.py
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
 */"""



"""
 write results from json file to excel table for better analysis.
 peviously you have to run run_evaluation.py to create output.json with results per method per dataset per config
"""
import os
import json
import pandas as pd
import numpy as np


# noinspection PyShadowingNames
def helper_function(filename: str) -> tuple:
    filename = filename.replace('.pkl', '')
    if "feedvalid" in filename:
        filename = filename.replace('feedvalid', '')
    
    filename = filename.split('_')  # Idea is to use these keywords to decide the location of values in dataframe
    _setting, _filtering, _dataset, _method, _lmbda = 'NA', 'NA', 'NA', 'NA', 'NA'
    for item in filename:
        window = 0 #int(filename[4])
        scaling_factor = 0 #float(filename[2])
        if window < 0:
            _setting = 'multistep'
        else:
            _setting = 'singlestep'
        # _setting = item if item in setting else _setting
        _filtering = item if item in filtering else _filtering
        _dataset = item if item in datasets else _dataset
        # Invert the method name from small letters to a consistent format
        _method = inv_dict[item] if item in method_names.values() else _method
        # if item == 'True':
        #     return None, None, None, None
    _method= filename
    _lmbda = 0 # filename[5]
    _alpha = 0 #filename[6]
    _rulefile = 0# filename[2]
    return _setting, _filtering, _dataset, _method, _alpha, _lmbda, _rulefile, scaling_factor


if __name__ == '__main__':

    ROOT = os.path.join(os.getcwd())
    with open(os.path.join(ROOT, 'output_final.json'), 'r') as stream:
        jsonfile = json.load(stream)
    # normalize_sub_dicts(jsonfile)

    setting = ['singlestep' ] #, 'singlesteponline']
    filtering = ['time', 'raw', 'static']
    metrics = ['mrr', 'hits@1', 'hits@3', 'hits@10']
    datasets = [ 'crisis2023']
    method_names = {        
        'TLogic': 'TLogic',
        'rule_baseline': 'rule_baseline',
        'RE-GCN': 'RE-GCN',
        'Timetraveler': 'Timetraveler',
        'static_baseline': 'static_baseline',
        'combi': 'combi'

    }
    inv_dict = {value: key for key, value in method_names.items()}  # used in helper function
    assert len(method_names.keys()) == len(jsonfile.keys()), 'Reports for all methods not present in jsonfile!'

    # Initialise variables relating to the dataframe
    column_names = [f'{dataset}_{metric}' for dataset in datasets for metric in metrics]
    raw_df = pd.DataFrame(columns=column_names)
    static_df = pd.DataFrame(columns=column_names)
    time_df = pd.DataFrame(columns=column_names)

    for method_name in method_names.keys():
        sub_dict = jsonfile[method_name]

        # Iterate on each sub-dict (.pkl report values)
        for pkl_name, report in sub_dict.items():
            print(pkl_name, '\n', '=' * 100)

            _setting, _, _dataset, _method, alpha, lmbda, rulefile, scaling_factor = helper_function(pkl_name)
            # if int(rulefile) == 2: #with the old tlogic rules
            #     continue
            if 'NA' in _method:
                print(pkl_name)
            # if _setting is None:  # special constraint check that avoids pkl files with `True` in their names
            #     continue

            if _setting == 'multistep':
                continue

            for filter, values in report.items():
                if 'mrr_per_rel' not in filter:
                    index = f'{_method}_{_setting}_{lmbda}_{alpha}_{scaling_factor}'
                    mrr = np.round(values[1] * 100, 2)
                    hits = [np.round(value * 100, 2) for value in values[2]]

                    if filter == 'raw':
                        raw_df.loc[index, f'{_dataset}_mrr'] = mrr
                        raw_df.loc[index, f'{_dataset}_hits@1'] = hits[0]
                        raw_df.loc[index, f'{_dataset}_hits@3'] = hits[1]
                        raw_df.loc[index, f'{_dataset}_hits@10'] = hits[2]
                    elif filter == 'static':
                        static_df.loc[index, f'{_dataset}_mrr'] = mrr
                        static_df.loc[index, f'{_dataset}_hits@1'] = hits[0]
                        static_df.loc[index, f'{_dataset}_hits@3'] = hits[1]
                        static_df.loc[index, f'{_dataset}_hits@10'] = hits[2]
                    elif filter == 'time':
                        time_df.loc[index, f'{_dataset}_mrr'] = mrr
                        time_df.loc[index, f'{_dataset}_hits@1'] = hits[0]
                        time_df.loc[index, f'{_dataset}_hits@3'] = hits[1]
                        time_df.loc[index, f'{_dataset}_hits@10'] = hits[2]
                    else:
                        raise Exception

    # Save the output as a .xlsx document
    writer = pd.ExcelWriter(os.path.join(ROOT, 'output_final_singlestep_2024.xlsx'), engine='xlsxwriter')
    raw_df.to_excel(writer, sheet_name='raw')
    static_df.to_excel(writer, sheet_name='static')
    time_df.to_excel(writer, sheet_name='time')
    writer._save()
