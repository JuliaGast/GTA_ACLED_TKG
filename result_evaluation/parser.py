"""/*
 *    Dynamic Representations of Global Crises: A Temporal Knowledge Graph For Conflicts, Trade and Value Networks
 *
 *        File: parser.py
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
