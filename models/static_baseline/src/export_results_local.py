"""*
 *     Static Baseline
 *
 *        File: export_results_local.py
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



import json
import os
# import paramiko
import requests
import sys
# from tomark import Tomark
import yaml
import csv
# from scp import SCPClient
import hydra 
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path='', config_name="conf")
def main(config:DictConfig, result_file_name): #'results_log.json'):
    # Input Data
    result_file_path = './'+ result_file_name +'.json'

    dataset_name = config.dataset['name']
    # Build new table row
    new_table_row = _build_new_table_row(result_file_path, config)

    dataset_name = config.dataset['name']

    with open("all_results_"+dataset_name+".csv", 'a') as csvFile:
            wr = csv.DictWriter(csvFile, fieldnames=list(new_table_row.keys()))
            # wr.writeheader() #use when first writing
            wr.writerow(new_table_row )    

    #TODO: delete result file
    os.remove(result_file_path)


def write(config:DictConfig, result_file_name, dataset=None): #'results_log.json'):
    # Input Data
    result_file_path = './'+ result_file_name +'.json'

    # Build new table row
    new_table_row = _build_new_table_row(result_file_path, config)
    if dataset !=None:
        dataset_name =dataset
    else:
        dataset_name = config.dataset['name']
    # Build new table row
    # new_table_row = _build_new_table_row(result_file_path, config)

    with open("all_results_"+dataset_name+".csv", 'a') as csvFile:
            wr = csv.DictWriter(csvFile, fieldnames=list(new_table_row.keys()))
            # wr.writeheader() #use when first writing
            wr.writerow(new_table_row )    

    #TODO: delete result file
    print("writing results from ", result_file_name)
    os.remove(result_file_path)


def _read_result_log(file_path: str) -> dict:
    with open(file_path) as json_file:
        result_dict = json.load(json_file)

    return result_dict


def _read_conf_yaml(file_path: str) -> dict:
    with open(file_path, 'r') as yaml_file:
        config_dict = yaml.safe_load(yaml_file)

    return config_dict


def _flatten_dict(input_dict: dict, parent_key: str = '', sep: str = '_') -> dict:
    items = []
    for config_key, config_value in input_dict.items():
        new_key = parent_key + sep + config_key if parent_key else config_key
        if isinstance(config_value, dict):
            items.extend(_flatten_dict(config_value, new_key, sep=sep).items())
        else:
            items.append((new_key, config_value))
    return dict(items)


def _load_markdown_table(gitlab_token: str, base_url: str, project_identifier: str, wiki_page_title: str) -> list:
    # Get Wiki Page
    response = requests.get(base_url + 'api/v4/projects/' + project_identifier + '/wikis/' + wiki_page_title,
                            headers={'PRIVATE-TOKEN': gitlab_token}, verify=False)

    if response.status_code == 404:  # page does not exist
        return []
    elif response.status_code != 200:  # something is strange, abort and throw an error
        print('[ExportResults] Unable to load Wiki page. Status Code: %s' % response.status_code, sys.stderr)
        sys.exit(1)

    # check the format of the wiki page
    content = response.json()
    if content['format'] != 'markdown':
        print('[ExportResults] Abort. The response need to be in Markdown!', sys.stderr)
        sys.exit(1)  # something is strange, abort and throw an error

    # parse table to dict
    markdown_table = content['content']
    rows_clean = []
    for row in markdown_table.splitlines():
        if '|' not in row:  # we only want the table
            continue
        rows_clean.append([entry.strip() for entry in row.split('|') if len(entry) > 0])

    # check if there is content
    if len(rows_clean) < 3:
        return []

    # build dict
    table_content = []
    for row in rows_clean[2:]:
        table_content.append(dict(zip(rows_clean[0], row)))

    return table_content


def _upload_csv_file_to_gitlab(gitlab_token: str, base_url: str, project_identifier: str, table_content: list) -> str:
    # convert to csv file
    csv_content = []
    header = list(table_content[0].keys())
    csv_content.append(','.join(header))
    for entry in table_content:
        csv_content.append(','.join([str(entry[column]) for column in header]))
    csv_content = '\n'.join(csv_content)

    # create new wiki page
    response = requests.post(base_url + 'api/v4/projects/' + project_identifier + '/wikis/attachments',
                             headers={'PRIVATE-TOKEN': gitlab_token}, files={'file': csv_content},
                             verify=False)

    return response.json()['link']['markdown']


def _convert_dict_to_table(values: any) -> str:
    return Tomark.table(values)


def _build_new_table_row(result_file_path: str, config:dict) -> dict:
    # Meta Information
    meta_dict = {
        # 'ID': -1,
        # 'Job': '[' + ci_pipeline_id + '-' + ci_job_id + '](' + git_project_url + '/-/jobs/' + ci_job_id + ')',
        # 'Git Commit ID': '[' + git_commit_id + '](' + git_project_url + '/-/tree/' + git_commit_id + ')',
        # 'Git Tag': '[' + git_tag + '](' + git_project_url + '/-/tags/' + git_tag + ')',
        # 'Status': 'Completed',
    }

    # Parse Result Log
    result_dict = {}
    if result_file_path and os.path.isfile(result_file_path):
        result_dict = _read_result_log(result_file_path)
        result_dict = _flatten_dict(result_dict)

    # Parse Config File
    config_dict = dict(config)
    # if config_file_path and os.path.isfile(config_file_path):
    #     config_dict = _read_conf_yaml(config_file_path)
    #     config_dict = _flatten_dict(config_dict)

    # Update Status, if the result dict is empty, the process was just started or crashed
    # if not result_dict:
    #     meta_dict['Status'] = '**Running**'

    # merge
    new_table_row = meta_dict | result_dict | config_dict

    # Add additional information. Do this here as it should be the last column.
    new_table_row['Config File'] = 'N/A'
    new_table_row['Comment'] = '-'

    # Merge everything
    return new_table_row


def _update_gitlab_wiki_page(gitlab_token: str, base_url: str, project_identifier: str, wiki_page_title: str,
                             new_table_row: dict):
    # Load history from Gitlab Wiki
    table_content = _load_markdown_table(gitlab_token, base_url, project_identifier, wiki_page_title)
    new_table_row['ID'] = len(table_content) + 1

    # check if ID already exists, and update status accordingly
    row_match = [row for row in table_content if row['Job'] == new_table_row['Job']]
    if row_match:
        table_content = [row for row in table_content if row['Job'] != new_table_row['Job']]
        new_table_row['ID'] = row_match[0]['ID']
    if row_match and new_table_row['Status'] == '**Running**':  # Entry exist, but it was marked as running -> Error
        new_table_row['Status'] = '**Error**'

    # Adding missing keys for 'Running' and 'Error' cases, otherwise we cannot build the table
    if table_content:
        for key_values in table_content[0]:
            if key_values not in new_table_row:
                new_table_row[key_values] = 'N/A'
    table_content.append(new_table_row)

    # consistency check: rebuild structure
    header = []
    for row in table_content:  # TODO: Order?
        for column_name in row.keys():
            if column_name not in header:
                header.append(column_name)
    table_content_tmp = []
    for row in table_content:
        new_row = {}
        for value in header:
            new_row[value] = row[value] if value in row else 'N/A'
        table_content_tmp.append(new_row)
    table_content = table_content_tmp

    # Keep this: If it crashes, we still have the content in the CI/CD console
    print(table_content)

    # Upload CSV result file
    markdown_result_file_link = _upload_csv_file_to_gitlab(gitlab_token, base_url, project_identifier, table_content)

    # Build and Save new Markdown Table
    table_markdown = _convert_dict_to_table(table_content)
    wiki_page_content = '**Download CSV:** ' + markdown_result_file_link + '\n\n' + table_markdown

    # delete existing page
    response = requests.delete(base_url + 'api/v4/projects/' + project_identifier + '/wikis/' + wiki_page_title,
                               headers={'PRIVATE-TOKEN': gitlab_token}, verify=False)

    if response.status_code not in [204, 404]:
        print('[ExportResults] Unable to delete Wiki Table. Status Code: %s' % response.status_code, sys.stderr)
        sys.exit(1)  # something is strange, abort and throw an error

    # create new wiki page
    my_data = {'format': 'markdown',
               'title': wiki_page_title,
               'content': wiki_page_content}
    response = requests.post(base_url + 'api/v4/projects/' + project_identifier + '/wikis',
                             headers={'PRIVATE-TOKEN': gitlab_token}, verify=False, json=my_data)

    if response.status_code not in [200, 201]:
        print('[ExportResults] Unable to update Wiki Table. Status Code: %s' % response.status_code, sys.stderr)
        sys.exit(1)  # something is strange, abort and throw an error


def _upload_binary_files_ssh(local_source_path: str, remote_ip: str, remote_port: int, remote_target: str,
                             auth_user: str, auth_password: str):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(remote_ip, remote_port, auth_user, auth_password)
    scp = SCPClient(client.get_transport())
    scp.put(local_source_path, recursive=True, remote_path=remote_target)
    scp.close()


if __name__ == "__main__":

    main(result_file_name='results_log.json')

