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

    with open("all_results_feb_"+dataset_name+".csv", 'a') as csvFile:
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

    with open("all_results_feb_"+dataset_name+".csv", 'a') as csvFile:
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

