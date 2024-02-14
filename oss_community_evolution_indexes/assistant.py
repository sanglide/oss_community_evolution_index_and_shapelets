import datetime
import math
import os.path

import data_source_io
import global_settings


def data_set_info():
    result_list = []
    no = 1

    for proj in global_settings.proj_list:
        proj_name = proj.replace('_', '/')
        conversations = open(f'./data/projData/{proj}.csv', 'r').readlines()
        count_lines = len(conversations) - 1
        start_time = datetime.datetime.strptime(conversations[1].split(',')[0], '%Y-%m-%dT%H-%M-%SZ')
        end_time = datetime.datetime.strptime(conversations[len(conversations) - 1].split(',')[0], '%Y-%m-%dT%H-%M-%SZ')
        duration = math.ceil((end_time - start_time).days / 7)  # weeks
        label = 'unknown'
        if proj in global_settings.activeProjL:
            label = 'active'
        elif proj in global_settings.failProjL:
            label = 'inactive'

        # result_list.append(f"{no} & {proj_name} & {count_lines} & {duration} & {label}")
        result_list.append(f"{no} & {proj_name} & {count_lines} & {duration} ")
        no = no + 1


    for i in range(16):
        print(f"{result_list[i]} & {result_list[i + 16]} \\\\")
        # print(f"{result_list[i]} & {result_list[i + 16]} \\\\ \\hline")


def diff_commands():
    path1 = 'result/2022-02-10T15-55-53Z_interval_7_days_12_intvl_gamma_e-1_base/communities/'
    path2 = 'result/2022-02-10T16-19-30Z_interval_7_days/communities/'
    for proj in global_settings.proj_list:
        print(f"diff {path1}{proj}.json {path2}{proj}.json")


def find_repos_missing_files():
    # todo
    global_settings.proj_list = data_source_io.list_projects("all")
    for proj in global_settings.proj_list:
        proj_filename = proj.replace("/", "__")
        issue_path = global_settings.NEW_DATA_DIR + "issue/" + proj_filename + ".csv"
        pr_path = global_settings.NEW_DATA_DIR + "issue/" + proj_filename + ".csv"
        commit_path = global_settings.NEW_DATA_DIR + "commit/" + proj_filename + ".csv"
        if not (os.path.exists(issue_path) and os.path.exists(pr_path) and os.path.exists(commit_path)):
            print(proj)


if __name__ == '__main__':
    # data_set_info()
    # diff_commands()
    find_repos_missing_files()
