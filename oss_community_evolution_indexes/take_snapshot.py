import copy
import datetime
import math
import random
import sys
import time
from itertools import combinations

import pandas as pd

# from data_source_io import read_data_from_csv
import data_source_io
import global_settings

'''
Interface provided by this module is build_dsn_snapshots().
The module builds a series of snapshots from the raw data of issue and PR discussions.
Key difference compared to the previous version is that we do not remove empty windows 
when generating the series of local DSNs. 
'''

__time_fmt_str = '%Y-%m-%dT%H:%M:%SZ'


def extractNode(df):
    '''
    df：scheme [startTime,createUser,commentsUser,proj]
    '''

    # df = pd.DataFrame(getSnapShots.getSnapShots("2010-09-02T15:58:28Z",40,"angular_angular.js"))
    def other2list(x):
        x = eval(x)
        res = []
        for each in x:
            res.append(each)
        return res

    df['commentsUser'] = df['commentsUser'].apply(lambda x: other2list(x))
    createUserList, user = df['createUser'].to_list(), df['commentsUser'].to_list()
    commentsUserListCP = copy.deepcopy(user)
    for i in range(len(createUserList)):
        # one for each conversation
        user[i].append(createUserList[i])
        user[i] = list(set(user[i]))
    df['commentsUser'], df['user'] = commentsUserListCP, user

    # scheme
    #              startTime createUser commentsUser                proj                  user
    # 0  2010-09-02T15:58:28Z   rodrigob    [mhevery]  angular_angular.js   [rodrigob, mhevery]
    return df


def extractEdge(df):
    '''
    df:scheme [startTime,createUser,commentsUser,proj]
    retrun:
    '''
    nodeList = df['user']
    linkList = []

    for each in nodeList.to_list():
        for evey in each:
            if type(evey) == float:  # remove NaN
                each.remove(evey)

    for each in nodeList:
        linkList.append(list(combinations(each, 2)))
    df['link'] = linkList

    return df


# def __get_one_window(df: pd.DataFrame, start_time, end_time, project_name: str = ''):
#     """
#     df: scheme [startTime,createUser,commentsUser,proj]
#     """
#     # str compare
#     # df_temp = df[df['startTime'].apply(lambda x: start_time <= x < end_time)]
#     df_temp = df[df['startTime'].apply(
#         lambda x: time.strptime(start_time, __time_fmt_str) <= time.strptime(x, __time_fmt_str) < time.strptime(
#             end_time, __time_fmt_str))]
#     # assert len(df_temp) == len(df_temp_test)
#     # if len(df_temp) > 0:
#     #     for i in range(len(df_temp)):
#     #         assert df_temp.iloc[i]['startTime'] == df_temp_test.iloc[i]['startTime']
#     #     if random.random() < 0.01:
#     #         print(f"checkpoint {len(df_temp)} :: {df_temp.iloc[0]['startTime']} == {df_temp_test.iloc[0]['startTime']}"
#     #               f" :: {df_temp.iloc[len(df_temp) - 1]['startTime']} == {df_temp_test.iloc[len(df_temp) - 1]['startTime']}")
#     return pd.DataFrame(df_temp)
#
#
# def __get_all_windows(df: pd.DataFrame, time_interval_days: int = 0, project_name: str = ''):
#     """
#     Segment the data of a project into windows
#     df: scheme [startTime,createUser,commentsUser,proj]
#     """
#     assert (time_interval_days > 0)
#     count_empty_window = 0
#     count_non_empty_window = 0
#     df_windows = []
#     start_time_list = []
#     global_start_time = df['startTime'].tolist()[0]
#     global_end_time = df['startTime'].tolist()[-1]
#     start_time = global_start_time
#     while start_time < global_end_time:
#         start_time_list.append(start_time)
#         end_time = (datetime.datetime.strptime(start_time, __time_fmt_str)
#                     + datetime.timedelta(days=time_interval_days)).strftime(
#             __time_fmt_str)  # start time+time interval days
#         if end_time == global_end_time:  # add one day to fix global end of the data
#             end_time = (datetime.datetime.strptime(end_time, __time_fmt_str)
#                         + datetime.timedelta(days=1)).strftime(__time_fmt_str)
#         window = __get_one_window(df, start_time, end_time, project_name)
#         if len(window) > 0:
#             count_non_empty_window = count_non_empty_window + 1
#         else:
#             count_empty_window = count_empty_window + 1
#         df_windows.append(window)
#         start_time = end_time
#     print(f"\tCount of non-empty windows [{count_non_empty_window}], empty windows [{count_empty_window}]")
#     return df_windows, start_time_list


def __get_all_windows_fast(df: pd.DataFrame, time_interval_days: int = 0, project_name: str = ''):
    """
    Segment the data of a project into windows
    df: scheme [startTime,createUser,commentsUser,proj]
    """
    assert (time_interval_days > 0)
    count_empty_window = 0
    count_non_empty_window = 0
    df_windows = []
    start_time_list = []
    count_issue_list = []
    count_pr_list = []
    count_issue_pr_list = []
    global_start_time = df['startTime'].tolist()[0]
    global_end_time = df['startTime'].tolist()[-1]
    start_time = global_start_time

    current_idx = 0
    dataframe_len = df.shape[0]
    print(f'size of issue pr data frame: {dataframe_len}')

    while start_time < global_end_time:
        start_time_list.append(start_time)
        end_time = (datetime.datetime.strptime(start_time, __time_fmt_str)
                    + datetime.timedelta(days=time_interval_days)).strftime(
            __time_fmt_str)  # start time+time interval days
        if end_time == global_end_time:  # add one day to fix global end of the data
            end_time = (datetime.datetime.strptime(end_time, __time_fmt_str)
                        + datetime.timedelta(days=1)).strftime(__time_fmt_str)

        window = []
        count_issue = 0
        count_pr = 0

        while current_idx < dataframe_len and time.strptime(start_time, __time_fmt_str) <= time.strptime(
                df.iloc[current_idx]['startTime'], __time_fmt_str) < time.strptime(end_time, __time_fmt_str):
            window.append(df.iloc[current_idx])
            if global_settings.USE_NEW_DATA:
                if df.iloc[current_idx]['type'] == 'issue':
                    count_issue = count_issue + 1
                elif df.iloc[current_idx]['type'] == 'pr':
                    count_pr = count_pr + 1
                else:
                    print(f"unknown type: {df.iloc[current_idx]['type']}")
                    sys.exit(-1)
            current_idx = current_idx + 1

        if len(window) > 0 and random.random() < 0.01:
            print(
                f"checkpoint issue_pr {len(window)} :: {window[0]['startTime']} :: {window[len(window) - 1]['startTime']}")

        if len(window) > 0:
            count_non_empty_window = count_non_empty_window + 1
        else:
            count_empty_window = count_empty_window + 1
        df_windows.append(pd.DataFrame(window, columns=["startTime", "createUser", "commentsUser", "proj", "year"]))
        count_issue_list.append(count_issue)
        count_pr_list.append(count_pr)
        count_issue_pr_list.append(count_issue + count_pr)
        start_time = end_time
    print(f"\tCount of non-empty windows [{count_non_empty_window}], empty windows [{count_empty_window}]")
    return df_windows, start_time_list, count_issue_list, count_pr_list, count_issue_pr_list


# def __get_commit_count_one_window(df_project_commits, start_time, end_time, time_fmt):
#     # df_temp = df_project_commits[df_project_commits['created_at'].apply(
#     #     lambda x: start_time <= x < end_time)]
#     df_temp = df_project_commits[df_project_commits['created_at'].apply(
#         lambda x: time.strptime(start_time, time_fmt) <= time.strptime(x, time_fmt) < time.strptime(end_time,
#                                                                                                     time_fmt))]
#     # assert len(df_temp) == len(df_temp_test)
#     # if len(df_temp) > 0:
#     #     for i in range(len(df_temp)):
#     #         assert df_temp.iloc[i]['created_at'] == df_temp_test.iloc[i]['created_at']
#     #     if random.random() < 0.01:
#     #         print(f"checkpoint {len(df_temp)} :: {df_temp.iloc[0]['created_at']} == {df_temp_test.iloc[0]['created_at']}"
#     #               f" :: {df_temp.iloc[len(df_temp) - 1]['created_at']} == {df_temp_test.iloc[len(df_temp) - 1]['created_at']}")
#     return len(df_temp)


# def __get_all_commits(df: pd.DataFrame, time_interval_days: int = 0, project_name: str = ''):
#     fse_commit_date_time_fmt = '%Y-%m-%d %H:%M:%S'
#     if global_settings.USE_NEW_DATA:
#         df_project_commits = data_source_io.load_commit_data_proj(project_name)
#         commit_time_fmt = __time_fmt_str
#     else:
#         df_commits = pd.read_csv("data/projCommit.csv")
#         if project_name:
#             df_project_commits = df_commits[df_commits['proj_name'] == project_name]
#         else:
#             assert False
#         commit_time_fmt = fse_commit_date_time_fmt
#     df_project_commits = df_project_commits.sort_values(by='created_at')
#
#     list_commit_counts = []
#     global_start_time = (datetime.datetime.strptime(df['startTime'].tolist()[0], __time_fmt_str)).strftime(
#         commit_time_fmt)
#     global_end_time = (datetime.datetime.strptime(df['startTime'].tolist()[-1], __time_fmt_str)).strftime(
#         commit_time_fmt)
#     start_time = global_start_time
#     while start_time < global_end_time:
#         end_time = (datetime.datetime.strptime(start_time, commit_time_fmt)
#                     + datetime.timedelta(days=time_interval_days)).strftime(commit_time_fmt)
#         if end_time == global_end_time:  # add one day to fix global end of the data
#             end_time = (datetime.datetime.strptime(end_time, commit_time_fmt)
#                         + datetime.timedelta(days=1)).strftime(commit_time_fmt)
#         count_commit = __get_commit_count_one_window(df_project_commits, start_time, end_time, commit_time_fmt)
#         list_commit_counts.append(count_commit)
#         start_time = end_time
#     return list_commit_counts


def __get_all_commits_fast(df: pd.DataFrame, time_interval_days: int = 0, project_name: str = ''):
    fse_commit_date_time_fmt = '%Y-%m-%d %H:%M:%S'
    if global_settings.USE_NEW_DATA:
        df_project_commits = data_source_io.load_commit_data_proj(project_name)
        commit_time_fmt = __time_fmt_str
    else:
        df_commits = pd.read_csv("data/projCommit.csv")
        if project_name:
            df_project_commits = df_commits[df_commits['proj_name'] == project_name]
        else:
            assert False
        commit_time_fmt = fse_commit_date_time_fmt
    df_project_commits = df_project_commits.sort_values(by='created_at')

    list_commit_counts = []
    global_start_time = (datetime.datetime.strptime(df['startTime'].tolist()[0], __time_fmt_str)).strftime(
        commit_time_fmt)
    global_end_time = (datetime.datetime.strptime(df['startTime'].tolist()[-1], __time_fmt_str)).strftime(
        commit_time_fmt)
    start_time = global_start_time

    current_idx = 0
    dataframe_len = df_project_commits.shape[0]
    print(f'size of commit data frame: {dataframe_len}')

    while start_time < global_end_time:
        end_time = (datetime.datetime.strptime(start_time, commit_time_fmt)
                    + datetime.timedelta(days=time_interval_days)).strftime(commit_time_fmt)
        if end_time == global_end_time:  # add one day to fix global end of the data
            end_time = (datetime.datetime.strptime(end_time, commit_time_fmt)
                        + datetime.timedelta(days=1)).strftime(commit_time_fmt)

        window = []

        while current_idx < dataframe_len and time.strptime(start_time, commit_time_fmt) > time.strptime(
                df_project_commits.iloc[current_idx]['created_at'], commit_time_fmt):
            current_idx = current_idx + 1

        while current_idx < dataframe_len and time.strptime(start_time, commit_time_fmt) <= time.strptime(
                df_project_commits.iloc[current_idx]['created_at'], commit_time_fmt) < time.strptime(end_time,
                                                                                                     commit_time_fmt):
            window.append(df_project_commits.iloc[current_idx])
            current_idx = current_idx + 1

        if len(window) > 0 and random.random() < 0.01:
            print(
                f"checkpoint commit {len(window)} :: {window[0]['created_at']} :: {window[len(window) - 1]['created_at']}")

        count_commit = len(window)
        list_commit_counts.append(count_commit)
        start_time = end_time
    return list_commit_counts


def __calculate_edge_weights(local_dsn_df: pd.DataFrame):
    def edge_weight_func(dc):
        for key in dc.keys():
            if global_settings.EDGE_WEIGHT_FUNC == 0:
                dc[key] = math.log(dc[key] + 1, math.e)  # edge weight function
            elif global_settings.EDGE_WEIGHT_FUNC == 1:
                dc[key] = 1
        return dc

    time_list = local_dsn_df['startTime'].tolist()
    link_list = local_dsn_df['link'].tolist()
    edge_count_dic = {}
    for i in range(len(time_list)):
        pairs = link_list[i]
        try:
            for pair in pairs:
                pair = sorted(pair)
                key_edge = "@".join(pair)  # user1@user2
                edge_count_dic[key_edge] = edge_count_dic.get(key_edge, 0) + 1  # 两位开发者讨论的次数
        except:
            print(f"Exception in calculate edge weights {time_list[i]}, {pairs}, {pair}")

    return edge_weight_func(edge_count_dic)


def __calculate_node_weights(local_dsn_df: pd.DataFrame, local_edge_weights_dic):
    def node_weight_func(dc):
        for key in dc.keys():
            if global_settings.NODE_WEIGHT_FUNC == 0:
                dc[key] = math.log(dc[key] + 1, math.e)  # + 1  # node weight function
            elif global_settings.NODE_WEIGHT_FUNC == 1:
                dc[key] = 1
        return dc

    time_list = local_dsn_df['startTime'].tolist()
    link_list = local_dsn_df['link'].tolist()
    user_list = local_dsn_df['user'].tolist()
    unique_link_dic = {}
    node_degree_dic = {}
    for i in range(len(time_list)):
        for user in user_list[i]:
            # default value for each user
            node_degree_dic[user] = node_degree_dic.get(user, 0)
            # if we put +1 here, then the weight the node is its degree + #conversations it joins

    for i in range(len(time_list)):
        pairs = link_list[i]
        try:
            for pair in pairs:
                pair = sorted(pair)
                key_edge = "@".join(pair)  # user1@user2
                unique_link_dic[key_edge] = unique_link_dic.get(key_edge, 0) + 1
                assert (key_edge in local_edge_weights_dic)
                if unique_link_dic.get(key_edge, 0) <= 1:
                    user1 = pair[0]
                    user2 = pair[1]
                    assert (user1 in node_degree_dic and user2 in node_degree_dic)
                    # the weighted degree of each node, calculated from the edge weights
                    node_degree_dic[user1] = node_degree_dic.get(user1, 0) + local_edge_weights_dic.get(key_edge)
                    node_degree_dic[user2] = node_degree_dic.get(user2, 0) + local_edge_weights_dic.get(key_edge)
        except:
            print(f"Exception in calculate node weights {time_list[i]}, {pairs}, {pair}, {user}")

    # ln(x+1), smooth value
    return node_weight_func(node_degree_dic)


def __build_single_local_dsn(df_window: pd.DataFrame):
    """
    build a local DSN graph for a single window
    df_window: scheme [startTime,createUser,commentsUser,proj]
    """
    if len(df_window) > 0:
        win_start_time = datetime.datetime.strptime(df_window['startTime'].tolist()[0], __time_fmt_str)
        win_end_time = datetime.datetime.strptime(df_window['startTime'].tolist()[-1], __time_fmt_str)
        assert (win_end_time.strftime(__time_fmt_str) >= win_start_time.strftime(__time_fmt_str))
        assert (win_end_time.strftime(__time_fmt_str) <
                (win_start_time + datetime.timedelta(days=global_settings.TIME_INTERVAL)).strftime(__time_fmt_str))
    # scheme [startTime,createUser,commentsUser,proj,user]
    local_dsn_df = extractNode(df_window)
    # scheme [startTime,createUser,commentsUser,proj,user,link]
    local_dsn_df = extractEdge(local_dsn_df)
    # assign edge weights to the local DSN
    local_edge_weights_dic = __calculate_edge_weights(local_dsn_df)
    # assign node weights to the local DSN
    # the set of nodes contains all the nodes appears in the edges and the standalone nodes
    local_node_weights_dic = __calculate_node_weights(local_dsn_df, local_edge_weights_dic)
    return local_dsn_df, local_edge_weights_dic, local_node_weights_dic


def __aggregate_local_dsn_to_single_snapshot(local_dsn_df_list, local_edge_weights_list, local_node_weights_list,
                                             index, project_name, gamma, list_commit_counts, count_issue_list,
                                             count_pr_list, count_issue_pr_list):
    """
    build a single snapshot by aggregating historical local DSNs
    index is the index of the snapshot
    gamma and beta are the decaying factor and threshold, respectively

    return the aggregated edge and node weights dict
    """

    assert (0 <= index < len(local_dsn_df_list))
    assert (len(local_dsn_df_list) == len(local_edge_weights_list) == len(local_node_weights_list))
    assert (len(local_dsn_df_list) > 0)
    snapshot_edge_weights = {}
    snapshot_node_weights = {}
    snapshot_commit_count = 0
    snapshot_issue_count = 0
    snapshot_pr_count = 0
    snapshot_issue_pr_count = 0
    decay_factor = 1
    current_index = index
    count_intervals = global_settings.TIME_INTERVALS_IN_SNAPSHOT

    # while decay_factor >= beta and current_index >= 0:
    while count_intervals > 0 and current_index >= 0:
        # aggregate the weights
        time_list = local_dsn_df_list[current_index]['startTime'].tolist()
        link_list = local_dsn_df_list[current_index]['link'].tolist()
        user_list = local_dsn_df_list[current_index]['user'].tolist()
        unique_link_dic = {}
        unique_user_dic = {}
        snapshot_commit_count = snapshot_commit_count + list_commit_counts[current_index]
        snapshot_issue_count = snapshot_issue_count + count_issue_list[current_index]
        snapshot_pr_count = snapshot_pr_count + count_pr_list[current_index]
        snapshot_issue_pr_count = snapshot_issue_pr_count + count_issue_pr_list[current_index]
        for i in range(len(time_list)):
            try:
                pairs = link_list[i]
                for pair in pairs:
                    pair = sorted(pair)
                    key_edge = "@".join(pair)  # user1@user2
                    unique_link_dic[key_edge] = unique_link_dic.get(key_edge, 0) + 1
                    assert (key_edge in local_edge_weights_list[current_index])
                    assert (pair[0] in user_list[i])
                    assert (pair[1] in user_list[i])
                    # decay edge weights
                    if unique_link_dic.get(key_edge, 0) <= 1:
                        snapshot_edge_weights[key_edge] = snapshot_edge_weights.get(key_edge, 0) \
                                                          + decay_factor * \
                                                          local_edge_weights_list[current_index].get(key_edge)

                for user_key in user_list[i]:
                    assert (user_key in local_node_weights_list[current_index])
                    unique_user_dic[user_key] = unique_user_dic.get(user_key, 0) + 1
                    # NOTE: do we need to decay node weights? NO
                    if unique_user_dic.get(user_key, 0) <= 1:
                        snapshot_node_weights[user_key] = snapshot_node_weights.get(user_key, 0) + \
                                                          local_node_weights_list[current_index].get(user_key)

            except Exception as e:
                print(f"Exception in aggregate into snapshots {index}, {current_index}, {decay_factor}")
                print(e.args)
                print(str(e))
                print(repr(e))
                assert False
        # update decay_factor
        current_index = current_index - 1
        count_intervals = count_intervals - 1
        decay_factor = decay_factor * gamma

    return snapshot_edge_weights, snapshot_node_weights, snapshot_commit_count, snapshot_issue_count, snapshot_pr_count, snapshot_issue_pr_count


def build_dsn_snapshots(project_name: str, time_interval_days: int, gamma: float):
    """
    build a series of DSN snapshots of a project
    gamma is the decaying factor
    beta is the threshold
    return two lists: edge weights: list of dict, node weights: list of dict
    """

    # extract project raw data from the file
    global df_project_data
    print(f"\nExtract data for project [{project_name}]...")
    start_time = datetime.datetime.now()

    # scheme [startTime,createUser,commentsUser,proj]
    if global_settings.USE_NEW_DATA:
        df_project_data = data_source_io.load_issue_pr_data_proj(project_name)
        df_project_issue_data = data_source_io.load_issue_data_proj(project_name).sort_values(by='startTime')
        df_project_pr_data = data_source_io.load_pr_data_proj(project_name).sort_values(by='startTime')
    else:
        df_raw_data = data_source_io.read_data_from_csv()
        if project_name:
            df_project_data = df_raw_data[df_raw_data['proj'] == project_name]

    df_project_data = df_project_data.sort_values(by='startTime')
    print(df_project_data)

    time_spent = datetime.datetime.now() - start_time
    print(f"Done. Time spent [{time_spent}]")

    ##############################################################################################

    # segment project data into windows of size `time_interval_days`,
    # actually controlled by global_settings.TIME_INTERVAL
    print(f"\nSegment data for project [{project_name}]...")
    start_time = datetime.datetime.now()

    df_windows, start_time_list, count_issue_list, count_pr_list, count_issue_pr_list \
        = __get_all_windows_fast(df_project_data, time_interval_days, project_name)
    # find the count of commits for each window
    list_commit_counts = __get_all_commits_fast(df_project_data, time_interval_days, project_name)

    time_spent = datetime.datetime.now() - start_time
    print(f"Done. Time spent [{time_spent}], #windows = [{len(df_windows)}]")

    ##############################################################################################

    # build local DSN graphs for each window
    # developer social network
    print(f"\nBuild local DSN graphs for project [{project_name}]...")
    start_time = datetime.datetime.now()

    local_dsn_list = []  # relation
    local_edge_weights_list = []  # edge weights value
    local_node_weights_list = []  # node weights
    developers_count=[]
    developers_change_count=[]

    before_developers_set=set()
    for win in df_windows:
        createUser_set=set(win["createUser"])
        commentUser_set=set(win["commentsUser"])
        merged_set = createUser_set.union(commentUser_set)

        developers_count.append(len(merged_set))
        developers_change_count.append(len(merged_set.difference(before_developers_set)))
        before_developers_set=merged_set

        local_dsn_df, local_edge_weights_dic, local_node_weights_dic = __build_single_local_dsn(win)
        local_dsn_list.append(local_dsn_df)
        local_edge_weights_list.append(local_edge_weights_dic)
        local_node_weights_list.append(local_node_weights_dic)
    time_spent = datetime.datetime.now() - start_time
    print(f"Done. Time spent [{time_spent}], #local DSNs = [{len(local_dsn_list)}]")

    ##############################################################################################

    # aggregate local DSNs to generate snapshots
    # with parameters gamma and beta
    # actually controlled by global_settings.GAMMA and global_settings.BETA
    print(f"\nBuild DSN snapshots for project [{project_name}]...")
    start_time = datetime.datetime.now()

    snapshot_edge_weights_list = []
    snapshot_node_weights_list = []
    snapshot_commit_count_list = []
    snapshot_issue_count_list = []
    snapshot_pr_count_list = []
    snapshot_issue_pr_count_list = []
    snapshot_member_count_list = []
    for index in range(len(local_dsn_list)):
        snapshot_edge_weights, snapshot_node_weights, snapshot_commit_count, \
        snapshot_issue_count, snapshot_pr_count, snapshot_issue_pr_count = __aggregate_local_dsn_to_single_snapshot(
            local_dsn_list, local_edge_weights_list, local_node_weights_list, index, project_name, gamma,
            list_commit_counts, count_issue_list, count_pr_list, count_issue_pr_list)

        snapshot_edge_weights_list.append(snapshot_edge_weights)
        snapshot_node_weights_list.append(snapshot_node_weights)
        snapshot_commit_count_list.append(snapshot_commit_count)
        snapshot_issue_count_list.append(snapshot_issue_count)
        snapshot_pr_count_list.append(snapshot_pr_count)
        snapshot_issue_pr_count_list.append(snapshot_issue_pr_count)
        snapshot_member_count_list.append(len(snapshot_node_weights))

    time_spent = datetime.datetime.now() - start_time
    print(f"Done. Time spent [{time_spent}], #DSN snapshots = [{len(snapshot_edge_weights_list)}]")

    ##############################################################################################

    # return the snapshots
    return snapshot_edge_weights_list, snapshot_node_weights_list, snapshot_commit_count_list, start_time_list, \
           snapshot_issue_count_list, snapshot_pr_count_list, snapshot_issue_pr_count_list, snapshot_member_count_list,developers_count,developers_change_count
