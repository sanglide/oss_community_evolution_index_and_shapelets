import datetime
import json
import os
import sys
from shutil import copyfile

import analyze_indexes
import calculate_indexes
import community_detection
import data_source_io
import evolution_patterns
import global_settings
import take_snapshot
import visualize

'''
compute the split, shrink, merge, and expand indexes of a single project
project_name: the project name
time_interval_days: the sliding window size in days
return: the resulting series of indexes
'''


def __log_single_migration(community_current, community_prior, community_next, snapshot_node_weights, aggregated_split,
                           aggregated_shrink,
                           aggregated_merge, aggregated_expand, each_split, each_shrink, each_merge, each_expand,
                           eta, entropy_psi, max_entropy_psi,
                           mu, entropy_phi, max_entropy_phi,
                           index,
                           fo):
    fo.write(f"\n..  community [{index}]  ..\n")
    for i in range(len(community_current)):
        comm = community_current[i]
        fo.write("\t{  ")
        comm_set = set(comm)
        for user in comm_set:
            fo.write(f"{user}:[{snapshot_node_weights.get(user)}], ")
        fo.write("}\n")
        fo.write(f"\t\t|-> to next community: split [{each_split[i] if len(each_split) > 0 else -1}], "
                 f"shrink [{each_shrink[i] if len(each_shrink) > 0 else -1}], "
                 f"eta [{eta[i] if len(eta) > 0 else -1}], "
                 f"entropy_psi [{entropy_psi[i] if len(entropy_psi) > 0 else -1}], "
                 f"max_entropy_psi [{max_entropy_psi[i] if len(max_entropy_psi) > 0 else -1}], "
                 f"\n\t\t|-> from prior community: merge [{each_merge[i] if len(each_merge) > 0 else -1}], "
                 f"expand [{each_expand[i] if len(each_expand) > 0 else -1}], "
                 f"mu [{mu[i] if len(mu) > 0 else -1}], "
                 f"entropy_phi [{entropy_phi[i] if len(entropy_phi) > 0 else -1}], "
                 f"max_entropy_phi [{max_entropy_phi[i] if len(max_entropy_phi) > 0 else -1}], "
                 f"\n\n")

    fo.write(f"..  new members to community [{index}] compare to [{index - 1}] ..\n")
    if len(community_prior) > 0:
        # new members to project
        current_set = set()
        for comm in community_current:
            current_set = current_set.union(set(comm))
        prior_set = set()
        for comm in community_prior:
            prior_set = prior_set.union(set(comm))
        for user in current_set.difference(prior_set):
            fo.write(f"\t{user}:[{snapshot_node_weights.get(user)}]\n")
        fo.write("\n")

    fo.write(f"..  members leave community [{index}] compare to [{index + 1}]  ..\n")
    if len(community_next) > 0:
        current_set = set()
        for comm in community_current:
            current_set = current_set.union(set(comm))
        next_set = set()
        for comm in community_next:
            next_set = next_set.union(set(comm))
        for user in current_set.difference(next_set):
            fo.write(f"\t{user}:[{snapshot_node_weights.get(user)}]\n")
        fo.write("\n")

    fo.write(f"\n\n==  indexes [{index}] >>> [{index + 1}]  ==\n\n")
    fo.write(f"\tsplit = [{aggregated_split}], "
             f"shrink = [{aggregated_shrink}], "
             f"merge = [{aggregated_merge}], "
             f"expand = [{aggregated_expand}]\n")


def __log_indexes(developers_count,developers_change_count,community_folder_path, project_name, aggregated_split, aggregated_shrink, aggregated_merge,
                  aggregated_expand, start_time_list):
    project_name = project_name.replace("/", "__")
    fo_index = open(community_folder_path + project_name + "_indexes.csv", "w")
    fo_index.write("time,split,shrink,merge,expand,developers_count,developers_change_count\n")
    for i in range(len(aggregated_split)):
        if start_time_list is not None:
            fo_index.write(
                f"{start_time_list[i]},{aggregated_split[i]},{aggregated_shrink[i]},{aggregated_merge[i]},{aggregated_expand[i]},{developers_count[i]},{developers_change_count[i]}\n")
        else:
            if i<len(developers_count) and i<len(developers_change_count):
                fo_index.write(
                f"0000-00-00T00:00:00Z,{aggregated_split[i]},{aggregated_shrink[i]},{aggregated_merge[i]},{aggregated_expand[i]},{developers_count[i]},{developers_change_count[i]}\n")
            else:
                # print(f'ddd {len(aggregated_split)} {len(aggregated_shrink)} {len(aggregated_merge)} {len(developers_count)}')
                fo_index.write(
                    f"0000-00-00T00:00:00Z,{aggregated_split[i]},{aggregated_shrink[i]},{aggregated_merge[i]},{aggregated_expand[i]},{0},{0}\n")
    fo_index.flush()
    fo_index.close()


def __log_all(community_list, project_name, snapshot_node_weights_list, aggregated_split, aggregated_shrink,
              aggregated_merge, aggregated_expand, split_index_list, shrink_index_list, eta_list, entropy_psi_list,
              max_entropy_psi_list, merge_index_list, expand_index_list, mu_list, entropy_phi_list,
              max_entropy_phi_list, start_time_list, community_folder_path,developers_count,developers_change_count):
    # log the communities detected, one for each line, followed by the indexes for the community
    # between two lines of communities is the evolutionary indexes
    project_name = project_name.replace("/", "__")
    if global_settings.TURN_ON_TEXT_LOG:
        fo = open(community_folder_path + project_name + "_log.txt", "w")
        for i in range(len(community_list)):
            __log_single_migration(community_list[i], community_list[i - 1] if i > 0 else [],
                                   community_list[i + 1] if i < len(community_list) - 1 else [],
                                   snapshot_node_weights_list[i],
                                   aggregated_split[i] if i < len(community_list) - 1 else -1,
                                   aggregated_shrink[i] if i < len(community_list) - 1 else -1,
                                   aggregated_merge[i] if i < len(community_list) - 1 else -1,
                                   aggregated_expand[i] if i < len(community_list) - 1 else -1,
                                   split_index_list[i] if i < len(community_list) - 1 else [],
                                   shrink_index_list[i] if i < len(community_list) - 1 else [],
                                   merge_index_list[i - 1] if i > 0 else [],
                                   expand_index_list[i - 1] if i > 0 else [],
                                   eta_list[i] if i < len(community_list) - 1 else [],
                                   entropy_psi_list[i] if i < len(community_list) - 1 else [],
                                   max_entropy_psi_list[i] if i < len(community_list) - 1 else [],
                                   mu_list[i - 1] if i > 0 else [], entropy_phi_list[i - 1] if i > 0 else [],
                                   max_entropy_phi_list[i - 1] if i > 0 else [], i, fo)
        fo.flush()
        fo.close()
    __log_indexes(developers_count,developers_change_count,community_folder_path, project_name, aggregated_split, aggregated_shrink, aggregated_merge,
                  aggregated_expand, start_time_list)


def __log_communities_detected(community_list, snapshot_node_weights_list, project_name, result_folder_path):
    assert (len(community_list) == len(snapshot_node_weights_list))
    project_name = project_name.replace("/", "__")
    fo = open(result_folder_path + "communities/" + project_name + ".json", "w")
    for i in range(len(community_list)):
        # output as json, which can be load later
        comm_dict = {"time_index": i}
        comm_list = []
        for comm in community_list[i]:
            single_comm = []
            for user in comm:
                single_comm.append({"user": user, "weight": snapshot_node_weights_list[i].get(user)})
            comm_list.append(single_comm)
        comm_dict['communities'] = comm_list
        fo.write(json.dumps(comm_dict))
        fo.write("\n")

    fo.flush()
    fo.close()


# def __load_communities_detected(dump_file_path):
#     assert os.path.exists(dump_file_path)
#     print(f"\nLoad communities from {dump_file_path}\n")
#     # complete list over time
#     community_list_load = []
#     snapshot_node_weights_list_load = []
#     file = open(dump_file_path, 'r')
#     try:
#         text_lines = file.readlines()
#         for line in text_lines:
#             # one snapshot
#             comm_dict = json.loads(line)
#             # communities in a snapshot
#             comms_snapshot = []
#             # user weights in a snapshot
#             node_weights = {}
#             for comm in comm_dict['communities']:
#                 # each community
#                 user_list = []
#                 for individual in comm:
#                     user_list.append(individual['user'])
#                     assert not (individual['user'] in node_weights)
#                     node_weights[individual['user']] = individual['weight']
#                 comms_snapshot.append(user_list)
#             community_list_load.append(comms_snapshot)
#             snapshot_node_weights_list_load.append(node_weights)
#     except:
#         assert False
#     finally:
#         file.close()
#     return community_list_load, snapshot_node_weights_list_load


def __log_commit_count_list(snapshot_commit_count_list, result_folder_path, project_name):
    project_name = project_name.replace("/", "__")
    fo = open(result_folder_path + "communities/" + project_name + "_commit_count.json", "w")
    fo.write(json.dumps({'snapshot_commit_count_list': snapshot_commit_count_list}))
    fo.write("\n")
    fo.flush()
    fo.close()


# def __load_commit_count_list(dump_file_path):
#     file = open(dump_file_path, 'r')
#     line = file.readline()
#     commit_count_dict = json.loads(line)
#     return commit_count_dict['snapshot_commit_count_list']


def __perform_community_detection(project_name: str, time_interval_days: int, gamma: float, result_folder_path):
    # build DSN snapshots of the project from raw data
    snapshot_edge_weights_list, snapshot_node_weights_list, snapshot_commit_count_list, start_time_list, \
    snapshot_issue_count_list, snapshot_pr_count_list, snapshot_issue_pr_count_list, snapshot_member_count_list,developers_count,developers_change_count \
        = take_snapshot.build_dsn_snapshots(project_name, time_interval_days, gamma)
    assert (len(snapshot_edge_weights_list) == len(snapshot_node_weights_list) == len(snapshot_commit_count_list))

    # detect communities in all snapshots
    # it is possible that the count of communities is 0 or 1 in a snapshot
    community_list \
        = community_detection.detect_communities_from_snapshot_list_parallel(snapshot_edge_weights_list,
                                                                             snapshot_node_weights_list,
                                                                             project_name)
    data_dict={"snapshot_edge_weights_list":snapshot_edge_weights_list,
               "snapshot_node_weights_list":snapshot_node_weights_list}

    os.makedirs(os.path.dirname(f'{result_folder_path}/graph/{project_name}.json'), exist_ok=True)
    with open(f'{result_folder_path}/graph/{project_name}.json', 'w') as file:
        json.dump(data_dict, file)

    # log the communities detected for further analysis

    __log_communities_detected(community_list, snapshot_node_weights_list, project_name, result_folder_path)
    __log_commit_count_list(snapshot_commit_count_list, result_folder_path, project_name)
    return community_list, snapshot_node_weights_list, snapshot_commit_count_list, start_time_list, \
           snapshot_issue_count_list, snapshot_pr_count_list, snapshot_issue_pr_count_list, snapshot_member_count_list,developers_count,developers_change_count


def remove_empty_snapshots(community_list, snapshot_node_weights_list):
    assert (len(community_list) == len(snapshot_node_weights_list))
    community_list_processed = []
    snapshot_node_weights_list_processed = []
    for i in range(len(community_list)):
        if len(community_list[i]) <= 1:
            print("remove")
            # assert len(snapshot_node_weights_list[i]) <= 1
            continue
        else:
            community_list_processed.append(community_list[i])
            snapshot_node_weights_list_processed.append(snapshot_node_weights_list[i])
    return community_list_processed, snapshot_node_weights_list_processed


def calculate_indexes_single_project \
                (project_name: str,
                 time_interval_days: int,
                 gamma: float,
                 result_folder_path,
                 community_folder_path):
    # print(f"gamma = {gamma}")

    community_list, snapshot_node_weights_list, snapshot_commit_count_list, start_time_list, \
    snapshot_issue_count_list, snapshot_pr_count_list, snapshot_issue_pr_count_list, snapshot_member_count_list,developers_count,developers_change_count \
        = __perform_community_detection(project_name, time_interval_days, gamma, result_folder_path)

    # community_list, snapshot_node_weights_list = remove_empty_snapshots(community_list, snapshot_node_weights_list)

    # perform index calculation for all consecutive pair of snapshots
    aggregated_split, aggregated_shrink, aggregated_merge, aggregated_expand, \
    split_index_list, shrink_index_list, eta_list, entropy_psi_list, max_entropy_psi_list, \
    merge_index_list, expand_index_list, mu_list, entropy_phi_list, max_entropy_phi_list, \
    max_entropy_list_plus_sigma_psi, max_entropy_list_plus_sigma_phi \
        = calculate_indexes.calculate_all_indexes(community_list, snapshot_node_weights_list, project_name)

    assert (len(aggregated_split) == len(start_time_list) - 1)

    # compare with existing evolution pattern detection results
    match_and_pattern_assign_results = evolution_patterns.match_and_assign(community_list, snapshot_node_weights_list)
    match_and_pattern_assign_results = evolution_patterns.append_index_based_results(match_and_pattern_assign_results,
                                                                                     split_index_list,
                                                                                     shrink_index_list,
                                                                                     max_entropy_list_plus_sigma_psi,
                                                                                     merge_index_list,
                                                                                     expand_index_list,
                                                                                     max_entropy_list_plus_sigma_phi)
    confusion_matrix = evolution_patterns.get_evolution_patterns_confusion_matrix(match_and_pattern_assign_results)
    evolution_patterns.calculate_log_accuracy(confusion_matrix, project_name, result_folder_path)

    # rest of the returns for debugging purposes

    __log_all(community_list, project_name, snapshot_node_weights_list, aggregated_split, aggregated_shrink,
              aggregated_merge, aggregated_expand, split_index_list, shrink_index_list, eta_list, entropy_psi_list,
              max_entropy_psi_list, merge_index_list, expand_index_list, mu_list, entropy_phi_list,
              max_entropy_phi_list, start_time_list, community_folder_path,developers_count,developers_change_count)

    # return the result
    # split and shrink for communities in time t, merge and expand for communities in time t+1
    # start time list for a series of t's
    return aggregated_split, aggregated_shrink, aggregated_merge, aggregated_expand, \
           confusion_matrix, snapshot_commit_count_list, start_time_list, \
           snapshot_issue_count_list, snapshot_pr_count_list, snapshot_issue_pr_count_list, snapshot_member_count_list,developers_count,developers_change_count


# def __log_indexes(aggregated_split, aggregated_shrink, aggregated_merge, aggregated_expand, filepath):
#     assert (len(aggregated_split) == len(aggregated_shrink) == len(aggregated_merge) == len(aggregated_expand))
#     fo = open(filepath + "_log.csv", "w")
#     fo.write("split,shrink,merge,expand\n")
#     for i in range(len(aggregated_split)):
#         fo.write(f"{aggregated_split[i]},{aggregated_shrink[i]},{aggregated_merge[i]},{aggregated_expand[i]}\n")
#     fo.flush()
#     fo.close()


def __log_brief_config(result_folder_path, evolution_pattern_threshold):
    fo = open(result_folder_path + "concurrent_validity.txt", "w")
    fo.write(str(evolution_pattern_threshold) + "\n")
    fo.flush()
    fo.close()

    fo = open(result_folder_path + "discrimant_validity.txt", "w")
    fo.write(str(evolution_pattern_threshold) + "\n")
    fo.flush()
    fo.close()


def test_kill_and_exit():
    if os.path.exists("./kill"):
        os.remove("./kill")
        f = open("./killed", "w")
        f.write(str(datetime.datetime.now()))
        f.close()
        sys.exit(-1)


def execute_all():
    # Entrance, prepare dir
    time_of_execution = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%SZ')
    result_folder_path = "./result/" + time_of_execution + "_interval_" + str(
        global_settings.TIME_INTERVAL) + "_days_x_" + str(global_settings.TIME_INTERVALS_IN_SNAPSHOT) + "/"
    os.makedirs(result_folder_path)
    __log_brief_config(result_folder_path, global_settings.EVOLUTION_PATTERN_THRESHOLD)
    fig_folder_path = result_folder_path + "fig/"
    community_folder_path = result_folder_path + "community_evolution/"
    os.makedirs(fig_folder_path)
    os.makedirs(community_folder_path)
    os.makedirs(result_folder_path + "communities/")
    copyfile("./global_settings.py", result_folder_path + "global_settings.py")
    confusion_matrix_sum = [([0] * 7) for i in range(7)]

    total_aggregated_split = []
    total_aggregated_shrink = []
    total_aggregated_merge = []
    total_aggregated_expand = []
    total_developers_count = []
    total_developers_change_count = []

    projects_aggregated_split = []
    projects_aggregated_shrink = []
    projects_aggregated_merge = []
    projects_aggregated_expand = []
    projects_commit_count = []
    projects_snapshot_start_time_list = []
    projects_snapshot_issue_count_list = []
    projects_snapshot_pr_count_list = []
    projects_snapshot_issue_pr_count_list = []
    projects_snapshot_member_count_list = []
    projects_developers_count=[]
    projects_developers_change_count=[]

    #
    # todo
    if global_settings.USE_NEW_DATA:
        # print(f'dddddd read repo name from 200repo-list')
        with open('200个仓库的列表.txt', 'r') as file:
            lines = [line.strip() for line in file]

        global_settings.proj_list = lines
        # global_settings.proj_list = data_source_io.list_projects("all")
        # global_settings.activeProjL = data_source_io.list_projects("active")
        # global_settings.failProjL = data_source_io.list_projects("failed")

    for proj in global_settings.proj_list:

        test_kill_and_exit()
        # calculate the indexes for a project
        p_aggregated_split, p_aggregated_shrink, \
        p_aggregated_merge, p_aggregated_expand, \
        p_confusion_matrx, snapshot_commit_count_list, \
        start_time_list, \
        snapshot_issue_count_list, snapshot_pr_count_list, snapshot_issue_pr_count_list, snapshot_member_count_list,developers_count,developers_change_count \
            = calculate_indexes_single_project(proj,
                                               global_settings.TIME_INTERVAL,
                                               global_settings.GAMMA,
                                               result_folder_path,
                                               community_folder_path)
        # store index result
        total_aggregated_split.extend(p_aggregated_split)
        total_aggregated_shrink.extend(p_aggregated_shrink)
        total_aggregated_merge.extend(p_aggregated_merge)
        total_aggregated_expand.extend(p_aggregated_expand)
        total_developers_count.extend(developers_count)
        total_developers_change_count.extend(developers_change_count)

        projects_aggregated_split.append(p_aggregated_split)
        projects_aggregated_shrink.append(p_aggregated_shrink)
        projects_aggregated_merge.append(p_aggregated_merge)
        projects_aggregated_expand.append(p_aggregated_expand)
        projects_commit_count.append(snapshot_commit_count_list)
        projects_snapshot_start_time_list.append(start_time_list)
        projects_snapshot_issue_count_list.append(snapshot_issue_count_list)
        projects_snapshot_pr_count_list.append(snapshot_pr_count_list)
        projects_snapshot_issue_pr_count_list.append(snapshot_issue_pr_count_list)
        projects_snapshot_member_count_list.append(snapshot_member_count_list)
        projects_developers_count.append(developers_count)
        projects_developers_change_count.append(developers_change_count)

        # traditional method for community evolution pattern detection
        confusion_matrix_sum = evolution_patterns.sum_confusion_matrix(confusion_matrix_sum, p_confusion_matrx)

        # status = "active"
        # if proj in global_settings.failProjL:
        #     status = "fail"
        #
        # visualize.plot_aggregate_index_curves_single_proj(p_aggregated_split, p_aggregated_shrink, p_aggregated_merge,
        #                                                   p_aggregated_expand, start_time_list, proj, status,
        #                                                   result_folder_path)

    # summary
    # concurrent validity
    evolution_patterns.calculate_log_accuracy(confusion_matrix_sum, 'summarize_all_projects', result_folder_path)
    # discriminant validity
    __log_indexes(total_developers_count,developers_change_count,community_folder_path, 'total', total_aggregated_split, total_aggregated_shrink,
                  total_aggregated_merge, total_aggregated_expand, None,)
    analyze_indexes.execute_index_independency_check(result_folder_path)
    # prepare data for correlation with project productivity
    analyze_indexes.prepare_data_productivity(projects_aggregated_split, projects_aggregated_shrink,
                                              projects_aggregated_merge, projects_aggregated_expand,
                                              projects_commit_count, projects_snapshot_start_time_list,
                                              projects_snapshot_issue_count_list, projects_snapshot_pr_count_list,
                                              projects_snapshot_issue_pr_count_list,
                                              projects_snapshot_member_count_list,
                                              projects_developers_change_count,
                                              result_folder_path)

if __name__ == "__main__":
    # window_size_experiment()
    execute_all()
    f = open("./terminated", "w")
    f.write(str(datetime.datetime.now()))
    f.close()
