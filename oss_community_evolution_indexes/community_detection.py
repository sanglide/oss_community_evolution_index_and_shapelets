# import indexAddNodeWeight_main as ianw # import this module will trigger the execution
import datetime
import math
import sys
from multiprocessing import Pool, Value

import networkx as nx

import global_settings

'''
from networkx.algorithms.community import greedy_modularity_communities
'''
from cnm_algor import greedy_modularity_communities


def __detect_communities_single_snapshot(snapshot_edge_weight_dict, snapshot_node_weight_dict):
    """
    Detect the communities in a single snapshot
    """
    if len(snapshot_node_weight_dict) == 0:
        # if there is no user in the graph, i.e., an empty graph, return an empty list
        print(f"\n\t\tGraph without node found, communities: []")
        return []
    else:
        # need to make sure the number of nodes in the graph is >= 1
        G = nx.Graph()
        # the set of nodes contains all users, including those have edges and those not
        # for user in snapshot_node_weight_dict.keys():
        #     G.add_node(user)
        for edge in snapshot_edge_weight_dict.keys():
            user1, user2 = edge.split("@")[0], edge.split("@")[1]
            # assert (G.has_node(user1) and G.has_node(user2))
            # add edge to the graph, the node will be added if not exists previously
            G.add_edge(user1, user2, weight=snapshot_edge_weight_dict.get(
                edge))  # weight=math.log(snapshot_edge_weight_dict.get(edge) + 1, math.e))
        if len(snapshot_edge_weight_dict) == 0:
            # if no edge is present in the graph, then each node becomes a community
            # Initialize single-node communities
            # communities = {n: frozenset([n]) for n in G}
            # return [communities]
            return []
        else:
            # will through exceptions if the edge set is empty
            # print(f"Is directed {G.is_directed()}")
            return greedy_modularity_communities(G, weight='weight', resolution=global_settings.CNM_RESOLUTION)


# '''
# Non-parallel version, very slow
# '''

# def detect_communities_from_snapshot_list(snapshot_edge_weights_list, snapshot_node_weights_list, project_name):
#     """
#     Detect the communities in a series of snapshots
#     """
#     print(f"\nPerform community detection for project [{project_name}]...")
#     start_time = datetime.datetime.now()
#
#     assert (len(snapshot_edge_weights_list) == len(snapshot_node_weights_list))
#
#     count = 1
#     total_count = len(snapshot_edge_weights_list)
#
#     community_list = []
#     for i in range(len(snapshot_edge_weights_list)):
#         print(f"\r\tProgress for [{project_name}]: [{math.floor(count / total_count * 100)}%] {count} / {total_count}",
#               end="", flush=True)
#         community = detect_communities_single_snapshot(snapshot_edge_weights_list[i], snapshot_node_weights_list[i])
#         community_list.append(community)
#         count = count + 1
#     print()
#
#     time_spent = datetime.datetime.now() - start_time
#     print(f"Done. Time spent [{time_spent}].")
#     return community_list


finish_task_count = 0
total_task_count = 0


def __init_parallel(arg1, arg2):
    ''' store the counter for later use '''
    global finish_task_count, total_task_count
    finish_task_count = arg1
    total_task_count = arg2


def __detect_communities_single_snapshot_parallel(task_dict):
    global finish_task_count, total_task_count
    snapshot_edge_weight_dict = task_dict.get('edge_weights')
    snapshot_node_weight_dict = task_dict.get('node_weights')
    communities = __detect_communities_single_snapshot(snapshot_edge_weight_dict, snapshot_node_weight_dict)
    with finish_task_count.get_lock():
        finish_task_count.value = finish_task_count.value + 1
        print(
            f"\r\tProgress for [{task_dict.get('project_name')}]: "
            f"[{math.floor(finish_task_count.value / total_task_count.value * 100)}%] "
            f"{finish_task_count.value} / {total_task_count.value}",
            end="", flush=True)
    return communities


def detect_communities_from_snapshot_list_parallel(snapshot_edge_weights_list, snapshot_node_weights_list,
                                                   project_name):
    """
    Detect the communities in a series of snapshots
    """
    # global community_list_parallel, total_task_number, completed_task_num

    print(f"\nPerform community detection for project [{project_name}]...")
    start_time = datetime.datetime.now()

    assert (len(snapshot_edge_weights_list) == len(snapshot_node_weights_list))

    global finish_task_count, total_task_count
    finish_task_count = Value('i', 0)
    total_task_count = Value('i', len(snapshot_edge_weights_list))

    task_dict_list = []
    for i in range(len(snapshot_edge_weights_list)):
        task_dict_list.append(
            {'edge_weights': snapshot_edge_weights_list[i], 'node_weights': snapshot_node_weights_list[i],
             'project_name': project_name})

    num_processes = 8

    if 'linux' in sys.platform or 'Linux' in sys.platform:
        num_processes = 51

    with Pool(initializer=__init_parallel, initargs=(finish_task_count, total_task_count),
              processes=num_processes) as p:
        community_list_parallel = p.map(__detect_communities_single_snapshot_parallel, task_dict_list)

    time_spent = datetime.datetime.now() - start_time
    print(f"\nDone. Time spent [{time_spent}].")

    

    return community_list_parallel


def test_community_detection():
    test_user_list = ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8", "u9", "u10"]

    # testcase 1
    test_edge_weight_dict = {"u1@u2": 1, "u1@u3": 1, "u2@u3": 1, "u3@u4": 1, "u4@u5": 1, "u5@u6": 1, "u4@u6": 1,
                             "u6@u7": 1, "u7@u8": 1, "u7@u9": 1, "u7@u10": 1, "u8@u9": 1, "u8@u10": 1, "u9@u10": 1}
    test_node_weight_dict = {"u1": 1, "u2": 1, "u3": 1, "u4": 1, "u5": 1, "u6": 1, "u7": 1, "u8": 1, "u9": 1,
                             "u10": 1}

    community_test_results = __detect_communities_single_snapshot(test_edge_weight_dict, test_node_weight_dict)
    print(community_test_results)

    # testcase 2

    test_edge_weight_dict = {"u1@u2": 1, "u1@u3": 1, "u3@u4": 1, "u4@u5": 1, "u5@u6": 1, "u4@u6": 1,
                             "u6@u7": 1, "u7@u8": 1, "u7@u9": 1, "u7@u10": 1, "u8@u9": 1, "u8@u10": 1, "u9@u10": 1}

    community_test_results = __detect_communities_single_snapshot(test_edge_weight_dict, test_node_weight_dict)
    print(community_test_results)

    # testcase 3

    test_edge_weight_dict = {"u1@u2": 1, "u1@u3": 1, "u2@u3": 1, "u4@u5": 1, "u5@u6": 1, "u4@u6": 1,
                             "u7@u8": 1, "u7@u9": 1, "u7@u10": 1, "u8@u9": 1, "u8@u10": 1, "u9@u10": 1}

    community_test_results = __detect_communities_single_snapshot(test_edge_weight_dict, test_node_weight_dict)
    print(community_test_results)

    # testcase 4

    test_edge_weight_dict = {"u1@u2": 1}

    community_test_results = __detect_communities_single_snapshot(test_edge_weight_dict, test_node_weight_dict)
    print(community_test_results)

    # testcase 5

    test_edge_weight_dict = {"u1@u2": 1, "u1@u3": 1}

    community_test_results = __detect_communities_single_snapshot(test_edge_weight_dict, test_node_weight_dict)
    print(community_test_results)

    # testcase 6

    test_edge_weight_dict = {}
    for i in range(len(test_user_list) - 1):
        for j in range(i + 1, len(test_user_list)):
            test_edge_weight_dict["@".join([test_user_list[i], test_user_list[j]])] = 1
    # print(test_edge_weight_dict)

    community_test_results = __detect_communities_single_snapshot(test_edge_weight_dict, test_node_weight_dict)
    print(community_test_results)


if __name__ == "__main__":
    test_community_detection()
