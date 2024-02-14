import math
import numpy as np
import datetime


def __calculate_migration_weights(communities_prior, communities_next, node_weights_prior, node_weights_next):
    """
    calculate migration related weights from the migration matrix
    """
    n = len(communities_prior)
    m = len(communities_next)
    # the weights of users in c_{t,i} who migrate to community c_{t+1,j}, i.e., f_t(a_{i,j})
    migration_weights_prior = [[0 for j in range(m)] for i in range(n)]  # j是列，i是行，用来存放t时刻社区和t+1时刻社区的交并比
    # the weights of users in c_{t+1,j} who migrate from community c_{t,i}, i.e., f_{t+1}(a_{i,j})
    migration_weights_next = [[0 for j in range(m)] for i in range(n)]  # j是列，i是行，用来存放t时刻社区和t+1时刻社区的交并比
    # the weights of users in c_{t,i} who leave the project, i.e., f_t(b_i)
    leave_weights_prior = [0 for i in range(n)]
    # the weights of users in c_{t+1,j} who are new users, i.e., f_{t+1}(d_j)
    emerge_weights_next = [0 for j in range(m)]
    # the total weight of users in c_{t,i}, i.e., f_t(c_{t,i})
    total_weights_prior = [0 for i in range(n)]
    # the total weight of users in c_{t+1,j}, i.e., f_{t+1}(c_{t+1,j})
    total_weights_next = [0 for j in range(m)]

    for i in range(n):
        c_prior_i = communities_prior[i]
        for j in range(m):
            try:
                c_next_j = communities_next[j]
                # a_{i,j}
                common_set = set(c_prior_i).intersection(set(c_next_j))
                # f_t(a_{i,j})
                migration_weights_prior[i][j] \
                    = sum([node_weights_prior[user] for user in common_set])
                # f_{t+1}(a_{i,j})
                migration_weights_next[i][j] \
                    = sum([node_weights_next[user] for user in common_set])
            except:
                print(f"{j} xxxxxx {m}: {communities_next}")

    for i in range(n):
        c_prior_i = communities_prior[i]
        # f_t(c_{t,i})
        total_weights_prior[i] = sum([node_weights_prior[user] for user in c_prior_i])
        # f_t(b_i)
        sum_migration_weights_prior = 0
        if m == 0:
            print(f"\tWarning calculate leave weights: n > 0, m == 0, {communities_prior}  ====>  {communities_next}")
        else:
            sum_migration_weights_prior = sum(migration_weights_prior[i][:])
        leave_weights_prior[i] = total_weights_prior[i] - sum_migration_weights_prior
        if leave_weights_prior[i] < 0 and math.fabs(leave_weights_prior[i]) < 1e-7:
            leave_weights_prior[i] = 0

    for j in range(m):
        c_next_j = communities_next[j]
        # f_{t+1}(c_{t+1,j})
        total_weights_next[j] = sum([node_weights_next[user] for user in c_next_j])
        # f_{t+t}(d_j)
        sum_migration_weights_next = 0
        if n == 0:
            print(f"\tWarning calculate emerge weights: n == 0, m > 0, {communities_prior}  ====>  {communities_next}")
        else:
            sum_migration_weights_next = sum(np.transpose(migration_weights_next).tolist()[j][:])
        emerge_weights_next[j] = total_weights_next[j] - sum_migration_weights_next
        if emerge_weights_next[j] < 0 and math.fabs(emerge_weights_next[j]) < 1e-7:
            emerge_weights_next[j] = 0

    return migration_weights_prior, leave_weights_prior, total_weights_prior, \
           migration_weights_next, emerge_weights_next, total_weights_next


def __entropy_h(lst):
    return sum([-i * math.log(i, 2) if i > 0 else 0 for i in lst])


def __entropy_max(n):
    assert (n >= 0)
    if n == 0:
        return 0
    else:
        return -math.log(1 / n, 2)


def __template_calculate_indexes(migration_weights, leave_emerge_weights, total_weights, communities_prior,
                                 communities_next, is_split_merge):
    """
    a template to compute a pair of indexes, i.e., split / shrink, or merge / expand
    """
    n = len(leave_emerge_weights)  # len(migration_weights) might be zero because m is zero
    m = len(np.transpose(migration_weights).tolist())
    assert (len(total_weights) == n >= len(migration_weights))

    if n == 0:
        print(f"\tWarning: n == 0, {communities_prior}  ====>  {communities_next}")
    if m == 0:
        print(f"\tWarning: m == 0, {communities_prior}  ====>  {communities_next}")

    split_merge_indexes = [0 for i in range(n)]
    shrink_expand_indexes = [0 for i in range(n)]
    eta_mu_list = [0 for i in range(n)]
    entropy_list = [0 for i in range(n)]
    max_entropy_list = [0 for i in range(n)]
    max_entropy_list_plus_sigma = [0 for i in range(n)]
    for i in range(n):
        # assert (total_weights[i] > 0) # communities with a single node will have total weight 0 now
        psi_phi = [0 for j in range(m)]
        hat_psi_phi = [0 for j in range(m)]
        for j in range(m):
            # total_weights[i] == 0 means the snapshot is empty
            # we will not get because we will have n == 0 now
            psi_phi[j] = migration_weights[i][j] / total_weights[i] if total_weights[i] > 0 else 0
        sum_psi_phi = sum(psi_phi[:])  # sum == 0 if m  == 0
        for j in range(m):
            # it is possible that m > 0, and sum_psi_phi = 0
            # means nobody stays / migrates from existing communities
            hat_psi_phi[j] = psi_phi[j] / sum_psi_phi if sum_psi_phi > 0 else 0

        eta_mu = leave_emerge_weights[i] / total_weights[i] if total_weights[i] > 0 else 0
        if total_weights[i] > 0:
            if not (math.fabs(sum(psi_phi[:]) + eta_mu - 1) < 1e-5):
                print(f"\n\n\nError: sum(psi_phi[:]) + eta_mu = {sum(psi_phi[:]) + eta_mu}\n\n\n", flush=True)
            assert (math.fabs(sum(psi_phi[:]) + eta_mu - 1) < 1e-5)
        sigma = 0.5 * eta_mu if m == 1 else 0
        entropy = __entropy_h(hat_psi_phi)  # entropy == 0 if m  == 0
        max_entropy = __entropy_max(m)  # max entropy == 0 if m  == 0
        split_merge_indexes[i] = (1 - eta_mu) * entropy
        shrink_expand_indexes[i] = eta_mu * (max_entropy - split_merge_indexes[i] + sigma)
        eta_mu_list[i] = eta_mu
        entropy_list[i] = entropy
        max_entropy_list[i] = max_entropy
        max_entropy_list_plus_sigma[i] = max_entropy + sigma
        # assert (split_merge_indexes[i] >= 0 and shrink_expand_indexes[i] >= 0)
        if not (split_merge_indexes[i] >= 0 and shrink_expand_indexes[i] >= 0):
            print(f"max_entropy = {max_entropy}")
            print(f"sigma = {sigma}")
            print(f"eta_mu = {eta_mu}")
            print(f"leave_emerge_weights[i] = {leave_emerge_weights[i]}")
            print(f"total_weights[i] = {total_weights[i]}")
            print(f"split_merge_indexes[i]   = {split_merge_indexes[i]}")
            print(f"shrink_expand_indexes[i] = {shrink_expand_indexes[i]}")
            assert False
    return split_merge_indexes, shrink_expand_indexes, eta_mu_list, entropy_list, max_entropy_list, max_entropy_list_plus_sigma


def __calculate_split_shrink_indexes_prior(migration_weights_prior, leave_weights_prior, total_weights_prior,
                                           communities_prior, communities_next):
    """
    compute the split and shrink indexes for all communities in time t
    """
    return __template_calculate_indexes(migration_weights_prior, leave_weights_prior, total_weights_prior,
                                        communities_prior, communities_next, True)


def __calculate_merge_expand_indexes_next(migration_weights_next, emerge_weights_next, total_weights_next,
                                          communities_prior, communities_next):
    """
    compute the merge and expand indexes for all communities in time t+1
    """

    return __template_calculate_indexes(np.transpose(migration_weights_next).tolist(), emerge_weights_next,
                                        total_weights_next, communities_prior, communities_next, False)


def __aggregate_index(indexes, weights):
    assert (len(indexes) == len(weights))
    weight_sum = sum(weights[:])
    aggregated_index = sum([indexes[i] * weights[i] / weight_sum for i in range(len(indexes))]) if weight_sum > 0 else 0
    return aggregated_index


def calculate_all_indexes(community_list, snapshot_node_weights_list, project_name):
    """
    calculate the series of split, shrink, merge, and expand indexes given the communities and node weights
    community_list: a series of communities (dict of frozensets)
    snapshot_node_weights: a series of dicts of node weights
    """
    assert (len(community_list) == len(snapshot_node_weights_list))
    print(f"\nCalculate indexes for project [{project_name}]...")
    start_time = datetime.datetime.now()

    aggregated_split = []
    aggregated_shrink = []
    aggregated_merge = []
    aggregated_expand = []

    split_index_list = []
    shrink_index_list = []
    eta_list = []
    entropy_psi_list = []
    max_entropy_psi_list = []
    max_entropy_psi_list_plus_sigma = []

    merge_index_list = []
    expand_index_list = []
    mu_list = []
    entropy_phi_list = []
    max_entropy_phi_list = []
    max_entropy_phi_list_plus_sigma = []

    for t in range(len(community_list) - 1):
        communities_prior = community_list[t]
        communities_next = community_list[t + 1]
        node_weights_prior = snapshot_node_weights_list[t]
        node_weights_next = snapshot_node_weights_list[t + 1]

        migration_weights_prior, leave_weights_prior, total_weights_prior, \
        migration_weights_next, emerge_weights_next, total_weights_next \
            = __calculate_migration_weights(communities_prior, communities_next, node_weights_prior, node_weights_next)

        split_indexes, shrink_indexes, eta, entropy_psi, max_entropy_psi, max_entropy_plus_sigma_psi \
            = __calculate_split_shrink_indexes_prior(migration_weights_prior, leave_weights_prior, total_weights_prior,
                                                     communities_prior, communities_next)
        split_index_list.append(split_indexes)
        shrink_index_list.append(shrink_indexes)
        eta_list.append(eta)
        entropy_psi_list.append(entropy_psi)
        max_entropy_psi_list.append(max_entropy_psi)
        max_entropy_psi_list_plus_sigma.append(max_entropy_plus_sigma_psi)

        merge_indexes, expand_indexes, mu, entropy_phi, max_entropy_phi, max_entropy_plus_sigma_phi \
            = __calculate_merge_expand_indexes_next(migration_weights_next, emerge_weights_next, total_weights_next,
                                                    communities_next, communities_prior)
        merge_index_list.append(merge_indexes)
        expand_index_list.append(expand_indexes)
        mu_list.append(mu)
        entropy_phi_list.append(entropy_phi)
        max_entropy_phi_list.append(max_entropy_phi)
        max_entropy_phi_list_plus_sigma.append(max_entropy_plus_sigma_phi)

        aggregated_split.append(__aggregate_index(split_indexes, total_weights_prior))
        aggregated_shrink.append(__aggregate_index(shrink_indexes, total_weights_prior))
        aggregated_merge.append(__aggregate_index(merge_indexes, total_weights_next))
        aggregated_expand.append(__aggregate_index(expand_indexes, total_weights_next))

    time_spent = datetime.datetime.now() - start_time
    print(f"Done. Time spent [{time_spent}].")
    # return the results, rest for debugging purposes
    return aggregated_split, aggregated_shrink, aggregated_merge, aggregated_expand, \
           split_index_list, shrink_index_list, eta_list, entropy_psi_list, max_entropy_psi_list, \
           merge_index_list, expand_index_list, mu_list, entropy_phi_list, max_entropy_phi_list, \
           max_entropy_psi_list_plus_sigma, max_entropy_phi_list_plus_sigma


def __cumulate_indexes_over_time(indexes):
    cumulated_indexes = []
    temp_cumulated = 0
    for idx in indexes:
        temp_cumulated = temp_cumulated + idx
        cumulated_indexes.append(temp_cumulated)
    return cumulated_indexes


def cumulate_all_indexes(aggregated_split, aggregated_shrink, aggregated_merge, aggregated_expand):
    cumulated_split = __cumulate_indexes_over_time(aggregated_split)
    cumulated_shrink = __cumulate_indexes_over_time(aggregated_shrink)
    cumulated_merge = __cumulate_indexes_over_time(aggregated_merge)
    cumulated_expand = __cumulate_indexes_over_time(aggregated_expand)
    return cumulated_split, cumulated_shrink, cumulated_merge, cumulated_expand


def diff_indexes(indexes_1, indexes_2):
    assert len(indexes_1) == len(indexes_2)
    diff_idx = []
    for i in range(len(indexes_1)):
        diff_idx.append(indexes_1[i] - indexes_2[i])
    return diff_idx
