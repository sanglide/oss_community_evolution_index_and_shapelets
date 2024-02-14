import math

import global_settings


def sum_confusion_matrix(confusion_matrix_sum, confusion_matrix):
    for i in range(7):
        for j in range(7):
            confusion_matrix_sum[i][j] = confusion_matrix_sum[i][j] + confusion_matrix[i][j]
    return confusion_matrix_sum


def calculate_log_accuracy(confusion_matrix, project_name, result_folder_path):
    """

    Parameters
    ----------
    confusion_matrix

    A 7x7 confusion matrix including undefined

    Returns
    -------

    """

    fo = open(result_folder_path + "concurrent_validity.txt", "a+")
    fo.write(project_name + "\n")

    count_correct = 0
    count_total = 0
    for i in range(7):
        for j in range(7):
            if i == j:
                count_correct = count_correct + confusion_matrix[i][j]
            count_total = count_total + confusion_matrix[i][j]
            print(f"{confusion_matrix[i][j]}\t", end='')
            fo.write(f"{confusion_matrix[i][j]}\t")
        print("")
        fo.write("\n")
    print(f"Accuracy: {count_correct / count_total}")
    fo.write(f"Accuracy: {count_correct / count_total}\n")

    count_correct = 0
    count_total = 0
    for i in range(6):
        for j in range(6):
            if i == j:
                count_correct = count_correct + confusion_matrix[i][j]
            count_total = count_total + confusion_matrix[i][j]
    print(f"Accuracy (w/o undefined): {count_correct / count_total}\n")
    fo.write((f"Accuracy (w/o undefined): {count_correct / count_total}\n\n"))

    fo.flush()
    fo.close()


def get_evolution_patterns_confusion_matrix(match_assign_results):
    """

    Parameters
    ----------
    match_assign_results: evolution patterns obtained after calling append_index_based_results

    Returns
    -------

    confusion matrix

    """

    def get_matrix_index(label):
        if label == 'split':
            return 0
        elif label == 'shrink':
            return 1
        elif label == 'merge':
            return 2
        elif label == 'expand':
            return 3
        elif label == 'emerge':
            return 4
        elif label == 'extinct':
            return 5
        elif label == 'undefined':
            return 6
        else:
            print(f"illegal label {label}")
            assert False
            return -1

    confusion_matrix = [([0] * 7) for i in range(7)]
    for p in range(len(match_assign_results)):
        for i in range(len(match_assign_results[p])):
            prl = match_assign_results[p][i]['prior_label']
            pril = match_assign_results[p][i]['prior_index_label']
            pol = match_assign_results[p][i]['post_label']
            poil = match_assign_results[p][i]['post_index_label']
            confusion_matrix[get_matrix_index(prl)][get_matrix_index(pril)] = confusion_matrix[get_matrix_index(prl)][
                                                                                  get_matrix_index(pril)] + 1
            confusion_matrix[get_matrix_index(pol)][get_matrix_index(poil)] = confusion_matrix[get_matrix_index(pol)][
                                                                                  get_matrix_index(poil)] + 1
    return confusion_matrix


def append_index_based_results(match_assign_results,
                               split_index_list, shrink_index_list, max_entropy_list_plus_sigma_psi,
                               merge_index_list, expand_index_list, max_entropy_list_plus_sigma_phi):
    """

    Parameters
    ----------
    match_assign_results: the results obtained by calling match_and_assign
    split_index_list: list of split indexes for each community in time step from 0 to T-1
    shrink_index_list: list of shrink indexes for each community in time step from 0 to T-1
    merge_index_list: list of merge indexes for each community in time step from 1 to T
    expand_index_list: list of expand indexes for each community in time step from 1 to T

    Returns
    -------

    add field 'prior_index_label', and 'post_index_label' to each community in the result series

    """
    for p in range(len(match_assign_results)):
        # currently step p
        # use split[p] and shrink[p]
        # use merge[p-1] and expand[p-1]
        # i am worried about the orders, hope it's cool
        # we should have chances for 'emerge' and 'extinct' here
        for i in range(len(match_assign_results[p])):
            # match_assign_results[p][i]
            if p < len(match_assign_results) - 1:
                if (math.fabs(split_index_list[p][i]) <= max_entropy_list_plus_sigma_psi[p][i] * 0.05) and (
                        shrink_index_list[p][i] >= max_entropy_list_plus_sigma_psi[p][i] * 0.95):
                    match_assign_results[p][i]['post_index_label'] = 'extinct'
                elif split_index_list[p][i] > shrink_index_list[p][i]:
                    match_assign_results[p][i]['post_index_label'] = 'split'
                elif split_index_list[p][i] < shrink_index_list[p][i]:
                    match_assign_results[p][i]['post_index_label'] = 'shrink'

                else:
                    # match_assign_results[p][i][
                    #     'post_index_label'] = (f'undefined_extinct_{"{:.2f}".format(math.fabs(split_index_list[p][i]) / max_entropy_list_plus_sigma_psi[p][i])}_'
                    #                            f'{"{:.2f}".format( shrink_index_list[p][i]/ float(max_entropy_list_plus_sigma_psi[p][i] ))}')
                    match_assign_results[p][i]['post_index_label'] = 'undefined'
                    # all cases are caused by both zeros
                    # print(f"{split_index_list[p][i]} <==> {shrink_index_list[p][i]}")
            if p > 0:
                if (math.fabs(merge_index_list[p - 1][i]) <= max_entropy_list_plus_sigma_phi[p - 1][i] * 0.05) and (
                        expand_index_list[p - 1][i] >= max_entropy_list_plus_sigma_phi[p - 1][i] * 0.95):
                    match_assign_results[p][i]['prior_index_label'] = 'emerge'
                elif merge_index_list[p - 1][i] > expand_index_list[p - 1][i]:
                    match_assign_results[p][i]['prior_index_label'] = 'merge'
                elif merge_index_list[p - 1][i] < expand_index_list[p - 1][i]:
                    match_assign_results[p][i]['prior_index_label'] = 'expand'
                else:
                    # match_assign_results[p][i][
                    #     'post_index_label'] = (f'undefined_emerge_{"{:.2f}".format(math.fabs(merge_index_list[p - 1][i])/float(max_entropy_list_plus_sigma_phi[p - 1][i]))}_'
                    #                            f'{"{:.2f}".format(expand_index_list[p - 1][i]/float(max_entropy_list_plus_sigma_phi[p - 1][i]))}')
                    match_assign_results[p][i]['prior_index_label'] = 'undefined'
                    # caused by merge_index_list[p - 1][i] == expand_index_list[p - 1][i]
                    # find cases with large community size
    print(f'------------------ match assign results -------------------')
    print(match_assign_results)
    return match_assign_results


def match_and_assign(community_list, snapshot_node_weights_list):
    """

    match and assign community evolution patterns based on the detection results

    Parameters
    ----------
    community_list: a list of list, example [['a', 'b', 'c'], ['d', 'e']]
    snapshot_node_weights_list: a list of [{'user': weight}]

    Returns
    -------

    the result of matching communities and assign evolution patterns following paper:

    * Qiaona Hong, Sunghun Kim, Shing Chi Cheung, and Christian Bird. 2011. Un-derstanding a developer social network and its evolution. In2011 27th IEEEinternational conference on software maintenance (ICSM). IEEE, 323–332.

    """

    match_assign_results = []

    for p in range(len(community_list)):
        match_assign_results.append([])
        # a list for all the communities in time p
        for i in range(len(community_list[p])):
            match_assign_results[p].append(
                {'community': community_list[p][i], 'weight_with_prior': 0, 'weight_with_next': 0, 'post': [],
                 'prior': [], 'post_label': 'undefined', 'post_index_label': 'undefined',
                 'prior_label': 'undefined', 'prior_index_label': 'undefined'})
        # each community is recorded by a dict
        # {'weight_with_prior': 1.0, 'weight_with_next': 1.0,
        # 'post': [{'community': j, 'weight': 1.0}],
        # 'prior': [{'community': j, 'weight': 1.0}],
        # 'post_label': 'split | shrink | extinct | undefined',
        # 'post_index_label': 'split | shrink | extinct | undefined',
        # 'priror_label': 'expand | merge | emerge | undefined',
        # 'priror_index_label': 'expand | merge | emerge | undefined'}


    # scan the communities and find matches
    for p in range(len(community_list) - 1):
        for i in range(len(community_list[p])):
            for j in range(len(community_list[p + 1])):
                comm_prior = community_list[p][i]
                comm_next = community_list[p + 1][j]
                # for all possible pairs of communities
                set_comm_prior = set(comm_prior)
                set_comm_next = set(comm_next)
                # common nodes shared by the two communities
                set_common_nodes = set_comm_prior.intersection(set_comm_next)

                # print(set_comm_prior)
                # print(set_comm_next)
                # print(set_common_nodes)

                # for simplicity, we use the max of a node's weight here
                weight_comm_prior = sum(
                    [max(snapshot_node_weights_list[p].get(u), snapshot_node_weights_list[p + 1].get(u, 0)) for u in
                     set_comm_prior])
                weight_comm_next = sum(
                    [max(snapshot_node_weights_list[p + 1].get(u), snapshot_node_weights_list[p].get(u, 0)) for u in
                     set_comm_next])
                match_assign_results[p][i]['weight_with_next'] = weight_comm_prior
                match_assign_results[p + 1][j]['weight_with_prior'] = weight_comm_next
                # over the size of the smaller community
                smaller_weight = min(weight_comm_prior, weight_comm_next)
                weight_common = \
                    sum([max(snapshot_node_weights_list[p].get(u), snapshot_node_weights_list[p + 1].get(u)) for u in
                         set_common_nodes])
                similarity = weight_common / smaller_weight if smaller_weight > 0 else 0
                similarity = similarity if similarity <= 1 else 1
                similarity = similarity if similarity >= 0 else 0

                if similarity >= global_settings.EVOLUTION_PATTERN_THRESHOLD:
                    # we have a match, i.e., set_comm_prior is a prior community of set_comm_next,
                    # and set_comm_next is a post community of set_comm_prior
                    match_assign_results[p][i]['post'].append(
                        {'community_id': j, 'community': set_comm_next, 'weight': weight_comm_next})
                    match_assign_results[p + 1][j]['prior'].append(
                        {'community_id': i, 'community': set_comm_prior, 'weight': weight_comm_prior})

    # assign labels based on match results
    for p in range(len(community_list)):
        for i in range(len(community_list[p])):
            list_post = match_assign_results[p][i]['post']
            if len(list_post) > 1:
                match_assign_results[p][i]['post_label'] = 'split'
            elif len(list_post) == 1:
                # we modify the rule for shrink in the original paper to match our settings
                if match_assign_results[p][i]['weight_with_next'] > list_post[0]['weight']:
                    match_assign_results[p][i]['post_label'] = 'shrink'
                # elif match_assign_results[p][i]['weight_with_next'] < list_post[0]['weight']:
                #     match_assign_results[p][i]['post_label'] = 'expand'
                else:
                    match_assign_results[p][i]['post_label'] = 'undefined'
            elif len(list_post) == 0:
                match_assign_results[p][i]['post_label'] = 'extinct'
            if p == len(community_list) - 1:
                match_assign_results[p][i]['post_label'] = 'undefined'

            list_prior = match_assign_results[p][i]['prior']
            if len(list_prior) > 1:
                match_assign_results[p][i]['prior_label'] = 'merge'
            elif len(list_prior) == 1:
                if match_assign_results[p][i]['weight_with_prior'] > list_prior[0]['weight']:
                    match_assign_results[p][i]['prior_label'] = 'expand'
                # elif match_assign_results[p][i]['weight_with_prior'] < list_prior[0]['weight']:
                #     match_assign_results[p][i]['prior_label'] = 'shrink'
                else:
                    match_assign_results[p][i]['prior_label'] = 'undefined'
            else:
                match_assign_results[p][i]['prior_label'] = 'emerge'
            if p == 0:
                match_assign_results[p][i]['prior_label'] = 'undefined'
    return match_assign_results
