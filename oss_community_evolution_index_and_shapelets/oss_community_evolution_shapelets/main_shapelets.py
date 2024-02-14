import csv
import datetime
import os
import sys
import traceback
from copy import deepcopy
import scipy.ndimage as filters
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from numpy import average

import classify_supervised_learning
import global_settings
import shapelet_dist_features
import shapelets as sp
import util
import visualize
from classify_simple_dist import ClassifySimpleDist
from shapelets import ShapeletsMV


def load_data(index_productivity_file: str):
    # load data from index_productivity.csv
    # project_name,time,split,shrink,merge,expand,commit_count,commit_count_diff,project_age,issue_count,pr_count,issue_pr_count,member_count

    # data format: [repo_1, repo_2, ...]
    # where repo_1: {name: str, time: list, split: list, shrink: list, merge: list, expand: list, commit_count: list}
    last_repo_name = "DUMMY"
    loaded_res = []
    repo_dict = {}

    with open(index_productivity_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dict_row = dict(row)
            if dict_row['project_name'] != last_repo_name:
                # we have a new repo
                if last_repo_name != 'DUMMY':
                    loaded_res.append(deepcopy(repo_dict))
                last_repo_name = dict_row['project_name']
                repo_dict = {'name': last_repo_name, 'time': [], 'split': [], 'shrink': [], 'merge': [], 'expand': [],
                             'commit_count': []}
            repo_dict['time'].append(dict_row['time'])
            repo_dict['split'].append(float(dict_row['split']))
            repo_dict['shrink'].append(float(dict_row['shrink']))
            repo_dict['merge'].append(float(dict_row['merge']))
            repo_dict['expand'].append(float(dict_row['expand']))
            repo_dict['commit_count'].append(float(dict_row['commit_count']))
        loaded_res.append(deepcopy(repo_dict))
    f.close()
    print(f"Load from {index_productivity_file}, num repos: {len(loaded_res)}")
    return loaded_res


import numpy as np


def moving_average(data, window_size=3):
    """
    Computes the moving average of the given data avoiding NaN at the edges.

    :param data: The input data as a list or numpy array.
    :param window_size: The number of data points to include in each average.
    :return: A numpy array containing the moving averages.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    # Initialize an empty list to store the moving averages
    moving_averages = []

    # Compute the moving average, avoiding the edges where we don't have enough data points
    for i in range(len(data)):
        # Determine the start and end of the window for this iteration
        start_index = max(0, i - window_size + 1)
        end_index = i + 1  # End index is exclusive

        # Extract the window of data for this iteration
        window = data[start_index:end_index]

        # Compute the average if we don't have NaN values in the window
        if not np.isnan(window).any():
            window_average = np.mean(window)
            moving_averages.append(window_average)
        else:
            # If there are NaN values, we can choose to append a NaN or skip it
            moving_averages.append(np.nan)

    return np.array(moving_averages)


def gaussian_smooth_data(sequence, sigma=1):
    """
    Smooths a data sequence using Gaussian smoothing with the specified sigma.

    :param sequence: The data sequence to smooth.
    :type sequence: list of float or int
    :param sigma: The standard deviation of the Gaussian kernel, defaults to 1.
    :type sigma: float, optional
    :return: The smoothed data sequence.
    :rtype: list of float
    """
    if len(sequence) < 2:
        raise ValueError("Sequence must be longer than 1.")

    # smoothed_sequence = filters.gaussian_filter1d(np.array(sequence), sigma)
    smoothed_sequence = filters.gaussian_filter1d(np.array(sequence), sigma)
    return smoothed_sequence.tolist()


def preprocess(loaded_raw_data, label_period_months, forecast_gap_months, data_period_months, window_size,
               evolution_event_selection, is_mining=False):
    # segment data, and generate labels

    # data format: [repo_1, repo_2, ...]
    # where repo_1: {name: str, seg_time: list, seg_split: list, seg_shrink: list, seg_merge: list, seg_expand: list, label: [0, 1]}
    # label 0 for inactive, and 1 for active

    def scale_to_max(data_seq, max_value, ratio):
        scale_seq = []
        for d in data_seq:
            scale_seq.append(ratio * d / max_value if max_value > 0 else 0)
        return scale_seq

    processed_data = []
    X_valid_length_list = []

    if is_mining and global_settings.DO_FIX_MINING_PARAMETERS:
        forecast_gap_months = global_settings.FIXED_MINING_FORECAST_GAP
        data_period_months = global_settings.FIXED_MINING_DATA_PERIOD

    for repo_data in loaded_raw_data:
        # seg_time是不等长的，有的repo没数据，是实际有效数据段。label是成功失败标签，-1代表没初始化
        processed_repo_dict = {'name': repo_data['name'], 'seg_time': [], 'seg_split': [], 'seg_shrink': [],
                               'seg_merge': [], 'seg_expand': [], 'label': -1}
        last_record_time = datetime.datetime.strptime(repo_data['time'][-1], "%Y-%m-%dT%H:%M:%SZ")
        first_record_time = datetime.datetime.strptime(repo_data['time'][0], "%Y-%m-%dT%H:%M:%SZ")
        last_commit_time_if_active = last_record_time - relativedelta(months=label_period_months)

        # 根据commit时间判断项目是active还是inactive
        last_commit_time = None
        for i in range(1, len(repo_data['time']) + 1):
            # average_week_commit = np.mean(repo_data['commit_count'][-i:]) if i <= 48 else np.mean(
            #     repo_data['commit_count'][-i:-i + 48])
            # if average_week_commit > 0.25:
            if repo_data['commit_count'][-i] > 0:
                last_commit_time = datetime.datetime.strptime(repo_data['time'][-i], "%Y-%m-%dT%H:%M:%SZ")
                break

        if (last_commit_time_if_active < last_commit_time) and (not repo_data['name'] in global_settings.failProjL):
            processed_repo_dict['label'] = 1  # active
        else:
            processed_repo_dict['label'] = 0  # inactive

        # 再往前切一段
        # the end of the data period is the last commit time minus the forecast gap
        data_end_time = last_commit_time - relativedelta(months=forecast_gap_months)
        # the start of the data period is the data end time minus the data period length
        data_start_time = data_end_time - relativedelta(months=data_period_months)

        # segment the data to get
        for i in range(len(repo_data['time'])):
            current_time = datetime.datetime.strptime(repo_data['time'][i], "%Y-%m-%dT%H:%M:%SZ")
            if current_time < data_start_time:
                continue
            elif current_time > data_end_time:
                break
            else:
                processed_repo_dict['seg_time'].append(str(repo_data['time'][i]))
                processed_repo_dict['seg_split'].append(float(repo_data['split'][i]))
                processed_repo_dict['seg_shrink'].append(float(repo_data['shrink'][i]))
                processed_repo_dict['seg_merge'].append(float(repo_data['merge'][i]))
                processed_repo_dict['seg_expand'].append(float(repo_data['expand'][i]))
        seq_len = len(processed_repo_dict['seg_time'])
        # 取了sequence length。如果没有效数据，仓库就无效，直接丢掉；否则记录有效数据长度在X_valid_length_list
        if seq_len <= 0 or seq_len < window_size:
            print(
                f"Warning: drop repo {repo_data['name']} for having short sequence length = {seq_len} < window_size = {window_size}")
            continue
        X_valid_length_list.append(seq_len)

        dict_name = ['seg_split', 'seg_shrink', 'seg_merge', 'seg_expand']
        if global_settings.DO_SMOOTH:
            if global_settings.SMOOTH_METHOD==1:
                # smooth data
                for j in dict_name:
                    # print(processed_repo_dict[j])
                    # processed_repo_dict[j] = smooth_data(processed_repo_dict[j], 10)
                    processed_repo_dict[j] = gaussian_smooth_data(processed_repo_dict[j], sigma=global_settings.GAUSS_SIGMA)
                # print(len(processed_repo_dict['seg_split']))
                # print(len(processed_repo_dict['time']))
            elif global_settings.SMOOTH_METHOD==2:
                # smooth data
                for j in dict_name:
                    # print(processed_repo_dict[j])
                    # processed_repo_dict[j] = smooth_data(processed_repo_dict[j], 10)
                    processed_repo_dict[j] = moving_average(processed_repo_dict[j])
                # print(len(processed_repo_dict['seg_split']))
                # print(len(processed_repo_dict['time']))

        # scale the data sequences
        # 把数据做scale，用的是scale_to_max，把值按照max_value等比例缩小

        candidate_max = []
        for it in evolution_event_selection:
            candidate_max.append(max(processed_repo_dict[dict_name[it]]))

        max_value = max(candidate_max)

        if global_settings.DO_GLOBAL_SCALE:
            processed_repo_dict['seg_split'] = scale_to_max(processed_repo_dict['seg_split'], max_value, 10.)
            processed_repo_dict['seg_shrink'] = scale_to_max(processed_repo_dict['seg_shrink'], max_value, 10.)
            processed_repo_dict['seg_merge'] = scale_to_max(processed_repo_dict['seg_merge'], max_value, 10.)
            processed_repo_dict['seg_expand'] = scale_to_max(processed_repo_dict['seg_expand'], max_value, 10.)

        processed_data.append(processed_repo_dict)

        # 注意：这里没有做平滑

    return processed_data, X_valid_length_list


def _extract_X_y(processed_data, n_channels, X_valid_length_list):
    max_len = max(X_valid_length_list)
    # X[channel][repo][data]
    X = [[] for ch in range(n_channels)]
    y = []
    channel_name = ['seg_split', 'seg_shrink', 'seg_merge', 'seg_expand']

    # prepare data and align the length of all sequences
    for repo in processed_data:
        y.append(repo['label'])
        nan_list = [np.nan for n in range(max_len - len(repo['seg_time']))]
        for ch in range(n_channels):
            temp_list = repo[channel_name[ch]].copy()
            # 使用nan填充，使得最后数据是矩阵
            X[ch].append(temp_list + nan_list.copy())
    return X, y


def mine_shapelets(processed_data, n_shapelets, window_size, window_step, n_channels, X_valid_length_list, n_jobs,
                   evolution_event_selection, remove_overlap=False):
    # mine shapelets from the training set
    # st = ShapeletTransform_mv(n_shapelets=n_shapelets, window_sizes=window_sizes_week)
    # X_new = st.fit_transform(x, y, 4)
    # 格式转换，变成矩阵和一维数据
    X, y = _extract_X_y(processed_data, n_channels, X_valid_length_list)

    # mine shapelets
    sh = ShapeletsMV()
    X_array = np.nan_to_num(np.array(X))
    shapelets, scores, info_gains, start_idx, end_idx, series_idx,log_list = sh.fit_all(X_array, np.array(y), n_shapelets
                                                                               , window_size, window_step,
                                                                               n_channels,
                                                                               np.array(X_valid_length_list), n_jobs,
                                                                               evolution_event_selection,
                                                                               remove_overlap)
    return shapelets, scores, info_gains, start_idx, end_idx, series_idx,log_list


def execute_mine(label_period_months, forecast_gap_months, data_period_months,
                 n_shapelets, window_size, window_step, n_channels,
                 n_jobs, data_path, result_folder_path, evolution_event_selection):
    shapelet_file_name = util.get_shapelet_file_name(label_period_months, forecast_gap_months, data_period_months,
                                                     n_shapelets, window_size, window_step, n_channels, data_path,
                                                     evolution_event_selection)

    print("\n\n=============================================")
    print(f"************ Start mine job {shapelet_file_name}")

    # load data
    loaded_res = load_data(data_path)

    # preprocess data
    processed_data, X_valid_length_list = preprocess(loaded_res, label_period_months, forecast_gap_months,
                                                     data_period_months, window_size, evolution_event_selection,
                                                     is_mining=True)

    repo_name_list = []
    repo_label_list = []
    for repo in processed_data:
        repo_name_list.append(repo['name'])
        repo_label_list.append(repo["label"])

    # 开始挖掘
    assert len(processed_data) == len(X_valid_length_list)
    print(f"************ Repo count after preprocess {len(processed_data)}")

    # mine shapelets
    shapelets, scores, info_gains, start_idx, end_idx, series_idx,log_list = mine_shapelets(processed_data, n_shapelets,
                                                                                   window_size, window_step,
                                                                                   n_channels, X_valid_length_list,
                                                                                   n_jobs, evolution_event_selection)

    result_file = f"{result_folder_path}{shapelet_file_name}"
    sp.log_mined_shapelets(shapelets, scores, info_gains, start_idx, end_idx, series_idx, repo_name_list,
                           repo_label_list, n_shapelets, window_size, window_step, n_channels, label_period_months,
                           forecast_gap_months, data_period_months, data_path, result_file, evolution_event_selection)

    print(f"************ Finish mine job {shapelet_file_name}")
    print("=============================================\n\n")
    return log_list


def execute_visualize(label_period_months, forecast_gap_months, data_period_months,
                      n_shapelets, window_size, window_step, n_channels,
                      data_path, result_folder_path,evolution_event_selection):
    shapelet_filename = util.get_shapelet_file_name(label_period_months, forecast_gap_months, data_period_months,
                                                    n_shapelets, window_size, window_step, n_channels, data_path,evolution_event_selection)
    # load shapelets from result json file
    shapelet_file_path = f"{result_folder_path}{shapelet_filename}"
    loaded_shape_dict = sp.load_mined_shapelets(shapelet_file_path)
    plot_path = f"{result_folder_path}plot/"
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    visualize.plot_n_shapelets(loaded_shape_dict, plot_path)


def execute_classify(label_period_months, forecast_gap_months, data_period_months,
                     n_shapelets, window_size, window_step, n_channels, data_path, shapelet_folder_path,
                     result_folder_path, train_data_path,log_list):
    shapelet_file_name = util.get_shapelet_file_name(label_period_months, forecast_gap_months, data_period_months,
                                                     n_shapelets, window_size, window_step, n_channels, train_data_path)

    print("=============================================")
    # print(f"************ Start classifying job {shapelet_file_name}")
    print(f"************ Start classifying job")
    loaded_res = load_data(data_path)
    processed_data, X_valid_length_list = preprocess(loaded_res, label_period_months, forecast_gap_months,
                                                     data_period_months, window_size)
    X, y = _extract_X_y(processed_data, n_channels, X_valid_length_list)

    # load shapelet mining results
    # shapelet_file_path = f"{shapelet_folder_path}{shapelet_file_name}"

    cs = ClassifySimpleDist()
    cs.load(f"{result_folder_path}{shapelet_file_name}.model")
    # 2. 调用classify方法
    threshold_list = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    for th in threshold_list:
        y_pred = cs.classify(np.array(X), np.array(X_valid_length_list), np.array(y), th)
        # 3. 日志打印
        cs.report(y, y_pred, label_period_months, forecast_gap_months, data_period_months,
                  result_folder_path,log_list)


# def execute_classify_multi_sizes(label_period_months, forecast_gap_months, data_period_months,
#                                  n_shapelets, window_size_list, window_step, n_channels, data_path,
#                                  shapelet_folder_path, result_folder_path, train_data_path):
#     print("=============================================")
#     print(f"************ Start classifying job")
#     loaded_res = load_data(data_path)
#     last_cs = None
#     threshold_list = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
#     for th in threshold_list:
#         y_all = None
#         y_pred_all = None
#         window_size_count = 0
#         for window_size in window_size_list:
#             shapelet_file_name = util.get_shapelet_file_name(label_period_months, forecast_gap_months,
#                                                              data_period_months,
#                                                              n_shapelets, window_size, window_step, n_channels,
#                                                              train_data_path)
#             processed_data, X_valid_length_list = preprocess(loaded_res, label_period_months, forecast_gap_months,
#                                                              data_period_months, window_size)
#             X, y = _extract_X_y(processed_data, n_channels, X_valid_length_list)
#             if y_all is None:
#                 y_all = deepcopy(y)
#             else:
#                 if len(y) != len(y_all):
#                     print(
#                         f"warning: len(y) != len(y_all), stop at wsz count {window_size_count} for {shapelet_file_name}")
#                     continue
#             window_size_count += 1
#             cs = ClassifySimpleDist()
#             cs.load(f"{result_folder_path}{shapelet_file_name}.model")
#             y_pred = cs.classify(np.array(X), np.array(X_valid_length_list), np.array(y), th)
#             if y_pred_all is None:
#                 y_pred_all = deepcopy(y_pred)
#             else:
#                 for i in range(len(y_pred_all)):
#                     y_pred_all[i] += y_pred[i]
#             last_cs = cs
#         # majority voting
#         for i in range(len(y_pred_all)):
#             if y_pred_all[i] >= window_size_count / 2.:
#                 y_pred_all[i] = 1
#             else:
#                 y_pred_all[i] = 0
#         # 3. 日志打印
#         last_cs.report(y_all, y_pred_all, label_period_months, forecast_gap_months, data_period_months,
#                        result_folder_path, report_file_name="prediction_report_multi_sizes.csv")


#
def execute_train(label_period_months, forecast_gap_months, data_period_months,
                  n_shapelets, window_size, window_step, n_channels, data_path, shapelet_folder_path,
                  result_folder_path, evolution_event_selection, do_visualize=False):
    shapelet_file_name = util.get_shapelet_file_name(label_period_months, forecast_gap_months, data_period_months,
                                                     n_shapelets, window_size, window_step, n_channels, data_path,
                                                     evolution_event_selection)

    print("=============================================")
    print(f"************ Start training job {shapelet_file_name}")
    # load training data
    loaded_res = load_data(data_path)
    # load preprocess data
    processed_data, X_valid_length_list = preprocess(loaded_res, label_period_months, forecast_gap_months,
                                                     data_period_months, window_size, evolution_event_selection)

    X, y = _extract_X_y(processed_data, n_channels, X_valid_length_list)

    # load shapelet mining results
    shapelet_file_path = f"{shapelet_folder_path}{shapelet_file_name}"

    # train shapelets-based classification models using training data in the data_path
    csd = ClassifySimpleDist()
    csd.set_init_parameters(shapelet_file_path, result_folder_path)
    csd.train(np.array(X), np.array(X_valid_length_list), np.array(y))
    if global_settings.DO_VIS or do_visualize:
        csd.log_label()
    csd.save(f"{result_folder_path}{shapelet_file_name}.model")


# def script_mine_shapelets(time_of_execution):
#     # 配置文件夹
#     result_root_dir = util.get_result_root_dir()
#     data_dir = util.get_data_dir()
#     result_folder_path = f"{result_root_dir}{time_of_execution}/"
#     os.makedirs(result_folder_path)
#
#     n_shapelets = 20
#     window_step = 1
#     n_channels = 4
#     n_jobs = 20
#
#     list_label_period_months = [12]
#     list_forecast_gap_months = [3]
#     list_data_period_months = [24]
#     # 以data point为单位
#     list_window_size = [3]  # , 8, 12, 16, 20, 24]  # , 28, 32, 36, 40, 44, 48]
#     # 记录的截止时间，单位是月，判断失败
#     # list_label_period_months = [6, 12]
#     # list_forecast_gap_months = [3, 6, 9, 12]
#     # list_data_period_months = [6, 12, 18, 24, 30, 36]
#     # # 以data point为单位
#     # list_window_size = [3, 4, 5, 6]  # , 8, 12, 16, 20, 24]  # , 28, 32, 36, 40, 44, 48]
#     # list_data_path = [f"{data_dir}index_productivity_32.csv", f"{data_dir}index_productivity_696_old.csv"]
#     list_data_path = [f"{data_dir}index_productivity_32.csv"]
#     # list_data_path = [f"{data_dir}index_productivity_696_old.csv"]
#
#     count = 1
#     total_count = len(list_label_period_months) * len(list_forecast_gap_months) * len(list_data_period_months) * len(
#         list_window_size) * len(list_data_path)
#
#     # 五层for循环取所有参数组合
#     for label_period_months in list_label_period_months:
#         for forecast_gap_months in list_forecast_gap_months:
#             for data_period_months in list_data_period_months:
#                 for window_size in list_window_size:  # week / data points
#                     for data_path in list_data_path:
#                         # 根据kill文档存不存在决定是否中途退出循环
#                         util.test_kill_and_exit()
#                         # 格式化生成文件名
#                         shapelet_filename = util.get_shapelet_file_name(label_period_months, forecast_gap_months,
#                                                                         data_period_months,
#                                                                         n_shapelets, window_size, window_step,
#                                                                         n_channels, data_path)
#                         print(f"## mine {count}/{total_count} {shapelet_filename}")
#
#                         # 执行一次挖掘
#                         execute_mine(label_period_months, forecast_gap_months, data_period_months,
#                                      n_shapelets, window_size, window_step, n_channels,
#                                      n_jobs, data_path, result_folder_path)
#
#                         print(f"## visualize {count}/{total_count} {shapelet_filename}")
#                         execute_visualize(label_period_months, forecast_gap_months, data_period_months,
#                                           n_shapelets, window_size, window_step, n_channels,
#                                           data_path, result_folder_path)
#                         count = count + 1


def script_train_shapelets(time_of_execution):
    result_root_dir = util.get_result_root_dir()
    data_dir = util.get_data_dir()
    result_folder_path = f"{result_root_dir}{time_of_execution}/"

    n_shapelets = 20
    window_step = 1
    n_channels = 4

    list_label_period_months = [12]
    list_forecast_gap_months = [3]
    list_data_period_months = [24]
    # 以data point为单位
    list_window_size = [3]  # , 8, 12, 16, 20, 24]   # , 28, 32, 36, 40, 44, 48]
    # 记录的截止时间，单位是月，判断失败
    # list_label_period_months = [6, 12]
    # list_forecast_gap_months = [3, 6, 9, 12]
    # list_data_period_months = [6, 12, 18, 24, 30, 36]
    # # 以data point为单位
    # list_window_size = [3, 4, 5, 6]  # , 8, 12, 16, 20, 24]  # , 28, 32, 36, 40, 44, 48]
    list_data_path = [f"{data_dir}index_productivity_32.csv"]

    count = 1
    total_count = len(list_label_period_months) * len(list_forecast_gap_months) * len(list_data_period_months) * len(
        list_window_size) * len(list_data_path)

    for label_period_months in list_label_period_months:
        for forecast_gap_months in list_forecast_gap_months:
            for data_period_months in list_data_period_months:
                for window_size in list_window_size:  # week / data points
                    for data_path in list_data_path:
                        util.test_kill_and_exit()
                        shapelet_filename = util.get_shapelet_file_name(label_period_months, forecast_gap_months,
                                                                        data_period_months,
                                                                        n_shapelets, window_size, window_step,
                                                                        n_channels, data_path)
                        print(f"## train {count}/{total_count} {shapelet_filename}")
                        execute_train(label_period_months, forecast_gap_months, data_period_months,
                                      n_shapelets, window_size, window_step, n_channels, data_path, result_folder_path,
                                      result_folder_path)
                        # execute_visualize(label_period_months, forecast_gap_months, data_period_months,
                        #                   n_shapelets, window_size, window_step, n_channels,
                        #                   data_path, result_folder_path)
                        count = count + 1

#
# def script_classify_shapelets(time_of_execution):
#     result_root_dir = util.get_result_root_dir()
#     data_dir = util.get_data_dir()
#     result_folder_path = f"{result_root_dir}{time_of_execution}/"
#
#     n_shapelets = 20
#     window_step = 1
#     n_channels = 4
#
#     list_label_period_months = [12]
#     list_forecast_gap_months = [3, 6, 9, 12, 15, 18, 21, 24]
#     list_data_period_months = [3, 6, 9, 12, 15, 18, 21, 24]
#     # 以data point为单位
#     list_window_size = [3]  # , 8, 12, 16, 20, 24]   # , 28, 32, 36, 40, 44, 48]
#     # 记录的截止时间，单位是月，判断失败
#     # list_label_period_months = [6, 12]
#     # list_forecast_gap_months = [3, 6, 9, 12]
#     # list_data_period_months = [6, 12, 18, 24, 30, 36]
#     # # 以data point为单位
#     # list_window_size = [3, 4, 5, 6]  # , 8, 12, 16, 20, 24]  # , 28, 32, 36, 40, 44, 48]
#     list_data_path = [f"{data_dir}index_productivity_696.csv"]
#
#     count = 1
#     total_count = len(list_label_period_months) * len(list_forecast_gap_months) * len(list_data_period_months) * len(
#         list_window_size) * len(list_data_path)
#
#     for label_period_months in list_label_period_months:
#         for forecast_gap_months in list_forecast_gap_months:
#             for data_period_months in list_data_period_months:
#                 for window_size in list_window_size:  # week / data points
#                     for data_path in list_data_path:
#                         util.test_kill_and_exit()
#                         shapelet_filename = util.get_shapelet_file_name(label_period_months, forecast_gap_months,
#                                                                         data_period_months,
#                                                                         n_shapelets, window_size, window_step,
#                                                                         n_channels, data_path)
#                         execute_classify(label_period_months, forecast_gap_months, data_period_months,
#                                          n_shapelets, window_size, window_step, n_channels, data_path,
#                                          result_folder_path, result_folder_path)
#                         # execute_visualize(label_period_months, forecast_gap_months, data_period_months,
#                         #                   n_shapelets, window_size, window_step, n_channels,
#                         #                   data_path, result_folder_path)
#                         count = count + 1


# def debug_mine_shapelets():
#     data_dir = util.get_data_dir()
#     result_dir = util.get_result_root_dir()
#     execute_mine(label_period_months=6, forecast_gap_months=12, data_period_months=6,
#                  n_shapelets=10, window_size=16, window_step=1, n_channels=4,
#                  n_jobs=1, data_path=f"{data_dir}index_productivity_696.csv", result_folder_path=result_dir)


def script_all(time_of_execution):
    result_root_dir = util.get_result_root_dir()
    data_dir = util.get_data_dir()
    result_folder_path = f"{result_root_dir}{time_of_execution}/"
    os.makedirs(result_folder_path)

    n_shapelets = global_settings.N_SHAPELETS
    window_step = global_settings.WINDOW_STEP
    n_channels = global_settings.N_CHANNELS
    n_jobs = global_settings.N_JOBS

    list_label_period_months = global_settings.LIST_LABEL_PERIOD_MONTHS
    list_forecast_gap_months = global_settings.LIST_FORECAST_GAP_MONTHS
    list_data_period_months = global_settings.LIST_DATA_PERIOD_MONTHS
    # 以data point为单位
    list_window_size = global_settings.LIST_WINDOW_SIZE  # , 8, 12, 16, 20, 24]   # , 28, 32, 36, 40, 44, 48]
    list_evolution_event_selection = global_settings.EVOLUTION_EVENT_COMBINATIONS
    # 记录的截止时间，单位是月，判断失败
    # list_label_period_months = [6, 12]
    # list_forecast_gap_months = [3, 6, 9, 12]
    # list_data_period_months = [6, 12, 18, 24, 30, 36]
    # # 以data point为单位
    # list_window_size = [3, 4, 5, 6]  # , 8, 12, 16, 20, 24]  # , 28, 32, 36, 40, 44, 48]

    train_data_path = f"{data_dir}index_productivity_32.csv"
    test_data_path = f"{data_dir}index_productivity_696.csv"
    count = 1
    total_count = len(list_label_period_months) * len(list_forecast_gap_months) * len(list_data_period_months) * len(
        list_window_size) * len(list_evolution_event_selection)

    for label_period_months in list_label_period_months:
        for forecast_gap_months in list_forecast_gap_months:
            for data_period_months in list_data_period_months:
                for window_size in list_window_size:  # week / data points
                    for evolution_event_selection in list_evolution_event_selection:

                        util.test_kill_and_exit()

                        # 格式化生成文件名
                        shapelet_filename = util.get_shapelet_file_name(label_period_months, forecast_gap_months,
                                                                        data_period_months, n_shapelets, window_size,
                                                                        window_step,
                                                                        n_channels, train_data_path,
                                                                        evolution_event_selection)
                        print(f"## mine {count}/{total_count} {shapelet_filename}")

                        # 执行一次挖掘
                        log_list=execute_mine(label_period_months, forecast_gap_months, data_period_months,
                                     n_shapelets, window_size, window_step, n_channels,
                                     n_jobs, train_data_path, result_folder_path, evolution_event_selection)
                        log_list[0]=average(np.array(log_list[0]))
                        # 假设变量为data，文件名为filename
                        with open(f'../000_shapelet_results/result/{time_of_execution}/f_{forecast_gap_months}_d_{data_period_months}_w_{window_size}_info.txt', 'w') as file:
                            file.write(str(log_list))

                        if global_settings.DO_VIS:
                            print(f"## visualize {count}/{total_count} {shapelet_filename}")
                            execute_visualize(label_period_months, forecast_gap_months, data_period_months,
                                              n_shapelets, window_size, window_step, n_channels,
                                              train_data_path, result_folder_path,evolution_event_selection)

                        print(f"## train {count}/{total_count} {shapelet_filename}")

                        execute_train(label_period_months, forecast_gap_months, data_period_months,
                                      n_shapelets, window_size, window_step, n_channels, train_data_path,
                                      result_folder_path, result_folder_path, evolution_event_selection)

                        # print(f"## classify {count}/{total_count} {shapelet_filename}")
                        #
                        # execute_classify(label_period_months, forecast_gap_months, data_period_months,
                        #                  n_shapelets, window_size, window_step, n_channels, test_data_path,
                        #                  result_folder_path, result_folder_path, train_data_path)
                        count = count + 1


def script_visualize(time_of_execution):
    result_root_dir = util.get_result_root_dir()
    data_dir = util.get_data_dir()
    result_folder_path = f"{result_root_dir}{time_of_execution}/"
    assert os.path.exists(result_folder_path)

    n_shapelets = global_settings.N_SHAPELETS
    window_step = global_settings.WINDOW_STEP
    n_channels = global_settings.N_CHANNELS
    n_jobs = global_settings.N_JOBS

    list_label_period_months = global_settings.LIST_LABEL_PERIOD_MONTHS
    list_forecast_gap_months = global_settings.LIST_FORECAST_GAP_MONTHS
    list_data_period_months = global_settings.LIST_DATA_PERIOD_MONTHS
    # 以data point为单位
    list_window_size = global_settings.LIST_WINDOW_SIZE  # , 8, 12, 16, 20, 24]   # , 28, 32, 36, 40, 44, 48]
    # 记录的截止时间，单位是月，判断失败
    # list_label_period_months = [6, 12]
    # list_forecast_gap_months = [3, 6, 9, 12]
    # list_data_period_months = [6, 12, 18, 24, 30, 36]
    # # 以data point为单位
    # list_window_size = [3, 4, 5, 6]  # , 8, 12, 16, 20, 24]  # , 28, 32, 36, 40, 44, 48]

    train_data_path = f"{data_dir}index_productivity_32.csv"
    test_data_path = f"{data_dir}index_productivity_696.csv"
    count = 1
    total_count = len(list_label_period_months) * len(list_forecast_gap_months) * len(list_data_period_months) * len(
        list_window_size)

    for label_period_months in list_label_period_months:
        for forecast_gap_months in list_forecast_gap_months:
            for data_period_months in list_data_period_months:
                for window_size in list_window_size:  # week / data points

                    util.test_kill_and_exit()

                    # 格式化生成文件名
                    shapelet_filename = util.get_shapelet_file_name(label_period_months, forecast_gap_months,
                                                                    data_period_months,
                                                                    n_shapelets, window_size, window_step,
                                                                    n_channels, train_data_path)

                    print(f"## visualize {count}/{total_count} {shapelet_filename}")
                    execute_visualize(label_period_months, forecast_gap_months, data_period_months,
                                      n_shapelets, window_size, window_step, n_channels,
                                      train_data_path, result_folder_path)

                    execute_train(label_period_months, forecast_gap_months, data_period_months,
                                  n_shapelets, window_size, window_step, n_channels, train_data_path,
                                  result_folder_path, result_folder_path, do_visualize=True)

                    count = count + 1


# def script_classify_multi_sizes(time_of_execution):
#     result_root_dir = util.get_result_root_dir()
#     data_dir = util.get_data_dir()
#     result_folder_path = f"{result_root_dir}{time_of_execution}/"
#     if not os.path.exists(result_folder_path):
#         os.makedirs(result_folder_path)
#
#     n_shapelets = 40
#     window_step = 1
#     n_channels = 4
#     n_jobs = 20
#
#     list_label_period_months = [12]  # [3, 6, 9, 12]
#     list_forecast_gap_months = [3, 6, 9, 12, 15, 18, 21, 24]
#     list_data_period_months = [3, 6, 9, 12, 15, 18, 21, 24]
#     # 以data point为单位
#     list_window_size = [3, 4, 5, 6, 7]  # , 8, 12, 16, 20, 24]   # , 28, 32, 36, 40, 44, 48]
#     # 记录的截止时间，单位是月，判断失败
#     # list_label_period_months = [6, 12]
#     # list_forecast_gap_months = [3, 6, 9, 12]
#     # list_data_period_months = [6, 12, 18, 24, 30, 36]
#     # # 以data point为单位
#     # list_window_size = [3, 4, 5, 6]  # , 8, 12, 16, 20, 24]  # , 28, 32, 36, 40, 44, 48]
#
#     train_data_path = f"{data_dir}index_productivity_32.csv"
#     test_data_path = f"{data_dir}index_productivity_696_old.csv"
#     count = 1
#
#     for label_period_months in list_label_period_months:
#         for forecast_gap_months in list_forecast_gap_months:
#             for data_period_months in list_data_period_months:
#                 util.test_kill_and_exit()
#
#                 execute_classify_multi_sizes(label_period_months, forecast_gap_months, data_period_months,
#                                              n_shapelets, list_window_size, window_step, n_channels, test_data_path,
#                                              result_folder_path, result_folder_path, train_data_path)
#                 count = count + 1


def script_shapelet_features(time_of_execution):
    result_root_dir = util.get_result_root_dir()
    data_dir = util.get_data_dir()
    result_folder_path = f"{result_root_dir}{time_of_execution}/"
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)

    feature_folder_path = f"{result_folder_path}features/"
    if not os.path.exists(feature_folder_path):
        os.makedirs(feature_folder_path)

    n_shapelets = global_settings.N_SHAPELETS
    window_step = global_settings.WINDOW_STEP
    n_channels = global_settings.N_CHANNELS
    n_jobs = global_settings.N_JOBS

    list_label_period_months = global_settings.LIST_LABEL_PERIOD_MONTHS
    list_forecast_gap_months = global_settings.LIST_FORECAST_GAP_MONTHS
    list_data_period_months = global_settings.LIST_DATA_PERIOD_MONTHS
    # 以data point为单位
    list_window_size = global_settings.LIST_WINDOW_SIZE  # , 8, 12, 16, 20, 24]   # , 28, 32, 36, 40, 44, 48]
    list_evolution_event_selection = global_settings.EVOLUTION_EVENT_COMBINATIONS

    # 记录的截止时间，单位是月，判断失败
    # list_label_period_months = [6, 12]
    # list_forecast_gap_months = [3, 6, 9, 12]
    # list_data_period_months = [6, 12, 18, 24, 30, 36]
    # # 以data point为单位
    # list_window_size = [3, 4, 5, 6]  # , 8, 12, 16, 20, 24]  # , 28, 32, 36, 40, 44, 48]

    train_data_path = f"{data_dir}index_productivity_32.csv"
    test_data_path = f"{data_dir}index_productivity_696.csv"
    count = 1
    total_count = len(list_label_period_months) * len(list_forecast_gap_months) * len(list_data_period_months) * len(
        list_window_size) * len(list_evolution_event_selection)

    for label_period_months in list_label_period_months:
        for forecast_gap_months in list_forecast_gap_months:
            for data_period_months in list_data_period_months:
                for window_size in list_window_size:  # week / data points
                    for evolution_event_selection in list_evolution_event_selection:

                        util.test_kill_and_exit()

                        shapelet_file_name = util.get_shapelet_file_name(label_period_months, forecast_gap_months,
                                                                         data_period_months, n_shapelets, window_size,
                                                                         window_step, n_channels, train_data_path,
                                                                         evolution_event_selection)

                        # print(f"************ Start classifying job {shapelet_file_name}")
                        print(f"************ Feature Extraction {count}/{total_count} {shapelet_file_name}")
                        loaded_res = load_data(train_data_path)

                        processed_data, X_valid_length_list = preprocess(loaded_res, label_period_months,
                                                                         forecast_gap_months, data_period_months,
                                                                         window_size, evolution_event_selection)
                        X, y = _extract_X_y(processed_data, n_channels, X_valid_length_list)

                        repo_names = []
                        for repo in processed_data:
                            repo_names.append(repo['name'])

                        model_filepath = f"{result_folder_path}{shapelet_file_name}.model"
                        output_file = f"{feature_folder_path}{shapelet_file_name}_32.csv"

                        shapelet_dist_features.generate_features_and_label(repo_names, np.array(X),
                                                                           np.array(X_valid_length_list),
                                                                           np.array(y), model_filepath, output_file,
                                                                           evolution_event_selection)

                        # ================================================

                        loaded_res = load_data(test_data_path)
                        processed_data, X_valid_length_list = preprocess(loaded_res, label_period_months,
                                                                         forecast_gap_months,
                                                                         data_period_months, window_size,
                                                                         evolution_event_selection)
                        X, y = _extract_X_y(processed_data, n_channels, X_valid_length_list)

                        repo_names = []
                        for repo in processed_data:
                            repo_names.append(repo['name'])

                        model_filepath = f"{result_folder_path}{shapelet_file_name}.model"
                        output_file = f"{feature_folder_path}{shapelet_file_name}_696.csv"

                        shapelet_dist_features.generate_features_and_label(repo_names, np.array(X),
                                                                           np.array(X_valid_length_list),
                                                                           np.array(y), model_filepath, output_file,
                                                                           evolution_event_selection)
                        count = count + 1


if __name__ == "__main__":
    time_of_start = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%SZ')
    # try:
    time_of_execution = time_of_start
    # time_of_execution = "2023-03-27T22-22-43Z"
    # time_of_execution = "2023-06-07T18-32-07Z"
    # time_of_execution = "2023-06-13T16-44-19Z"
    # time_of_execution = "2023-06-14T08-37-16Z"

    script_all(time_of_execution)
    script_shapelet_features(time_of_execution)
    shapelet_dist_features.script_merge_feature_files(time_of_execution)
    classify_supervised_learning.script_classification_ml_multi_sizes(time_of_execution)
    if global_settings.DO_VIS:
        script_visualize(time_of_execution)

    # == no use
    # classify_supervised_learning.script_classification_ml(time_of_execution)
    # time_of_execution = "2023-03-13T15-10-01Z"
    # script_classify_multi_sizes(time_of_execution)
    # debug_mine_shapelets()
    # script_mine_shapelets(time_of_execution)
    # script_train_shapelets(time_of_execution)
    # time_of_execution = "2023-03-13T09-12-23Z"
    # script_classify_shapelets(time_of_execution)
    # execute_train(label_period_months, forecast_gap_months, data_period_months,
    #               n_shapelets, window_size, window_step, n_channels, data_path, shapelet_folder_path,
    #               result_folder_path,shapelets_plot_output_root_dir)
    # script_visualize_shapelets("2023-01-31T13-01-54Z")

    f = open(f"../terminated_{time_of_start}", "w")
    f.write(str(datetime.datetime.now()))
    f.close()
    # except Exception as e:
        # print(e.args)
        # print(str(e))
        # print(repr(e))
        # f = open(f"../terminated_error_{time_of_start}", "w")
        # f.write(str(datetime.datetime.now()) + "\n")
        # f.write(str(e.args) + "\n")
        # f.write(str(e) + "\n")
        # f.write(str(repr(e)) + "\n")
        # f.write(str(sys.exc_info()))
        # f.write(traceback.format_exc())
        # f.close()
