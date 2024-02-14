import copy

import pandas as pd

import global_settings
import util

from classify_simple_dist import ClassifySimpleDist
import os


def generate_features_and_label(repo_names, X, X_valid_length_list, y, model_filepath, output_file,
                                evolution_event_selection):
    cs = ClassifySimpleDist()
    cs.load(model_filepath)
    dist_matrix = cs.get_distance_matrix(X, X_valid_length_list, evolution_event_selection)
    head_str = "repo,label"
    for i in range(len(dist_matrix[0])):
        head_str = head_str + f",dist_{i}_{cs.get_window_size()}_{cs.get_shapelet_label(i)}"
    with open(output_file, "w") as f:
        f.write(head_str + "\n")
        for idx_seq in range(len(dist_matrix)):
            f.write(f"{repo_names[idx_seq]},{str(y[idx_seq])}")
            for idx_shape in range(len(dist_matrix[0])):
                f.write(f",{dist_matrix[idx_seq][idx_shape]}")
            f.write('\n')


def _line_count(filepath):
    with open(filepath, "r") as f:
        line_count = len(f.readlines())
    return line_count


def merge_feature_files(filepath_list, output_file):
    merged_data = None
    for idx, filepath in enumerate(filepath_list):
        data = pd.read_csv(filepath)
        if merged_data is None:
            merged_data = data
        else:
            merged_data = pd.merge(merged_data, data, how='inner', on=['repo', 'label'])
    merged_data.to_csv(output_file, index=False)

    # if global_settings.DO_FEATURE_ENGINEERING:
    #     heads = [col for col in merged_data.columns]
    #     if 'Unnamed: 0' in heads:
    #         heads.remove('Unnamed: 0')
    #     merged_data = merged_data[heads]
    #     labels = ['repo', 'label']
    #     heads_0 = []
    #     heads_1 = []
    #     for h in heads:
    #         if h == 'repo' or h == 'label':
    #             continue
    #         if h[-1] == '0':
    #             heads_0.append(h)
    #         elif h[-1] == '1':
    #             heads_1.append(h)
    #         else:
    #             print(f"Cannot recognize head {h}")
    #             assert False
    #     inactive_shapelet_dist = merged_data[heads_0]
    #     active_shapelet_dist = merged_data[heads_1]
    #
    #     # Sort each row in the inactive_shapelet_dist and active_shapelet_dist
    #     inactive_shapelet_dist = inactive_shapelet_dist.apply(sorted, axis=1)
    #     active_shapelet_dist = active_shapelet_dist.apply(sorted, axis=1)
    #
    #     # write each row of merged_data[labels], inactive_shapelet_dist, and active_shapelet_dist to output_file
    #     with open(output_file, 'w') as f:
    #         f.write(f",repo,label")
    #         for i in range(len(heads_0)):
    #             f.write(f",inactive_{i}")
    #         for i in range(len(heads_1)):
    #             f.write(f",active_{i}")
    #         f.write('\n')
    #         for index, row in merged_data.iterrows():
    #             label_row = row[labels]
    #             inactive_row = inactive_shapelet_dist.iloc[index]
    #             active_row = active_shapelet_dist.iloc[index]
    #             f.write(str(index) + ',' + ','.join(map(str, label_row)) + ',' + ','.join(
    #                 map(str, inactive_row)) + ',' + ','.join(map(str, active_row)) + '\n')
    #
    # else:
    #     merged_data.to_csv(output_file, index=False)


def reorder_feature_files(input_file, output_file):
    data = pd.read_csv(input_file)

    # rows = []
    # y = []
    # for idx, filepath in enumerate(filepath_list):
    #     data = pandas.read_csv(filepath)
    #     if idx != 0:
    #         y_tmp = data['label']
    #         assert len(y) == len(y_tmp)
    #         for i in range(len(y_tmp)):
    #             assert y[i] == y_tmp[i]
    #     else:
    #         y = data['label']
    #     heads = [col for col in data.columns]
    #     heads.remove('label')
    #     data = data[heads]
    #     for i in range(len(y)):  # remove headline
    #         row_value_tmp = [val for val in data.iloc[i]]
    #         if idx == 0:
    #             rows.append(copy.deepcopy(row_value_tmp))
    #         else:
    #             rows[i].extend(row_value_tmp)
    #
    # head_str = "label"
    # for i in range(len(rows[0])):
    #     head_str = head_str + f",dist{i}"
    # with open(output_file, "w") as f:
    #     f.write(head_str + "\n")
    #     for idx_seq in range(len(rows)):
    #         f.write(str(y[idx_seq]))
    #         for idx_shape in range(len(rows[0])):
    #             f.write(f",{rows[idx_seq][idx_shape]}")
    #         f.write('\n')


def script_merge_feature_files(time_of_execution):
    result_root_dir = util.get_result_root_dir()
    data_dir = util.get_data_dir()
    result_folder_path = f"{result_root_dir}{time_of_execution}/"
    assert os.path.exists(result_folder_path)

    feature_folder_path = f"{result_folder_path}features/"
    assert os.path.exists(feature_folder_path)

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

    list_data_path = [f"_32.csv", f"_696.csv"]
    train_data_path = f"{data_dir}index_productivity_32.csv"
    count = 1
    total_count = len(list_label_period_months) * len(list_forecast_gap_months) * len(list_data_period_months) * len(
        list_window_size)*len(list_evolution_event_selection)

    for label_period_months in list_label_period_months:
        for forecast_gap_months in list_forecast_gap_months:
            for data_period_months in list_data_period_months:
                for data_path in list_data_path:
                    for evolution_event_selection in list_evolution_event_selection:
                        filepath_list = []
                        print(f"************ Merge Features {count}/{total_count}")
                        for window_size in list_window_size:  # week / data points
                            shapelet_file_name = util.get_shapelet_file_name(label_period_months, forecast_gap_months,
                                                                             data_period_months,
                                                                             n_shapelets, window_size, window_step,
                                                                             n_channels, train_data_path, evolution_event_selection)
                            filepath_list.append(f"{feature_folder_path}{shapelet_file_name}{data_path}")
                        # window size = 3
                        output_file = f"{filepath_list[0]}.multi_sizes.csv"
                        merge_feature_files(filepath_list, output_file)
                        count += 1


if __name__ == "__main__":
    # create DataFrames
    df1 = pd.DataFrame({'a': [0, 0, 1, 1, 2],
                        'b': [0, 0, 1, 1, 1],
                        'c': [11, 8, 10, 6, 6]})

    df2 = pd.DataFrame({'a': [0, 1, 1, 1, 3],
                        'b': [0, 0, 0, 1, 1],
                        'd': [22, 24, 25, 33, 37]})

    df1 = pd.merge(df1, df2, how='inner', on=['a', 'b'])
    print(df1)
