import os

import global_settings
import shapelets
import util


def print_shapelets():
    data_dir = util.get_data_dir()

    n_shapelets = global_settings.N_SHAPELETS
    window_step = global_settings.WINDOW_STEP
    n_channels = global_settings.N_CHANNELS
    n_jobs = global_settings.N_JOBS

    list_label_period_months = [12]
    list_forecast_gap_months = [3]
    list_data_period_months = [12]
    # 以data point为单位
    list_window_size = global_settings.LIST_WINDOW_SIZE  # , 8, 12, 16, 20, 24]   # , 28, 32, 36, 40, 44, 48]
    list_evolution_event_selection = [[0, 1, 2, 3]]
    # list_evolution_event_selection = global_settings.EVOLUTION_EVENT_COMBINATIONS

    train_data_path = f"{data_dir}index_productivity_32.csv"

    for label_period_months in list_label_period_months:
        for forecast_gap_months in list_forecast_gap_months:
            for data_period_months in list_data_period_months:
                for window_size in list_window_size:  # week / data points
                    positive_s = None
                    nagative_s = None
                    positive_comb = None
                    nagative_comb = None
                    print('-----------------------------')
                    print(f"window size: {window_size}")
                    for evolution_event_selection in list_evolution_event_selection:
                        shapelet_file_name = util.get_shapelet_file_name(label_period_months, forecast_gap_months,
                                                                         data_period_months,
                                                                         n_shapelets, window_size, window_step,
                                                                         n_channels, train_data_path,
                                                                         evolution_event_selection)
                        # root_folder = "..\\000_shapelet_results\\result\\2023-06-14T08-37-16Z_smooth05_global_noorder_penaltyF_abs_参数扫描_演化事件扫描_fix_mining_f3_d12_最终采用的结果\\"
                        root_folder = "..\\000_shapelet_results\\result\\2023-06-19T17-26-24Z\\"

                        shapelet_file_path = f"{root_folder}{shapelet_file_name}"
                        sp_dict = shapelets.load_mined_shapelets(shapelet_file_path)
                        sps = sp_dict["shapelets"]
                        for shape_doc in sps:
                            if shape_doc["score"] > 1:
                                if positive_s is None:
                                    positive_s = shape_doc
                                    positive_comb = evolution_event_selection
                                elif positive_s['info_gain'] < shape_doc['info_gain']:
                                    positive_s = shape_doc
                                    positive_comb = evolution_event_selection
                                else:
                                    pass

                            elif shape_doc["score"] < 1:
                                if nagative_s is None:
                                    nagative_s = shape_doc
                                    nagative_comb = evolution_event_selection
                                elif nagative_s['info_gain'] < shape_doc['info_gain']:
                                    nagative_s = shape_doc
                                    nagative_comb = evolution_event_selection
                                else:
                                    pass
                            else:
                                pass

                    print(positive_s)
                    print(positive_comb)
                    print(nagative_s)
                    print(nagative_comb)
                    print('-----------------------------')


if __name__ == '__main__':
    print_shapelets()
