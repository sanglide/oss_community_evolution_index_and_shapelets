import pandas as pd
import numpy as np
import global_settings


#
# def read_result_csv(csv_path, forecast_gap, data_period, evolution_event_selection):
#     data = pd.read_csv(csv_path)
#     # fix the following code to select rows with the "forecast_gap_months" field equals to value forecast_gap
#     # filtered_data = data[data["forecast_gap_months"] == forecast_gap]
#     filtered_data = data[data["data_period_months"] == data_period]
#     #
#
#     # for evol in global_settings.EVOLUTION_EVENT_COMBINATIONS:
#     #     evol_str = str(evol)
#     #     temp_filtered_data = filtered_data[filtered_data["evolution_event_selection"] == evol_str]
#     #     print(f"{evol_str.replace(' ','')}", end="")
#     #     for forecast in global_settings.LIST_FORECAST_GAP_MONTHS:
#     #         accuracy = temp_filtered_data[temp_filtered_data["forecast_gap_months"] == forecast]["accuracy"]
#     #         print(f" {float(accuracy)}", end="")
#     #     print()
#
#     print("forecast_gap", end='')
#     for evol in global_settings.EVOLUTION_EVENT_COMBINATIONS:
#         evol_str = str(evol)
#         print(f" {evol_str.replace(' ','')}", end='')
#     print()
#
#
#     for forecast in [3,6,9,12,15,18,21,24]:
#         temp_filtered_data = filtered_data[filtered_data["forecast_gap_months"] == forecast]
#         print(f"{forecast}", end="")
#         for evol in global_settings.EVOLUTION_EVENT_COMBINATIONS:
#             evol_str = str(evol)
#             accuracy = temp_filtered_data[temp_filtered_data["evolution_event_selection"] == evol_str]["accuracy"]
#             print(f" {float(accuracy)}", end="")
#         print()
#
#
#
#     # filtered_data.to_csv("./result/filtered.csv")

def get_line_chart(csv_path, f_g_list, d_p_list, e_s):
    data = pd.read_csv(csv_path)
    re = []
    evol_str = str(e_s)
    filtered_data_es = data[data["evolution_event_selection"] == evol_str]
    for f_g in f_g_list:
        temp = []
        filtered_data = filtered_data_es[filtered_data_es["forecast_gap_months"] == f_g]
        for d_p in d_p_list:
            temp.append(float(filtered_data[filtered_data["data_period_months"] == d_p]["accuracy"]))
        re.append(temp)
    data_df = pd.DataFrame(data=re, columns=["d_p_" + str(x) for x in d_p_list])
    index = pd.Series(["f_g_" + str(x) for x in f_g_list])
    data_df.set_index(index, inplace=True)

    data_df = pd.DataFrame(data_df.values.T, columns=index, index=["d_p_" + str(x) for x in d_p_list])
    return data_df


def get_bar_chart(csv_path, f_g, d_p, e_s):
    data = pd.read_csv(csv_path)
    re = []
    evol_str = str(e_s)
    filtered_data_es = data[data["evolution_event_selection"] == evol_str]

    temp = []
    filtered_data = filtered_data_es[filtered_data_es["forecast_gap_months"] == f_g]

    accuracy = float(filtered_data[filtered_data["data_period_months"] == d_p]["accuracy"])
    f1 = (float(filtered_data[filtered_data["data_period_months"] == d_p]["cr1_f1"]) + float(
        filtered_data[filtered_data["data_period_months"] == d_p]["cr0_f1"])) / 2
    recall = (float(filtered_data[filtered_data["data_period_months"] == d_p]["cr0_recall"]) + float(
        filtered_data[filtered_data["data_period_months"] == d_p]["cr1_recall"])) / 2
    precision = (float(filtered_data[filtered_data["data_period_months"] == d_p]["cr0_precision"]) + float(
        filtered_data[filtered_data["data_period_months"] == d_p]["cr1_precision"])) / 2

    re.append([accuracy, f1, recall, precision])
    data_df = pd.DataFrame(data=re, columns=['Accuracy', 'F1 Score', 'Recall', 'Precision'])
    # index = pd.Series(["f_g_" + str(x) for x in f_g_list])
    # data_df.set_index(index, inplace=True)

    # data_df = pd.DataFrame(data_df.values.T, columns=index, index=["d_p_" + str(x) for x in d_p_list])
    return data_df


if __name__ == '__main__':
    forcast_gap = 6  # no use
    evolution_event_selection = "[0, 1, 2, 3]"  # no use

    # csv_path = 'result\\2023-06-09T17-40-07Z_smooth05_global_noorder_penalty2_abs_参数扫描_演化事件扫描_添加离散化_随机森林破70%\\results\\prediction_report_direct_RandomForestClassifier_50.csv'
    # data_period = 6

    # csv_path = 'result\\2023-06-10T11-33-24Z_smooth05_global_noorder_penalty2_euclidean_参数扫描_演化事件扫描_添加离散化_随机森林bagging决策树破70%\\prediction_report_direct_RandomForestClassifier_10.csv'
    # data_period = 3

    # csv_path = 'result\\2023-06-10T11-33-24Z_smooth05_global_noorder_penalty2_euclidean_参数扫描_演化事件扫描_添加离散化_随机森林bagging决策树破70%\\prediction_report_direct_BaggingClassifier_DecisionTree.csv'
    # data_period = 6
    # data_period = 12

    csv_path = 'prediction_report_direct_RandomForestClassifier_100.csv'
    # csv_path = '..\\000_shapelet_results\\result\\2023-06-14T08-37-16Z_smooth05_global_noorder_penaltyF_abs_参数扫描_演化事件扫描_fix_mining_f3_d12\\prediction_report_direct_RandomForestClassifier_100.csv'
    data_period = 12

    f_g_list, d_p_list = [3, 6, 9, 12, 15, 18, 21, 24], [3, 6, 9, 12, 15, 18, 21, 24]
    # read_result_csv(csv_path, forcast_gap, data_period, evolution_event_selection)
    # data_df=get_line_chart(csv_path, f_g_list, d_p_list, [0, 1, 2, 3])
    # data_df.to_csv("line_chart.csv")

    re = []
    e_s_list = [[0], [1], [2], [3], [0, 1], [0, 2], [0, 3],
                [1, 2],
                [1, 3],
                [2, 3],
                [0, 1, 2],
                [0, 1, 3],
                [0, 2, 3],
                [1, 2, 3],
                [0, 1, 2, 3]]
    # da=pd.DataFrame()
    # for e_s in e_s_list:
    #     data_df=get_line_chart(csv_path,f_g_list,d_p_list,e_s)
    #     co=np.array([str(e_s) for i in range(len(data_df))])
    #     data_df.insert(0, "e_s", co)
    #     da=pd.concat([da, data_df])
    # da.to_csv("index_line_chart.csv")

    f_g, d_p = 3, 12
    da = pd.DataFrame()
    str_es = ["split", "shrink", "merge", "expand"]
    for e_s in e_s_list:
        data_df = get_bar_chart(csv_path, f_g, d_p, e_s)
        esss = [str_es[i] for i in e_s]
        esss_str = esss[0]
        for j in range(1, len(esss)):
            esss_str = esss_str + "," + esss[j]
        co = np.array([esss_str for i in range(len(data_df))])
        data_df.insert(0, "e_s", co)
        da = pd.concat([da, data_df])
    da.to_csv("index_bar_chart_" + str(f_g) + "_" + str(d_p) + ".csv")
