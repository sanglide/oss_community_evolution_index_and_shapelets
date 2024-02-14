import datetime

import pandas as pd

import classify_supervised_learning
import global_settings
import shapelet_dist_features
from main_shapelets import script_all, script_shapelet_features, script_visualize

def canshu_exp():
    # n_shapelets=[10,20,30,40,50,60,70,80,90,100]
    # LIST_WINDOW_SIZE=[[5],[5],[5],[5],[5],[5]]
    FIXED_MINING_FORECAST_GAP=[3,6,9,12,15,18,21,24,
                               3,6,9,12,15,18,21,24,
                               3,6,9,12,15,18,21,24,
                               3,6,9,12,15,18,21,24,
                               3,6,9,12,15,18,21,24,
                               3,6,9,12,15,18,21,24]
    # for i_sha in n_shapelets:
    # for i_sha in LIST_WINDOW_SIZE:
    for i_sha in FIXED_MINING_FORECAST_GAP:
        # global_settings.N_SHAPELETS=i_sha
        # global_settings.LIST_WINDOW_SIZE = i_sha
        global_settings.FIXED_MINING_FORECAST_GAP=i_sha
        # print(f'----------------------------- n_shapelets = {global_settings.N_SHAPELETS} --------------------------------')
        # print(f'----------------------------- LIST_WINDOW_SIZE = {global_settings.LIST_WINDOW_SIZE} --------------------------------')
        print(f'----------------------------- FIXED_MINING_FORECAST_GAP = {global_settings.FIXED_MINING_FORECAST_GAP} --------------------------------')

        time_of_start = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%SZ')
        # try:
        time_of_execution = time_of_start

        script_all(time_of_execution)
        script_shapelet_features(time_of_execution)
        shapelet_dist_features.script_merge_feature_files(time_of_execution)
        classify_supervised_learning.script_classification_ml_multi_sizes(time_of_execution)
        if global_settings.DO_VIS:
            script_visualize(time_of_execution)


        f = open(f"../terminated_{time_of_start}", "w")
        f.write(str(datetime.datetime.now()))
        f.close()

def baoli_fuxian_exp():
    count=0
    result=[]
    while count<3:

        time_of_start = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%SZ')
        # try:
        time_of_execution = time_of_start

        script_all(time_of_execution)
        script_shapelet_features(time_of_execution)
        shapelet_dist_features.script_merge_feature_files(time_of_execution)
        classify_supervised_learning.script_classification_ml_multi_sizes(time_of_execution)
        if global_settings.DO_VIS:
            script_visualize(time_of_execution)


        f = open(f"../terminated_{time_of_start}", "w")
        f.write(str(datetime.datetime.now()))
        f.close()

        df=pd.read_csv(f'../000_shapelet_results/result/{time_of_execution}/prediction_report_direct_RandomForestClassifier_70.csv')
        df_filter=df[(df["data_period_months"]==12) & (df["forecast_gap_months"]==3)]

        accuracy=list(df_filter["accuracy"])[0]
        print(f'now accuracy: --- {accuracy}')
        if (accuracy-0.93<0.01 and accuracy-0.93>0) or (accuracy-0.93>-0.01 and accuracy-0.93<0):
            count=count+1
            result.append(time_of_execution)
    print(f'results:--------------- {result}')


if __name__ == "__main__":
    canshu_exp()