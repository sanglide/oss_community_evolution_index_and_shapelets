import datetime
import os
import sys

import numpy as np
import pandas as pd
import pyecharts.options as opts
from pyecharts.charts import Line
from pyecharts.charts import Scatter
from pyecharts.render import make_snapshot
from sklearn import preprocessing
# from snapshot_phantomjs import snapshot
import snapshot

import data_source_io
import global_settings


def test_kill_and_exit():
    if os.path.exists("./kill"):
        os.remove("./kill")
        f = open("./killed", "w")
        f.write(str(datetime.datetime.now()))
        f.close()
        sys.exit(-1)


# https://www.csdn.net/tags/MtjakgwsNjU2OS1ibG9n.html
# https://www.kaggle.com/questions-and-answers/49165

def normalize_0_1(arr):
    maxv = max(arr)
    minv = min(arr) if min(arr) > 0 else 0
    return [(a - minv) / (maxv - minv) if a > 0 else a / (maxv - minv) for a in arr]


def z_normalize(arr):
    arr = np.asarray(arr)
    arr = arr.reshape(-1, 1)
    zscore = preprocessing.StandardScaler()
    return np.squeeze(zscore.fit_transform(arr))


def minmax_normalize(arr):
    arr = np.asarray(arr)
    arr = arr.reshape(-1, 1)
    minmax_scale = preprocessing.MinMaxScaler()
    return np.squeeze(minmax_scale.fit_transform(arr))


def maxabs_normalize(arr):
    arr = np.asarray(arr)
    arr = arr.reshape(-1, 1)
    maxabs_scale = preprocessing.MaxAbsScaler()
    return np.squeeze(maxabs_scale.fit_transform(arr))


def np_move_avg(a, n, mode="same"):
    a = maxabs_normalize(a)
    return (np.convolve(a, np.ones((n,)) / n, mode=mode))


def __plot_idx_productivity_scatter(time, split, shrink, merge,
                                    expand, x_data, png_path):
    if 'linux' in sys.platform or 'Linux' in sys.platform:
        # https://www.cnblogs.com/hodge01/p/15742492.html
        linux_js_file_path = "{}/".format(os.path.dirname(os.path.abspath("/home/wangliang/lib/echarts.min.js")))
        scatter = Scatter(init_opts=opts.InitOpts(js_host=linux_js_file_path))
    else:
        scatter = Scatter()
    scatter.add_xaxis(xaxis_data=x_data)
    scatter.add_yaxis(series_name='split', y_axis=np.asarray(split), symbol_size=5,
                      label_opts=opts.LabelOpts(is_show=False))
    scatter.add_yaxis(series_name='shrink', y_axis=np.asarray(shrink), symbol_size=5,
                      label_opts=opts.LabelOpts(is_show=False))
    scatter.add_yaxis(series_name='merge', y_axis=np.asarray(merge), symbol='rect', symbol_size=5,
                      label_opts=opts.LabelOpts(is_show=False))
    scatter.add_yaxis(series_name='expand', y_axis=np.asarray(expand), symbol='rect', symbol_size=5,
                      label_opts=opts.LabelOpts(is_show=False))
    scatter.set_series_opts().set_global_opts(
        xaxis_opts=opts.AxisOpts(
            type_="value", splitline_opts=opts.SplitLineOpts(is_show=True)
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        tooltip_opts=opts.TooltipOpts(is_show=False),
    )
    make_snapshot(snapshot, scatter.render("scatter.html"), png_path)


def __plot_idx_productivity(time, split, shrink, merge,
                            expand, commit_count, commit_count_diff, issue_count, pr_count, issue_pr_count,
                            member_count, proj, png_path):
    ma_n = 20

    if 'linux' in sys.platform or 'Linux' in sys.platform:
        # https://www.cnblogs.com/hodge01/p/15742492.html
        linux_js_file_path = "{}/".format(os.path.dirname(os.path.abspath("/home/wangliang/lib/echarts.min.js")))
        l = Line(init_opts=opts.InitOpts(js_host=linux_js_file_path))
    else:
        l = Line()
    l.set_global_opts(
        # title_opts=opts.TitleOpts(title=f"{proj}"),
        tooltip_opts=opts.TooltipOpts(is_show=False),
        xaxis_opts=opts.AxisOpts(type_="category"),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
    ).add_xaxis(xaxis_data=time).add_yaxis(
        series_name="split",
        y_axis=np_move_avg(split, ma_n),
        symbol="emptyCircle",
        is_symbol_show=True,
        label_opts=opts.LabelOpts(is_show=False),
    ).add_yaxis(
        series_name="shrink",
        y_axis=np_move_avg(shrink, ma_n),
        symbol="emptyCircle",
        is_symbol_show=True,
        label_opts=opts.LabelOpts(is_show=False),
    ).add_yaxis(
        series_name="merge",
        y_axis=np_move_avg(merge, ma_n),
        symbol="emptyCircle",
        is_symbol_show=True,
        label_opts=opts.LabelOpts(is_show=False),
    ).add_yaxis(
        series_name="expand",
        y_axis=np_move_avg(expand, ma_n),
        symbol="emptyCircle",
        is_symbol_show=True,
        label_opts=opts.LabelOpts(is_show=False),
    ).add_yaxis(
        series_name="commit_count",
        y_axis=np_move_avg(commit_count, ma_n),
        symbol="emptyCircle",
        is_symbol_show=True,
        label_opts=opts.LabelOpts(is_show=False),
    ).add_yaxis(
        series_name="commit_count_diff",
        y_axis=np_move_avg(commit_count_diff, ma_n),
        symbol="emptyCircle",
        is_symbol_show=True,
        label_opts=opts.LabelOpts(is_show=False),
    )
    # ).add_yaxis(
    #     series_name="issue_count",
    #     y_axis=np_move_avg(issue_count, ma_n),
    #     symbol="square",
    #     is_symbol_show=True,
    #     label_opts=opts.LabelOpts(is_show=False),
    # ).add_yaxis(
    #     series_name="pr_count",
    #     y_axis=np_move_avg(pr_count, ma_n),
    #     symbol="diamond",
    #     is_symbol_show=True,
    #     label_opts=opts.LabelOpts(is_show=False),
    # ).add_yaxis(
    #     series_name="issue_pr_count",
    #     y_axis=np_move_avg(issue_pr_count, ma_n),
    #     symbol="rect",
    #     is_symbol_show=True,
    #     label_opts=opts.LabelOpts(is_show=False),
    # ).add_yaxis(
    #     series_name="member_count",
    #     y_axis=np_move_avg(member_count, ma_n),
    #     symbol="triangle",
    #     is_symbol_show=True,
    #     label_opts=opts.LabelOpts(is_show=False),
    # )
    make_snapshot(snapshot, l.render(), png_path)


def __write_productivity_csv(file_path, data):
    fo = open(file_path, 'w')
    fo.write(
        'project_name,time,split,shrink,merge,expand,commit_count,commit_count_diff,project_age,issue_count,pr_count,issue_pr_count,member_count\n')
    for i in range(data.shape[0]):
        fo.write(
            f"{data.iloc[i]['project_name']},{data.iloc[i]['time']},"
            f"{data.iloc[i]['split']},{data.iloc[i]['shrink']},"
            f"{data.iloc[i]['merge']},{data.iloc[i]['expand']},"
            f"{data.iloc[i]['commit_count']},{data.iloc[i]['commit_count_diff']},"
            f"{data.iloc[i]['project_age']},"
            f"{data.iloc[i]['issue_count']},"
            f"{data.iloc[i]['pr_count']},"
            f"{data.iloc[i]['issue_pr_count']},"
            f"{data.iloc[i]['member_count']},"
            f"\n")
    fo.flush()
    fo.close()


def productivity_analysis(result_path):
    index_productivity_file = f'{result_path}index_productivity.csv'
    productivity_path = f'{result_path}productivity/'
    if not os.path.exists(productivity_path):
        os.makedirs(productivity_path)
    data_product = pd.read_csv(index_productivity_file,
                               usecols=["project_name", "time", "split", "shrink", "merge", "expand", "commit_count",
                                        "commit_count_diff", "project_age", "issue_count", "pr_count", "issue_pr_count",
                                        "member_count"])
    # data_product_per_proj = []
    for proj in global_settings.proj_list:
        test_kill_and_exit()
        print(proj)
        proj_file_name = f"{proj.replace('/', '__')}"
        proj_file_path = productivity_path + proj_file_name + ".csv"
        if os.path.exists(productivity_path + proj_file_name + "__scatter_commit_count.png"):
            continue
        proj_data = data_product[data_product['project_name'] == proj]
        proj_data = proj_data.sort_values(by='time')
        __write_productivity_csv(proj_file_path, proj_data)
        __plot_idx_productivity(proj_data['time'], proj_data['split'], proj_data['shrink'], proj_data['merge'],
                                proj_data['expand'], proj_data['commit_count'], proj_data['commit_count_diff'],
                                proj_data['issue_count'], proj_data['pr_count'], proj_data['issue_pr_count'],
                                proj_data['member_count'], proj,
                                productivity_path + proj_file_name + ".png")
        __plot_idx_productivity_scatter(proj_data['time'], proj_data['split'], proj_data['shrink'], proj_data['merge'],
                                        proj_data['expand'], proj_data['commit_count_diff'],
                                        productivity_path + proj_file_name + "__scatter_commit_count_diff.png")

        __plot_idx_productivity_scatter(proj_data['time'], proj_data['split'], proj_data['shrink'], proj_data['merge'],
                                        proj_data['expand'], proj_data['commit_count'],
                                        productivity_path + proj_file_name + "__scatter_commit_count.png")
        print("\t\tDone: " + proj)
    return True


def execute_analysis(result_folder_name):
    if global_settings.USE_NEW_DATA:
        # todo
        global_settings.proj_list = data_source_io.list_projects("all")
        global_settings.activeProjL = data_source_io.list_projects("active")
        global_settings.failProjL = data_source_io.list_projects("failed")
    result_path = f'./result/productivity_plots/{result_folder_name}/'
    return productivity_analysis(result_path)


if __name__ == '__main__':
    # while True:
    #     try:
    #         if execute_analysis('2022-06-10T00-59-43Z_interval_7_days_x_12'):
    #             break
    #     except Exception as e:
    #         print(e)

    while True:
        try:
            if execute_analysis('2022-06-09T15-03-32Z_interval_7_days_x_12'):
                break
        except Exception as e:
            print(e)

    while True:
        try:
            if execute_analysis('2022-06-10T17-09-43Z_interval_7_days_x_12'):
                break
        except Exception as e:
            print(e)

    f = open("./terminated", "w")
    f.write(str(datetime.datetime.now()))
    f.close()
