import os
import sys

import numpy as np
import pyecharts.options as opts
from pyecharts.charts import Line
from pyecharts.render import make_snapshot
from snapshot_phantomjs import snapshot

import global_settings


def np_move_avg(a, n, mode="same"):
    return (np.convolve(a, np.ones((n,)) / n, mode=mode))


def plot_aggregate_index_curves_single_proj(aggregated_split, aggregated_shrink, aggregated_merge,
                                            aggregated_expand, start_time_list, proj, status, result_folder_path):
    plot_folder_path = result_folder_path + "plots/"
    if not os.path.exists(plot_folder_path):
        os.makedirs(plot_folder_path)

    ma_n = 12

    if 'linux' in sys.platform or 'Linux' in sys.platform:
        # https://www.cnblogs.com/hodge01/p/15742492.html
        linux_js_file_path = "{}/".format(os.path.dirname(os.path.abspath("/home/wangliang/lib/echarts.min.js")))
        l = Line(init_opts=opts.InitOpts(js_host=linux_js_file_path))
    else:
        l = Line()
    l.set_global_opts(
        title_opts=opts.TitleOpts(title=f"{proj} {status}"),
        tooltip_opts=opts.TooltipOpts(is_show=False),
        xaxis_opts=opts.AxisOpts(type_="category"),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
    ).add_xaxis(xaxis_data=start_time_list).add_yaxis(
        series_name="split",
        y_axis=np_move_avg(aggregated_split, ma_n),
        symbol="emptyCircle",
        is_symbol_show=True,
        label_opts=opts.LabelOpts(is_show=False),
    ).add_yaxis(
        series_name="shrink",
        y_axis=np_move_avg(aggregated_shrink, ma_n),
        symbol="emptyCircle",
        is_symbol_show=True,
        label_opts=opts.LabelOpts(is_show=False),
    ).add_yaxis(
        series_name="merge",
        y_axis=np_move_avg(aggregated_merge, ma_n),
        symbol="emptyCircle",
        is_symbol_show=True,
        label_opts=opts.LabelOpts(is_show=False),
    ).add_yaxis(
        series_name="expand",
        y_axis=np_move_avg(aggregated_expand, ma_n),
        symbol="emptyCircle",
        is_symbol_show=True,
        label_opts=opts.LabelOpts(is_show=False),
    )
    proj_name = proj
    if global_settings.USE_NEW_DATA:
        proj_name = proj.replace("/", "__")
    make_snapshot(snapshot, l.render(), plot_folder_path + proj_name + ".png")
