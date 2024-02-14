import os
import sys

import numpy as np
import pyecharts.options as opts
from pyecharts.charts import Line
from pyecharts.render import make_snapshot
from snapshot_phantomjs import snapshot
# import snapshot

import global_settings
import util


# def np_move_avg(a, n, mode="same"):
#     return (np.convolve(a, np.ones((n,)) / n, mode=mode))


def plot_n_shapelets(loaded_shape_dict, output_root_dir):
    n_shapelets = loaded_shape_dict["n_shapelets"]
    window_size = loaded_shape_dict["window_size"]
    window_step = loaded_shape_dict["window_step"]
    n_channels = loaded_shape_dict["n_channels"]
    data_path = loaded_shape_dict["data_path"]
    label_period_months = loaded_shape_dict["label_period_months"]
    forecast_gap_months = loaded_shape_dict["forecast_gap_months"]
    data_period_months = loaded_shape_dict["data_period_months"]

    png_filename_prefix = util.get_shapelet_file_name(label_period_months, forecast_gap_months, data_period_months,
                                                      n_shapelets, window_size, window_step, n_channels, data_path,global_settings.EVOLUTION_EVENT_COMBINATIONS)

    for idx, shape_dict in enumerate(loaded_shape_dict["shapelets"]):
        shapelet = [[] for ch in range(n_channels)]
        shapelet[0] = shape_dict["shape_split"]
        shapelet[1] = shape_dict["shape_shrink"]
        shapelet[2] = shape_dict["shape_merge"]
        shapelet[3] = shape_dict["shape_expand"]
        repo_name = shape_dict["from_repo_name"]
        score = shape_dict["score"]
        repo_label = shape_dict["from_repo_label"]

        title = f"repo_lb:{repo_label} repo:{repo_name} ws:{window_size} score:{score} label:{label_period_months} fore:{forecast_gap_months} data:{data_period_months}"
        png_path = f"{output_root_dir}{png_filename_prefix}_{idx}.png"
        plot_one_shapelet(shapelet, title, window_size, png_path)


def plot_one_shapelet(shapelet, title, window_size, png_path):
    if 'linux' in sys.platform or 'Linux' in sys.platform:
        # https://www.cnblogs.com/hodge01/p/15742492.html
        linux_js_file_path = "{}/".format(os.path.dirname(os.path.abspath("")))
        line = Line(init_opts=opts.InitOpts(js_host=linux_js_file_path))
    else:
        line = Line()
    line.set_global_opts(
        title_opts=opts.TitleOpts(title=f"{title}", pos_bottom="0"),
        tooltip_opts=opts.TooltipOpts(is_show=False),
        xaxis_opts=opts.AxisOpts(type_="category"),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
    ).add_xaxis(xaxis_data=[i for i in range(window_size)]
                ).add_yaxis(
        series_name="split",
        y_axis=shapelet[0],
        symbol="emptyCircle",
        is_symbol_show=True,
        label_opts=opts.LabelOpts(is_show=True),
    ).add_yaxis(
        series_name="shrink",
        y_axis=shapelet[1],
        symbol="emptyCircle",
        is_symbol_show=True,
        label_opts=opts.LabelOpts(is_show=True),
    ).add_yaxis(
        series_name="merge",
        y_axis=shapelet[2],
        symbol="emptyCircle",
        is_symbol_show=True,
        label_opts=opts.LabelOpts(is_show=True),
    ).add_yaxis(
        series_name="expand",
        y_axis=shapelet[3],
        symbol="emptyCircle",
        is_symbol_show=True,
        label_opts=opts.LabelOpts(is_show=True),
    )
    make_snapshot(snapshot, line.render(), png_path)
