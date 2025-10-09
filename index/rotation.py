import polars as pl
import numpy as np
from pprint import pprint
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource
from datetime import datetime
from pandas_market_calendars import get_calendar
from zoneinfo import ZoneInfo
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from gold.momentum import to_html


def preprocess():
    all_index_df = pl.read_csv("data/index/s[hz]*.csv").with_columns(
        pl.col("date").str.to_datetime()
    )

    start_date = all_index_df.select(pl.col("date").min().over("code")).max().item()
    end_date = all_index_df.select(pl.col("date").max().over("code")).min().item()

    # 添加未来交易日
    sse = get_calendar("SSE")
    next_date = sse.schedule(end_date, "2030-12-31").index[1]

    future_rows = all_index_df.group_by("code").last().with_columns(date=next_date)
    all_index_df = (
        pl.concat([all_index_df.select(future_rows.columns), future_rows])
        .sort(["date", "code"])
        .with_columns(
            ret=pl.col("close").pct_change().over("code"),
            # (pl.col('close') / pl.col('open') - 1).over('code').alias('ret'),
        )
    )

    return all_index_df, start_date, end_date


def backtest(n, th):
    bt_df = (
        all_index_df.with_columns(
            n_ret=pl.col("close").pct_change(n).over("code"),
        )
        .with_columns(
            n_ret_rank=pl.col("n_ret").rank(descending=True).over("date"),
        )
        .with_columns(
            pl.col("n_ret").shift(1).over("code"),
            pl.col("n_ret_rank").shift(1).over("code"),
        )
        .with_columns(
            pl.col("n_ret").max().over("date").alias("max_n_ret"),
        )
        .filter(
            (pl.col("date") >= start_date)
            & pl.col("max_n_ret").is_not_null()
            & (pl.col("n_ret_rank") <= 1)
            & (pl.col("max_n_ret") > th)
        )
        .with_columns(
            weighted_ret=1 / pl.count("code").over("date") * pl.col("ret"),
        )
        .group_by("date")
        .agg(port_ret=pl.sum("weighted_ret"), holdings=pl.col("code"))
        .sort("date")
        .with_columns(
            nav=(1 + pl.col("port_ret").fill_null(0)).cum_prod(),
            holdings_str=pl.col("holdings").list.join(", "),
        )
    )
    return bt_df


def performance(v, days, annual_days=365):
    ann_ret = (v[-1] / v[0]) ** (annual_days / days) - 1
    mdd = np.min(v / np.maximum.accumulate(v) - 1)
    calmar = -ann_ret / mdd if mdd < 0 else np.inf
    ret = v[1:] / v[:-1] - 1
    ann_vol = ret.std() * np.sqrt(annual_days)
    sharpe = ann_ret / ann_vol if ann_vol else np.inf
    return {
        "年化收益": ann_ret,
        "最大回撤": mdd,
        "卡玛比率": calmar,
        "夏普比率": sharpe,
    }


if __name__ == "__main__":
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    if get_calendar("SSE").schedule(today, today).empty:
        print(f"非交易日: {today}")
        exit(0)

    all_index_df, start_date, end_date = preprocess()

    best_n = 1
    best_th = 0
    best_perf = 0
    for n in range(1, 32):
        for th in np.arange(0, 0.05, 0.001):
            bt_df = backtest(n, th)[-244 * 2 :]
            days = (bt_df["date"].last() - bt_df["date"].first()).days
            nav = bt_df["nav"].to_numpy()
            perf = performance(nav, days)["年化收益"]
            if perf > best_perf:
                best_n = n
                best_th = th
                best_perf = perf
    print(best_n, best_th, best_perf)

    bt_df = backtest(best_n, best_th)[-244 * 2 :]
    days = (bt_df["date"].last() - bt_df["date"].first()).days
    nav = bt_df["nav"].to_numpy()
    pprint(performance(nav, days))

    # 画图
    p = figure(
        title="轮动策略",
        x_axis_label="日期",
        y_axis_label="净值",
        x_axis_type="datetime",
        width=800,
        height=400,
    )

    source = ColumnDataSource(bt_df.with_columns(pl.col("date").cast(pl.Datetime)))
    p.line(
        x="date",
        y="nav",
        source=source,
        legend_label="策略净值",
        line_width=2,
        color="red",
        alpha=0.7,
    )

    hover = HoverTool(
        tooltips=[
            ("日期", "@date{%F}"),
            ("净值", "@nav{0.000}"),
            ("当日收益", "@port_ret{0.00%}"),
            ("持仓", "@holdings_str"),
        ],
        formatters={"@date": "datetime"},
        mode="vline",
    )
    p.add_tools(hover)

    p.legend.location = "top_left"

    to_html(p, "index/rotation.html")
