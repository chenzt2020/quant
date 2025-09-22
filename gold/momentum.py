import akshare as ak
import numpy as np
import pandas as pd
from bokeh.embed import file_html
from bokeh.layouts import column
from bokeh.models import CrosshairTool, Div, HoverTool
from bokeh.plotting import figure
from bokeh.resources import CDN
from datetime import datetime
from pandas_market_calendars import get_calendar
from zoneinfo import ZoneInfo


def get_stock_data(code):
    df = ak.fund_etf_hist_em(symbol=code, adjust="hfq").sort_values("日期")
    df["日期"] = pd.to_datetime(df["日期"])
    return df["收盘"].to_numpy(), df["日期"].to_numpy()


def get_stock_data_sina(code):
    df = ak.fund_etf_hist_sina(symbol=code).sort_values("date")
    df["date"] = pd.to_datetime(df["date"])
    return df["close"].to_numpy(), df["date"].to_numpy()


def annual_return(v, annual_days=244):
    return v[-1] ** (annual_days / len(v)) - 1


def max_drawdown(v):
    return np.min(v / np.maximum.accumulate(v) - 1)


def sharpe(v, annual_days=244, risk_free=0.02):
    ret = np.diff(v) / v[:-1]
    return (ret.mean() - risk_free / annual_days) / ret.std() * np.sqrt(annual_days)


def momentum_strategy(values, n, print_log=False):
    # 计算动量和信号
    momentum = pd.Series(values).pct_change(n)
    cond = momentum > 0.0
    signal_0 = cond.astype(int).diff().fillna(0).astype(int)
    signal = signal_0.shift(1).fillna(0).astype(int)
    # signal = signal_0

    first_buy = signal[signal == 1].index[0] if 1 in signal.values else None
    if first_buy:
        signal.loc[: first_buy - 1] = 0

    # 回测
    cash, pos = 1.0, 0
    trade_log, port = [], []

    for i, sig in enumerate(signal):
        if sig == 1:
            pos, cash = cash / values[i], 0
            trade_log.append((i, values[i]))
        elif sig == -1:
            cash, pos = pos * values[i], 0
            trade_log.append((i, values[i]))
        port.append(cash + pos * values[i])

    if signal_0.iloc[-1] != 0:
        trade_log.append((-1, values[-1]))

    # 计算策略指标
    port = np.array(port)
    ann_ret = annual_return(port)
    mdd = max_drawdown(port)
    calmar_ratio = -ann_ret / mdd if mdd < 0 else np.inf
    sharpe_ratio = sharpe(port)
    trade_times = len(trade_log)

    wins = sum(
        trade_log[i][1] > trade_log[i - 1][1] for i in range(1, len(trade_log), 2)
    )
    win_rate = wins / (trade_times // 2) if trade_times >= 2 else 0

    # 打印策略结果
    if print_log:
        print(
            f"momentum({n}):\n"
            f"年化收益: {ann_ret:.2%} 最大回撤: {mdd:.2%}\n"
            f"卡玛比率: {calmar_ratio:.2f} 夏普比率: {sharpe_ratio:.2f}\n"
            f"交易次数: {trade_times} 胜率: {win_rate:.2%}"
        )
    return (
        port,
        trade_log,
        (ann_ret, mdd, calmar_ratio, sharpe_ratio, trade_times, win_rate),
    )


if __name__ == "__main__":
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    if get_calendar("SSE").schedule(today, today).empty:
        print(f"非交易日: {today}")
        exit(0)

    # 加载数据
    values, dates = get_stock_data_sina("sh518880")
    if len(values) > 244:
        values = values[-244:]
        dates = dates[-244:]
    values /= values[0]

    # 原始策略指标
    ann_ret = annual_return(values)
    mdd = max_drawdown(values)
    metrics_text_0 = (
        f"原始策略:\n年化收益: {ann_ret:.2%} 最大回撤: {mdd:.2%}\n"
        f"卡玛比率: {-ann_ret / mdd:.2f} 夏普比率: {sharpe(values):.2f}"
    )
    print(metrics_text_0)

    # 优化策略
    best_n, best_calmar = None, -np.inf
    for n in range(1, 61):
        _, _, metrics = momentum_strategy(values, n)
        if metrics[0] > best_calmar:
            best_calmar, best_n = metrics[0], n

    port, trade_log, metrics = momentum_strategy(values, best_n, True)
    ann_ret, mdd, calmar_ratio, sharpe_ratio, trade_times, win_rate = metrics

    # 准备绘图数据
    buy_dates, buy_prices, sell_dates, sell_prices = [], [], [], []
    for i, (idx, price) in enumerate(trade_log):
        date = dates[idx]
        if idx == -1:
            sse = get_calendar("SSE")
            date = sse.schedule(dates[-1], "2030-12-31").index[1]

        if i % 2 == 0:  # 买入点
            buy_dates.append(date)
            buy_prices.append(port[idx])
        else:  # 卖出点
            sell_dates.append(date)
            sell_prices.append(port[idx])
    buy_dates = np.array(buy_dates)
    buy_prices = np.array(buy_prices)
    sell_dates = np.array(sell_dates)
    sell_prices = np.array(sell_prices)
    drawdown_values = port / np.maximum.accumulate(port) - 1

    # 图 1: 策略净值
    p1 = figure(
        title="策略优化",
        x_axis_label="日期",
        y_axis_label="净值",
        x_axis_type="datetime",
        width=800,
        height=350,
    )

    line1 = p1.line(
        dates, values, legend_label="基础资产", line_width=2, color="gray", alpha=0.7
    )
    line2 = p1.line(
        dates, port, legend_label="策略净值", line_width=2, color="red", alpha=0.7
    )
    buy_scatter = p1.scatter(
        buy_dates,
        buy_prices,
        marker="triangle",
        size=7,
        color="red",
        legend_label="买入点",
        alpha=0.7,
    )
    sell_scatter = p1.scatter(
        sell_dates,
        sell_prices,
        marker="inverted_triangle",
        size=7,
        color="green",
        legend_label="卖出点",
        alpha=0.7,
    )

    hover1 = HoverTool(
        tooltips=[("日期", "@x{%F}"), ("净值", "@y{0.000}")],
        formatters={"@x": "datetime"},
        renderers=[line1, line2],
        mode="vline",
    )
    hover_buy = HoverTool(
        tooltips=[("买入价", "@y{0.000}")],
        renderers=[buy_scatter],
    )
    hover_sell = HoverTool(
        tooltips=[("卖出价", "@y{0.000}")],
        renderers=[sell_scatter],
    )
    crosshair1 = CrosshairTool(dimensions="height", line_color="gray", line_alpha=0.5)
    p1.add_tools(hover1, hover_buy, hover_sell, crosshair1)

    p1.legend.location = "top_left"
    p1.legend.click_policy = "hide"

    # 图 2: 策略回撤
    p2 = figure(
        title="策略回撤",
        x_axis_label="日期",
        y_axis_label="回撤",
        x_axis_type="datetime",
        width=800,
        height=250,
    )

    varea = p2.varea(
        x=dates, y1=0, y2=drawdown_values, color="red", alpha=0.5, legend_label="回撤"
    )
    p2.line(dates, np.zeros(len(dates)), line_width=1, color="black", alpha=0.5)

    hover2 = HoverTool(
        tooltips=[("日期", "@x{%F}"), ("回撤", "@y2{0.00%}")],
        formatters={"@x": "datetime"},
        renderers=[varea],
    )
    crosshair2 = CrosshairTool(dimensions="height", line_color="gray", line_alpha=0.5)
    p2.add_tools(hover2, crosshair2)

    p2.legend.location = "bottom_right"

    # 链接 x 轴
    p2.x_range = p1.x_range

    metrics_text = (
        f"{metrics_text_0}\nmomentum({best_n}):\n"
        f"年化收益: {ann_ret:.2%} 最大回撤: {mdd:.2%}\n"
        f"卡玛比率: {calmar_ratio:.2f} 夏普比率: {sharpe_ratio:.2f}\n"
        f"交易次数: {trade_times} 胜率: {win_rate:.2%}"
    )
    metrics_div = Div(
        text=metrics_text.replace("\n", "<br>"),
        width=800,
        height=100,
        styles={
            "font-size": "14px",
            "font-family": "Helvetica, Arial, sans-serif",
            "white-space": "pre",
        },
    )

    layout = column(p1, p2, metrics_div)
    html = (
        file_html(layout, resources=CDN)
        .replace(
            "https://cdn.bokeh.org/bokeh/release/bokeh-3.8.0.min.js",
            "https://cdnjs.cloudflare.com/ajax/libs/bokeh/3.8.0/bokeh.min.js",
        )
        .replace(
            "display: flow-root;",
            "display: flex;justify-content: center;",
        )
    )

    with open("momentum.html", "w", encoding="utf-8") as f:
        f.write(html)
