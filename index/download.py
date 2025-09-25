import glob
import akshare as ak
import pandas as pd

index_list = [
    "sh000001",  # 上证指数
    "sh000016",  # 上证50
    "sh000300",  # 沪深300
    "sh000905",  # 中证500
    "sh000906",  # 中证800
    "sh000852",  # 中证1000
    "sz399303",  # 国证2000
    "sz399006",  # 创业板指
    "sz399673",  # 创业板50
    "sh000680",  # 科创综指
    "sh000688",  # 科创50
]

saved_index_map = {}
for file in glob.glob("data/index/*.csv"):
    df = pd.read_csv(file, engine="pyarrow")
    saved_index_map[df["code"].iloc[0]] = df

new_index_map = {}
end_date = pd.to_datetime("today").strftime("%Y-%m-%d")

for code in index_list:
    if code in saved_index_map:
        start_date = str(saved_index_map[code]["date"].max())
        if start_date >= end_date:
            continue
    new_index_map[code] = ak.stock_zh_index_daily_em(symbol=code)
    print(code)

for code, df in new_index_map.items():
    saved_df = saved_index_map.get(code)
    if saved_df is not None:
        df = pd.concat([saved_df, df], ignore_index=True)

    df["date"] = pd.to_datetime(df["date"])
    df["code"] = code
    df = df.drop_duplicates().sort_values("date").reset_index(drop=True)
    df.to_csv(f"data/index/{code}.csv", index=False)
