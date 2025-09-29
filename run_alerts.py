#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_alerts.py
- 从 config.json 读取组合
- 用 AKShare 拉基金净值（优先 fund_open_fund_daily_em，备用 fund_open_fund_info_em）
- 计算组合净值与 180 日 MA
- 如果组合净值跌破均线，则推送到微信（Server酱/企业微信）
"""

import os
import json
import argparse
import datetime as dt
import logging
import time
import pandas as pd
import numpy as np
import requests

try:
    import akshare as ak
except Exception as e:
    raise ImportError("请先安装 akshare: pip install akshare")

LOG = logging.getLogger("fund_alerts")
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
LOG.addHandler(handler)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(THIS_DIR, "config.json")


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def date_to_ak(d: dt.date):
    return d.strftime("%Y%m%d")


# --------------------
# 获取基金净值
# --------------------
def fetch_single_fund_nav(code: str, start_date: str, end_date: str) -> pd.Series:
    LOG.info(f"拉取基金 {code} 净值：{start_date} -> {end_date}")
    df = None

    # 先尝试日度净值接口
    try:
        df = ak.fund_open_fund_daily_em(symbol=code)
        if df is not None and not df.empty:
            df = df.rename(columns={"净值日期": "date", "单位净值": "nav"})
    except Exception:
        df = None

    # 如果失败，尝试带起止日期的接口
    if df is None or df.empty:
        try:
            df = ak.fund_open_fund_info_em(fund=code, start_date=start_date, end_date=end_date)
            if df is not None and not df.empty:
                if "累计净值" in df.columns:
                    df = df.rename(columns={"净值日期": "date", "累计净值": "nav"})
                elif "单位净值" in df.columns:
                    df = df.rename(columns={"净值日期": "date", "单位净值": "nav"})
        except Exception:
            df = None

    if df is None or df.empty:
        raise ValueError(f"未能获取基金 {code} 的净值数据")

    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]
    series = pd.Series(df["nav"].values, index=df["date"], name=code)
    series = series.sort_index()
    return series


def fetch_nav_for_codes(codes, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    start_s = date_to_ak(start_date)
    end_s = date_to_ak(end_date)
    series_list = []
    for c in codes:
        try:
            s = fetch_single_fund_nav(c, start_s, end_s)
            series_list.append(s.rename(c))
            time.sleep(0.3)
        except Exception as e:
            LOG.error(f"拉取 {c} 失败: {e}")
            continue
    if not series_list:
        raise ValueError("没有任何基金成功获取数据。")
    df = pd.concat(series_list, axis=1)
    df = df.sort_index()
    return df


# --------------------
# 推送通知
# --------------------
def send_serverchan(sckey: str, title: str, desp: str) -> bool:
    url = f"https://sctapi.ftqq.com/{sckey}.send"
    payload = {"title": title, "desp": desp}
    try:
        r = requests.post(url, json=payload, timeout=10)
        LOG.info("Server酱返回: %s %s" % (r.status_code, r.text))
        return r.status_code == 200
    except Exception as e:
        LOG.error("Server酱发送失败: %s" % e)
        return False


def send_enterprise_wechat(webhook_url: str, text: str) -> bool:
    payload = {"msgtype": "text", "text": {"content": text}}
    try:
        r = requests.post(webhook_url, json=payload, timeout=10)
        LOG.info("企业微信返回: %s %s" % (r.status_code, r.text))
        return r.status_code == 200
    except Exception as e:
        LOG.error("企业微信发送失败: %s" % e)
        return False


def send_notifications(title: str, body: str):
    sckey = os.environ.get("SERVERCHAN_SCKEY") or os.environ.get("SERVERCHAN_SCKEY_TURBO")
    wechat_webhook = os.environ.get("WECHAT_WEBHOOK")
    ok = False
    if sckey:
        ok = send_serverchan(sckey, title, body)
    if not ok and wechat_webhook:
        ok = send_enterprise_wechat(wechat_webhook, f"{title}\n\n{body}")
    if not ok:
        LOG.warning("未发送任何提醒，请检查配置。")


# --------------------
# 主逻辑
# --------------------
def main(debug=False):
    cfg = load_config()
    roll_window = int(cfg.get("roll_window", 180))
    fetch_days_back = int(cfg.get("fetch_days_back", 600))
    default_cap = float(cfg.get("initial_capital_default", 100000.0))
    portfolios = cfg.get("portfolios", {})

    fund_codes = sorted({c for p in portfolios.values() for c in p.get("funds", {})})
    LOG.info(f"共 {len(fund_codes)} 只基金，代码示例：{fund_codes[:6]}")

    today = dt.date.today()
    start_date = today - dt.timedelta(days=fetch_days_back)
    end_date = today
    LOG.info(f"拉取区间: {start_date} ~ {end_date}")

    nav_df = fetch_nav_for_codes(fund_codes, start_date, end_date)

    alerts = []
    for pname, pconf in portfolios.items():
        funds = pconf.get("funds", {})
        codes = list(funds.keys())
        sub_nav = nav_df.loc[:, codes].dropna(how="any")
        if sub_nav.empty:
            continue

        weights = pd.Series(funds, dtype=float)
        if abs(weights.sum() - 1.0) > 1e-8:
            weights = weights / weights.sum()

        initial_cap = float(pconf.get("initial_capital", default_cap))
        start_prices = sub_nav.iloc[0]
        units = {c: initial_cap * weights[c] / start_prices[c] for c in codes}
        units_s = pd.Series(units)

        portfolio_value = (sub_nav * units_s).sum(axis=1)
        portfolio_nv = portfolio_value / portfolio_value.iloc[0]
        ma_port = portfolio_nv.rolling(roll_window, min_periods=roll_window).mean()

        last_date = portfolio_nv.index[-1].date()
        last_nv = float(portfolio_nv.iloc[-1])
        last_ma = float(ma_port.iloc[-1]) if not pd.isna(ma_port.iloc[-1]) else None

        LOG.info(f"{pname}: {last_date} NAV={last_nv:.6f}, MA{roll_window}={last_ma}")

        if last_ma and last_nv < last_ma:
            pct_gap = (last_nv - last_ma) / last_ma
            msg_title = f"[预警] 组合 {pname} 低于 MA{roll_window}"
            msg_body = (
                f"组合: {pname}\n"
                f"日期: {last_date}\n"
                f"NAV: {last_nv:.6f}\n"
                f"MA{roll_window}: {last_ma:.6f}\n"
                f"低于幅度: {pct_gap:.2%}\n"
            )
            alerts.append((msg_title, msg_body))

    if alerts:
        for t, b in alerts:
            send_notifications(t, b)
    else:
        LOG.info("无需要报警的组合。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.debug:
        LOG.setLevel(logging.DEBUG)
    main(debug=args.debug)
