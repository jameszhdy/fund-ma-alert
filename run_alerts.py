#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_alerts.py
按 config.json 中的组合，用 AKShare 拉基金净值，计算组合净值与 180 日 MA，
若最新净值 < MA 则发送推送（Server酱 或 企业微信 webhook）。
可在本地用 --debug 测试。
"""
import os
import json
import argparse
import datetime as dt
import logging
import time
from typing import Dict, List

import pandas as pd
import numpy as np
import requests

# 尝试导入 akshare
try:
    import akshare as ak
except Exception as e:
    raise ImportError("请先安装 akshare（pip install akshare）: %s" % e)

LOG = logging.getLogger("fund_alerts")
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
LOG.addHandler(handler)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(THIS_DIR, "config.json")

def load_config(path=CONFIG_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到配置文件：{path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg

def date_to_ak(d: dt.date):
    return d.strftime("%Y%m%d")

def fetch_single_fund_nav(code: str, start_date: str, end_date: str) -> pd.Series:
    LOG.info(f"拉取基金 {code} 净值：{start_date} -> {end_date}")
    candidates = [
        "fund_open_fund_daily_em",
        "fund_open_fund_info_em",
        "fund_em_open_fund_daily",
        "fund_em_open_fund_info"
    ]
    df = None
    for name in candidates:
        func = getattr(ak, name, None)
        if func is None:
            continue
        try:
            # 两种常见调用方式尝试
            try:
                df_try = func(fund=code, start_date=start_date, end_date=end_date)
            except TypeError:
                try:
                    df_try = func(code, start_date, end_date)
                except Exception:
                    df_try = None
            except Exception:
                df_try = None

            if df_try is None or df_try.empty:
                continue
            df = df_try.copy()
            LOG.info(f"使用 akshare 的 {name} 获取到数据，行数 {len(df)}")
            break
        except Exception as e:
            LOG.warning(f"尝试 {name} 失败: {e}")
            continue

    if df is None or df.empty:
        raise ValueError(f"未能通过 akshare 获取到基金 {code} 的净值数据（尝试了多个接口）。")

    # 找日期列
    date_col = None
    for c in ["净值日期", "date", "日期", "statistic_date"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        if isinstance(df.index, pd.DatetimeIndex):
            idx = pd.to_datetime(df.index)
        else:
            raise ValueError("未找到日期列，数据表字段为: %s" % (list(df.columns),))
    else:
        idx = pd.to_datetime(df[date_col])

    # 找净值列（优先累计净值）
    nav_col = None
    for c in ["累计净值", "累计净值(元)", "accumulative_net_value", "sum_value"]:
        if c in df.columns:
            nav_col = c
            break
    if nav_col is None:
        for c in ["单位净值", "单位净值(元)", "net_value"]:
            if c in df.columns:
                nav_col = c
                break
    if nav_col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            nav_col = numeric_cols[0]
        else:
            raise ValueError(f"未能识别净值列，字段为: {list(df.columns)}")

    series = pd.Series(df[nav_col].values, index=pd.to_datetime(idx).normalize(), name=code)
    series = series[~series.isna()].sort_index()
    series = series[~series.index.duplicated(keep='last')]
    if series.empty:
        raise ValueError(f"基金 {code} 拉取后净值序列为空。")
    return series

def fetch_nav_for_codes(codes: List[str], start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    start_s = date_to_ak(start_date)
    end_s = date_to_ak(end_date)
    series_list = []
    for c in codes:
        try:
            s = fetch_single_fund_nav(c, start_s, end_s)
            series_list.append(s.rename(c))
            time.sleep(0.5)
        except Exception as e:
            LOG.error(f"拉取 {c} 失败: {e}")
            raise
    df = pd.concat(series_list, axis=1)
    df = df.sort_index()
    return df

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
    headers = {"Content-Type": "application/json"}
    try:
        r = requests.post(webhook_url, json=payload, headers=headers, timeout=10)
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
        LOG.info("使用 Server酱 发送提醒")
        ok = send_serverchan(sckey, title, body)
    if not ok and wechat_webhook:
        LOG.info("使用 企业微信 webhook 发送提醒")
        ok = send_enterprise_wechat(wechat_webhook, f"{title}\n\n{body}")
    if not ok:
        LOG.warning("未发送任何提醒（未配置或发送失败）。请检查 SERVERCHAN_SCKEY / WECHAT_WEBHOOK 是否设置正确。")

def main(debug=False):
    cfg = load_config()
    roll_window = int(cfg.get("roll_window", 180))
    fetch_days_back = int(cfg.get("fetch_days_back", 600))
    default_cap = float(cfg.get("initial_capital_default", 100000.0))
    portfolios = cfg.get("portfolios", {})
    if not portfolios:
        LOG.error("配置文件中没有发现任何组合（portfolios）。")
        return

    fund_codes = set()
    for p, v in portfolios.items():
        funds = v.get("funds", {})
        for code in funds.keys():
            fund_codes.add(code)
    fund_codes = sorted(list(fund_codes))
    LOG.info(f"共 {len(fund_codes)} 只基金，代码示例：{fund_codes[:6]}")

    today = dt.date.today()
    start_date = today - dt.timedelta(days=fetch_days_back)
    end_date = today
    LOG.info(f"拉取区间: {start_date} ~ {end_date}")

    nav_df = fetch_nav_for_codes(fund_codes, start_date, end_date)
    LOG.info(f"获取净值表：行 {len(nav_df)} 列 {len(nav_df.columns)}")

    alerts = []
    for pname, pconf in portfolios.items():
        LOG.info(f"处理组合：{pname}")
        funds = pconf.get("funds", {})
        if not funds:
            LOG.warning(f"{pname} 没有 funds 条目，跳过")
            continue
        codes = list(funds.keys())
        sub_nav = nav_df.loc[:, codes].dropna(how="any")
        if sub_nav.empty:
            LOG.warning(f"{pname} 的基金在相同交易日上的交集为空，尝试用向前填充后再计算")
            # 备用方案：外连接并向前填充（如果你想容忍少量缺失）
            sub_nav = nav_df.loc[:, codes].ffill().dropna(how="any")
            if sub_nav.empty:
                LOG.warning(f"备用填充后仍为空，跳过 {pname}")
                continue

        weights = pd.Series(funds, dtype=float)
        ws = float(weights.sum())
        if abs(ws - 1.0) > 1e-8:
            LOG.info(f"{pname} 权重和为 {ws:.6f}，将自动归一化")
            weights = weights / ws

        initial_cap = float(pconf.get("initial_capital", default_cap))
        start_prices = sub_nav.iloc[0]
        units = {}
        for c in codes:
            price0 = start_prices[c]
            units[c] = initial_cap * float(weights[c]) / float(price0)
        units_s = pd.Series(units)

        portfolio_value = (sub_nav * units_s).sum(axis=1)
        portfolio_nv = portfolio_value / float(portfolio_value.iloc[0])

        ma_port = portfolio_nv.rolling(roll_window, min_periods=roll_window).mean()
        last_date = pd.to_datetime(portfolio_nv.index[-1]).date()
        last_nv = float(portfolio_nv.iloc[-1])
        last_ma = float(ma_port.iloc[-1]) if not pd.isna(ma_port.iloc[-1]) else None

        LOG.info(f"{pname} 最后交易日 {last_date} NAV={last_nv:.6f} MA{roll_window}={last_ma if last_ma is not None else 'N/A'}")

        if last_ma is None or np.isnan(last_ma):
            LOG.info(f"{pname} 的 MA{roll_window} 尚未计算（数据不足），跳过报警判断。")
            continue

        if last_nv < last_ma:
            pct_gap = (last_nv - last_ma) / last_ma
            msg_title = f"[预警] 组合 {pname} 低于 MA{roll_window}"
            msg_body = (
                f"组合: {pname}\n"
                f"日期: {last_date}\n"
                f"NAV: {last_nv:.6f}\n"
                f"MA{roll_window}: {last_ma:.6f}\n"
                f"低于幅度: {pct_gap:.2%}\n\n"
                "持仓明细:\n"
            )
            for c in codes:
                msg_body += f"  {c}: weight={weights[c]:.3f}, latest_nav={sub_nav[c].iloc[-1]:.6f}\n"
            alerts.append((msg_title, msg_body))
        else:
            LOG.info(f"{pname} 当前净值位于均线之上（无需报警）。")

    if alerts:
        for title, body in alerts:
            send_notifications(title, body)
    else:
        LOG.info("无需要报警的组合。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="打印调试信息")
    args = parser.parse_args()
    if args.debug:
        LOG.setLevel(logging.DEBUG)
    main(debug=args.debug)
