#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_alerts.py
- 从 config.json 读取组合
- 用东财 pingzhongdata 抓历史净值（主），若失败则用新浪页面抓（备）
- 计算组合净值、180 日移动平均；若最新净值 < MA180 则发送推送（Server酱 / 企业微信 webhook）
"""
import os
import json
import time
import re
import argparse
import datetime as dt
import logging
from typing import List
from io import StringIO

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

LOG = logging.getLogger("fund_alerts")
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
LOG.addHandler(handler)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(THIS_DIR, "config.json")

HEADERS_COMMON = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
}

def load_config(path=CONFIG_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到配置文件：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ms_to_date(ms:int):
    return pd.to_datetime(ms, unit='ms').normalize()

# ---------------- 数据源 A：东财 pingzhongdata ----------------
def fetch_from_eastmoney(code:str, start_date:str, end_date:str, timeout=8):
    url = f"https://fund.eastmoney.com/pingzhongdata/{code}.js?v={int(time.time()*1000)}"
    headers = HEADERS_COMMON.copy()
    headers["Referer"] = f"https://fund.eastmoney.com/{code}.html"
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        text = r.text

        # 累计净值
        m_ac = re.search(r'var\s+Data_ACWorthTrend\s*=\s*(\[.*?\])\s*;', text, flags=re.S)
        if m_ac:
            arr = json.loads(m_ac.group(1))
            dates = [ms_to_date(int(item[0])) for item in arr]
            vals = [float(item[1]) for item in arr]
            return pd.Series(vals, index=pd.to_datetime(dates), name=code).sort_index()

        # 单位净值
        m_net = re.search(r'var\s+Data_netWorthTrend\s*=\s*(\[.*?\])\s*;', text, flags=re.S)
        if m_net:
            arr = json.loads(m_net.group(1))
            dates = [ms_to_date(int(item["x"])) for item in arr]
            vals = [float(item["y"]) for item in arr]
            return pd.Series(vals, index=pd.to_datetime(dates), name=code).sort_index()

        raise RuntimeError("EastMoney: 未匹配到净值数据")
    except Exception as e:
        LOG.debug("EastMoney fetch error for %s: %s", code, e)
        raise

# ---------------- 数据源 B：新浪页面解析 ----------------
def fetch_from_sina(code:str, start_date:str, end_date:str, timeout=8):
    url = f"https://stock.finance.sina.com.cn/fundInfo/view/FundInfo_LSJZ.php?symbol={code}"
    headers = HEADERS_COMMON.copy()
    headers["Referer"] = "https://finance.sina.com.cn/"
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        html = r.text

        # 先尝试 pandas.read_html
        dfs = []
        try:
            dfs = pd.read_html(StringIO(html))
        except Exception:
            pass

        target_df = None
        for df in dfs:
            cols = [str(c) for c in df.columns]
            if any("净值" in c for c in cols):
                target_df = df
                break

        # 如果 pandas 没解析成功，用 BeautifulSoup
        if target_df is None:
            soup = BeautifulSoup(html, "lxml")
            tables = soup.find_all("table")
            for t in tables:
                try:
                    df_try = pd.read_html(StringIO(str(t)))[0]
                    cols = [str(c) for c in df_try.columns]
                    if any("净值" in c for c in cols):
                        target_df = df_try
                        break
                except Exception:
                    continue

        if target_df is None or target_df.empty:
            raise RuntimeError("新浪未解析到净值表格")

        # 拍平多级表头
        if isinstance(target_df.columns, pd.MultiIndex):
            target_df.columns = ["".join([str(c) for c in tup]) for tup in target_df.columns]

        # 寻找日期/净值列
        date_col, nav_col = None, None
        for c in target_df.columns:
            s = str(c)
            if any(k in s for k in ["日期", "时间"]):
                date_col = c
            if any(k in s for k in ["累计净值", "单位净值", "基金净值", "净值"]):
                nav_col = c
        if not date_col or not nav_col:
            raise RuntimeError("新浪解析：未找到日期列或净值列")

        target_df = target_df[[date_col, nav_col]].dropna()
        target_df[date_col] = pd.to_datetime(target_df[date_col])
        target_df[nav_col] = pd.to_numeric(target_df[nav_col].astype(str).str.replace('%',''), errors='coerce')
        series = pd.Series(target_df[nav_col].values, index=target_df[date_col].dt.normalize(), name=code).sort_index()
        sdate, edate = pd.to_datetime(start_date), pd.to_datetime(end_date)
        return series[(series.index >= sdate) & (series.index <= edate)]
    except Exception as e:
        LOG.debug("Sina fetch error for %s: %s", code, e)
        raise

# ---------------- 单只基金统一抓取 ----------------
def fetch_single_fund_nav(code:str, start_date:str, end_date:str, retries=2):
    last_err = None
    for attempt in range(retries):
        try:
            return fetch_from_eastmoney(code, start_date, end_date)
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    for attempt in range(retries):
        try:
            return fetch_from_sina(code, start_date, end_date)
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    raise RuntimeError(f"未能获取基金 {code} 的净值数据 ({last_err})")

# ---------------- 批量抓取 ----------------
def fetch_nav_for_codes(codes:List[str], start_date:dt.date, end_date:dt.date):
    start_s, end_s = start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")
    series_list = []
    for c in codes:
        LOG.info("拉取基金 %s 净值：%s -> %s", c, start_s, end_s)
        try:
            s = fetch_single_fund_nav(c, start_s, end_s)
            series_list.append(s.rename(c))
            time.sleep(0.2)
        except Exception as e:
            LOG.error(" 拉取 %s 失败: %s", c, e)
            continue
    if not series_list:
        raise ValueError("没有任何基金成功获取数据。")
    df = pd.concat(series_list, axis=1).sort_index()
    df.index = pd.to_datetime(df.index).normalize()
    return df

# ---------------- 推送 ----------------
def send_serverchan(sckey: str, title: str, desp: str) -> bool:
    url = f"https://sctapi.ftqq.com/{sckey}.send"
    payload = {"title": title, "desp": desp}
    try:
        r = requests.post(url, json=payload, timeout=8)
        LOG.info("Server酱返回: %s %s", r.status_code, r.text)
        return r.status_code == 200
    except Exception as e:
        LOG.error("Server酱发送失败: %s", e)
        return False

def send_enterprise_wechat(webhook_url: str, text: str) -> bool:
    payload = {"msgtype": "text", "text": {"content": text}}
    headers = {"Content-Type": "application/json"}
    try:
        r = requests.post(webhook_url, json=payload, headers=headers, timeout=8)
        LOG.info("企业微信返回: %s %s", r.status_code, r.text)
        return r.status_code == 200
    except Exception as e:
        LOG.error("企业微信发送失败: %s", e)
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
        LOG.warning("未发送任何提醒（未配置或发送失败）。")

# ---------------- 主逻辑 ----------------
def main(debug=False):
    if debug:
        LOG.setLevel(logging.DEBUG)

    cfg = load_config()
    roll_window = int(cfg.get("roll_window", 180))
    fetch_days_back = int(cfg.get("fetch_days_back", 800))
    default_cap = float(cfg.get("initial_capital_default", 100000.0))
    portfolios = cfg.get("portfolios", {})

    fund_codes = sorted({c for p in portfolios.values() for c in p.get("funds", {})})
    LOG.info("共 %d 只基金，代码示例：%s", len(fund_codes), fund_codes[:6])

    today = dt.date.today()
    start_date, end_date = today - dt.timedelta(days=fetch_days_back), today
    LOG.info("拉取区间: %s ~ %s", start_date, end_date)

    nav_df = fetch_nav_for_codes(fund_codes, start_date, end_date)
    LOG.info("获取到净值表：行 %d 列 %d", len(nav_df), len(nav_df.columns))

    alerts = []
    for pname, pconf in portfolios.items():
        LOG.info("处理组合：%s", pname)
        funds = pconf.get("funds", {})
        if not funds:
            LOG.warning("%s 没有 funds，跳过", pname)
            continue
        codes = list(funds.keys())
        sub_nav = nav_df.reindex(columns=codes).ffill().dropna(how="any")
        if sub_nav.empty:
            LOG.warning("%s 的数据不足，跳过", pname)
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

        LOG.info("%s 最后交易日 %s NAV=%f MA%d=%s", pname, last_date, last_nv, roll_window, str(last_ma))
        if last_ma and last_nv < last_ma:
            pct_gap = (last_nv - last_ma) / last_ma
            title = f"[预警] 组合 {pname} 低于 MA{roll_window}"
            body = (f"组合: {pname}\n日期: {last_date}\nNAV: {last_nv:.6f}\nMA{roll_window}: {last_ma:.6f}\n"
                    f"低于幅度: {pct_gap:.2%}\n\n持仓明细:\n")
            for c in codes:
                body += f"  {c}: weight={weights[c]:.2%}, latest_nav={sub_nav[c].iloc[-1]:.6f}\n"
            alerts.append((title, body))

    if alerts:
        for t, b in alerts:
            send_notifications(t, b)
    else:
        LOG.info("无需要报警的组合。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="输出调试日志")
    args = parser.parse_args()
    main(debug=args.debug)
