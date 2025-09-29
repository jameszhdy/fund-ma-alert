# test_fetch.py
import datetime as dt
from run_alerts import fetch_single_fund_nav

code = "000218"
today = dt.date.today()
start_date = (today - dt.timedelta(days=600)).strftime("%Y%m%d")
end_date = today.strftime("%Y%m%d")
s = fetch_single_fund_nav(code, start_date, end_date)
print(code, "拿到数据行数:", len(s))
print(s.tail(5))
