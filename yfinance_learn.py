import yfinance as yf
import pandas as pd

gvp = yf.Ticker("GVP")
apple = yf.Ticker("aapl")


info = apple.info

print(info.keys())

df = pd.DataFrame(apple.get_financials())


print(df.head())