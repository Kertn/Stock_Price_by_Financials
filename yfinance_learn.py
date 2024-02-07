import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from datetime import date

def drop_incorrect_column(df, ticker):
    pred = '00-00-00'
    for i in df.columns.tolist():
        split = str(i).split('-')
        #print(split)
        if split[0] == pred[0] and split[1] != pred[1]:
            print('DELETE IT - ', ticker)
            print('str(i).split()[0]', str(i).split()[0])
            df = df.drop(str(i).split()[0], axis=1)
        pred = split
    return df

def data(ticker):
    company = yf.Ticker(ticker)

    fin = company.financials
    balance = company.balancesheet
    cash = company.cashflow

    all = pd.concat([fin, balance, cash])

    df = pd.DataFrame(all)
    pd.set_option('display.max_rows', None)

    pred = '00-00-00'
    for i in df.columns.tolist():
        split = str(i).split('-')
        #print(split)
        if split[0] == pred[0] and split[1] != pred[1]:
            print('DELETE IT - ', ticker)
            print('str(i).split()[0]', str(i).split()[0])
            df = df.drop(str(i).split()[0], axis=1)
        pred = split

    try:
        stock_price = company.history(interval='1wk', start=sorted(df.columns)[0], end=sorted(df.columns)[-1])
    except:
        print("Ticker hasn't been found - ", ticker)
        return 0


    if len(list(reversed(stock_price[::52].Open.values))) == 0:
        print("Data doesn't exist - ", ticker)
        return 0


    for i in [-1, 0]:
        split = str(df.columns.tolist()[i]).split('-')
        split[-1] = split[-1].split()[0]
        split_stock = str(list(reversed(stock_price.index))[i]).split('-')[0:3]
        split_stock[-1] = split_stock[-1].split()[0]
        if abs((date(int(split[0]), int(split[1]), int(split[2])) - date(int(split_stock[0]), int(split_stock[1]), int(split_stock[2]))).days) > 80:
            print("Missing data is more than 80 days - ", ticker)
            return 0

    Stock_Price = list(reversed(stock_price[::52].Open.values))

    #print(Stock_Price)

    #print(df.shape)

    df = df.T


    if len(Stock_Price) != df.shape[0]:
        print('insert end - ', ticker)
        Stock_Price.insert(0, stock_price.Open.values[-1])

    try:
        df.insert(0, 'Stock_Price', Stock_Price)
    except:
        print("Ticker or history doesn't work - ", ticker)
        return 0
    df = df.fillna(0)
    df = df.astype(float)

    return df

def main():
    all = []
    for i in tqdm(pd.read_csv(r'C:\Program\Neural_Network\Market_Ratios_Model\Full_list.csv', encoding='utf-8').values[0:400]):
        #print('i', i[0])
        all.append(data(i[0]))
    #print('all', all)
    print(len(all))
    total = 0
    for i in all:
        if isinstance(i, int):
            total += 1
    print('total erors', total)

if __name__ == '__main__':
    main()