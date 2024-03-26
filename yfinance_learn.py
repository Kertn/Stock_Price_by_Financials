# -*- coding: utf8 -*-

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import preprocessing
import optuna
from scipy import stats
from datetime import date
from warnings import simplefilter
from train_models import *
from Optuna_search import Optuna_Optim, Invest
from estimate_func import estimate_annualy_income_test

def drop_incorrect_column(df, ticker):
    pred = '00-00-00'
    for i in df.columns.tolist():
        split = str(i).split('-')
        if split[0] == pred[0] and split[1] != pred[1]:
            df = df.drop(str(i).split()[0], axis=1)
        pred = split
    return df

def history_date_missing(df, stock_price, ticker):
    for i in [-1, 0]:
        split = str(df.columns.tolist()[i]).split('-')
        split[-1] = split[-1].split()[0]
        split_stock = str(list(reversed(stock_price.index))[i]).split('-')[0:3]
        split_stock[-1] = split_stock[-1].split()[0]
        if abs((date(int(split[0]), int(split[1]), int(split[2])) - date(int(split_stock[0]), int(split_stock[1]), int(split_stock[2]))).days) > 80:
            return 0
    return 1
def data(ticker, ticker_count):
    company = yf.Ticker(ticker)

    fin = company.financials
    balance = company.balancesheet
    cash = company.cashflow

    all = pd.concat([fin, balance, cash])

    df = pd.DataFrame(all)
    pd.set_option('display.max_rows', None)

    # Drop Function
    df = drop_incorrect_column(df, ticker)

    try:
        stock_price = company.history(interval='1wk', start=sorted(df.columns)[0], end=sorted(df.columns)[-1])
    except:
        return 0

    if len(list(reversed(stock_price[::52].Open.values))) == 0:
        return 0

    # Check the date_missing function
    if not history_date_missing(df, stock_price, ticker):
        return 0

    Stock_Price = list(reversed(stock_price[::52].Open.values))

    df = df.T


    if len(Stock_Price) != df.shape[0]:
        Stock_Price.insert(0, stock_price.Open.values[-1])

    try:
        df.insert(0, 'Stock_Price', Stock_Price)
    except:
        return 0

    df = df.fillna(-1)
    df = df.astype(float)

    df.insert(0, 'Ticker', [f'{ticker_count+1}.'+str(i) for i in reversed(range(1, len(df)+1))])

    return df

def main():
    pd.options.mode.chained_assignment = None
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    # all_filters_system = ['Aerospace', 'Auto_Tires_Trucks', 'Basic_Materials', 'Business_Services', 'Computer_and_Technology', 'Construction', 'Consumer_Discretionary', 'Consumer_Staples', 'Finance', 'Industrial_Products', 'Medical_Multi-Sector_Conglomerates', 'Oils_Energy', 'Retail_Wholesale', 'Transportation', 'Unclassified', 'Utilities', 'Full_list']
    #
    # for name in all_filters_system:
    #     df = pd.DataFrame()
    #     total = 0
    #     for ticker_count, i in tqdm(enumerate(pd.read_csv(fr'C:\Program\Neural_Network\Market_Ratios_Model\All_lists\{name}.csv', encoding='utf-8').values)):
    #         answ = data(i[0], ticker_count)
    #         if answ is not 0:
    #             df = pd.concat([df, answ])
    #         else:
    #             total += 1
    #     print('total erors', total)
    #     df = df.fillna(-1)
    #     df.to_csv(fr'C:\Program\Neural_Network\Market_Ratios_Model\All_lists_collected\{name}_collected.csv', index=False)
    Optuna_Optim()
    #Invest()

if __name__ == '__main__':
    main()