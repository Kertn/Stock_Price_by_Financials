import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
from tqdm import tqdm
from scipy import stats
from datetime import date
from warnings import simplefilter

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

def history_date_missing(df, stock_price, ticker):
    for i in [-1, 0]:
        split = str(df.columns.tolist()[i]).split('-')
        split[-1] = split[-1].split()[0]
        split_stock = str(list(reversed(stock_price.index))[i]).split('-')[0:3]
        split_stock[-1] = split_stock[-1].split()[0]
        if abs((date(int(split[0]), int(split[1]), int(split[2])) - date(int(split_stock[0]), int(split_stock[1]), int(split_stock[2]))).days) > 80:
            print("Missing data is more than 80 days - ", ticker)
            return 0
    return 1
def data(ticker):
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
        print("Ticker hasn't been found - ", ticker)
        return 0

    if len(list(reversed(stock_price[::52].Open.values))) == 0:
        print("Data doesn't exist - ", ticker)
        return 0

    # Check the date_missing function
    if not history_date_missing(df, stock_price, ticker):
        return 0

    Stock_Price = list(reversed(stock_price[::52].Open.values))

    df = df.T


    if len(Stock_Price) != df.shape[0]:
        print('insert end - ', ticker)
        Stock_Price.insert(0, stock_price.Open.values[-1])

    try:
        df.insert(0, 'Stock_Price', Stock_Price)
    except:
        print("Ticker or history doesn't work - ", ticker)
        return 0

    df = df.fillna(-1)
    df = df.astype(float)
    df['Ticker'] = ticker

    return df

def add_miss_values(df):
    df.index = list(range(0, df.shape[0]))
    df_drop = df.copy()
    for i in range(df_drop.shape[0]):
        df_1 = df_drop.iloc[i]
        miss_values = len(df_1[df_1 == -1])
        if miss_values > 30:
            df.drop(df_1.name, inplace=True)
    df.index = list(range(0, df.shape[0]))

    for col_name in df.columns:
        print(col_name)
        df_train = df.loc[df[col_name] != -1]
        X = df_train['Stock_Price']
        df_train = df_train[col_name]
        Y_train = df_train.values.reshape(-1, 1)
        X_train = X.values.reshape(-1, 1)
        regr = LinearRegression()
        regr.fit(X_train, Y_train)
        column = []
        for miss_data, X_val in zip(df[col_name], df['Stock_Price']):
            if miss_data == -1:
                miss_data = int(regr.predict(np.array([X_val]).reshape(-1,1))[0][0])
            column.append(miss_data)
        # print(df[col_name])
        df[col_name] = column
        # print(df[col_name])
    print('Final shape', df.shape)
def regplot_corr(df, nlarge):
    print('Final Shape', df.shape)
    df = df.fillna(-1)
    df_new = df[['Stock_Price', 'Ticker', 'Normalized EBITDA']]
    df_pred_corr = df.drop("Ticker", axis='columns')


    corr_df = df_pred_corr.corr()['Stock_Price'].abs().nlargest(n=nlarge)

    df_pred = df_pred_corr[corr_df.index]

    add_miss_values(df_pred)

    # sns.regplot(x='Amortization', y='Stock_Price', data=df)
    # pearson_coef, p_value = stats.pearsonr(df['Amortization'], df['Stock_Price'])
    # print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
    # plt.show()

def main():
    pd.options.mode.chained_assignment = None
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
    df = pd.DataFrame()
    total = 0
    for i in tqdm(pd.read_csv(r'C:\Program\Neural_Network\Market_Ratios_Model\Aerospace.csv', encoding='utf-8').values[0:50]):
        answ = data(i[0])
        if answ is not 0:
            df = pd.concat([df, answ])
        else:
            total += 1
    print('total erors', total)
    regplot_corr(df, 100)

if __name__ == '__main__':
    main()