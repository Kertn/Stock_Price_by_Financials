import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
import optuna
from scipy import stats
from datetime import date
from warnings import simplefilter
from sklearn import preprocessing
from train_models import *
from estimate_func import estimate_annualy_income, estimate_annualy_income_test

def drop_incorrect_column(df, ticker):
    pred = '00-00-00'
    for i in df.columns.tolist():
        split = str(i).split('-')
        #print(split)
        if split[0] == pred[0] and split[1] != pred[1]:
            #print('DELETE IT - ', ticker)
            #print('str(i).split()[0]', str(i).split()[0])
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
            #print("Missing data is more than 80 days - ", ticker)
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
        #print("Ticker hasn't been found - ", ticker)
        return 0

    if len(list(reversed(stock_price[::52].Open.values))) == 0:
        #print("Data doesn't exist - ", ticker)
        return 0

    # Check the date_missing function
    if not history_date_missing(df, stock_price, ticker):
        return 0

    Stock_Price = list(reversed(stock_price[::52].Open.values))

    df = df.T


    if len(Stock_Price) != df.shape[0]:
        #print('insert end - ', ticker)
        Stock_Price.insert(0, stock_price.Open.values[-1])

    try:
        df.insert(0, 'Stock_Price', Stock_Price)
    except:
        #print("Ticker or history doesn't work - ", ticker)
        return 0

    df = df.fillna(-1)
    df = df.astype(float)

    df.insert(0, 'Ticker', [f'{ticker_count+1}.'+str(i) for i in reversed(range(1, len(df)+1))])

    return df

def add_miss_values(df):
    for col_name in df.columns:
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
        df[col_name] = column
    #print('Final Shape', df.shape)
    return df

def remove_most_miss(df, miss_data_column_allowed, miss_data_row_allowed):
    df_drop = df.copy()
    for column_name in df.columns:
        df_1 = df_drop[column_name]
        miss_values = len(df_1[df_1 == -1])
        len_data = len(df_1)
        if miss_values > len_data * miss_data_column_allowed:
            df.drop(column_name, axis='columns', inplace=True)

    df.index = list(range(0, df.shape[0]))
    df_drop = df.copy()
    for i in range(df_drop.shape[0]):
        df_1 = df_drop.iloc[i]
        miss_values = len(df_1[df_1 == -1])

        #Check does len(df.columns) work correct
        if miss_values > len(df.columns) * miss_data_row_allowed:
            df.drop(df_1.name, inplace=True)

    df.index = list(range(0, df.shape[0]))

    return df

def disp_best_corr(corr_df, df_initial, df):
    #print('index 0', corr_df.index[1])
    sns.regplot(x=corr_df.index[1], y='Stock_Price', data=df_initial)
    pearson_coef, p_value = stats.pearsonr(df_initial[corr_df.index[1]], df['Stock_Price'])
    #print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
    plt.show()

#def preprocess(df_initial, nlarge, miss_data_column_allowed, miss_data_row_allowed):
def preprocess(optuna):

    # General

    miss_data_column_allowed = optuna.suggest_float('miss_data_column_allowed', 0.05, 0.8, step=0.01)
    miss_data_row_allowed = optuna.suggest_float('miss_data_row_allowed', 0.05, 0.8, step=0.01)
    price_discount = optuna.suggest_float('price_discount', 0.01, 0.99, step=0.02)
    nlarge = optuna.suggest_int('nlarge', 20, 120, 5)


    # PyTorch

    # num_of_epochs = optuna.suggest_int('num_of_epochs', 10000, 250000, 5000)
    # lr = optuna.suggest_float('lr', 0.001, 0.1, step=0.001)

    # RandomForest

    bootstrap = optuna.suggest_categorical("bootstrap", [True, False])
    max_depth = optuna.suggest_int('max_depth', 10, 200, 10)
    max_features = optuna.suggest_categorical("max_features", ['sqrt', 'log2'])
    min_samples_leaf = optuna.suggest_int('min_samples_leaf', 1, 6, 1)
    min_samples_split = optuna.suggest_int('min_samples_split', 2, 12, 2)
    n_estimators = optuna.suggest_int('n_estimators', 200, 6000, 200)

    random_forest_dict = {'bootstrap': bootstrap,
                          'max_depth': max_depth,
                          'max_features': max_features,
                          'min_samples_leaf': min_samples_leaf,
                          'min_samples_split': min_samples_split,
                          'n_estimators': n_estimators}

    # print('Shape before', df_initial.shape)
    # df_initial = df_initial.fillna(-1)

    # # Save downloaded data
    #df_initial.to_csv('Full_list_collected.csv', index=False, encoding='utf-8')
    df_initial = pd.read_csv('Full_list_collected.csv')[:30]
    # print('Shape before', df_initial.shape)

    df = df_initial
    #df = df_initial.drop("Ticker", axis='columns')

    df_remove = remove_most_miss(df, miss_data_column_allowed, miss_data_row_allowed)

    Ticker_col = df_remove['Ticker']
    df_remove = df_remove.drop("Ticker", axis='columns')

    df = add_miss_values(df_remove)

    corr_df = df_remove.corr()['Stock_Price'].abs().nlargest(n=nlarge)

    #disp_best_corr(corr_df, df_initial, df)

    df_final = df[corr_df.index]

    df_final = pd.concat([df_final, Ticker_col], axis=1)

    print('Shinemachuynia', df_final.groupby(df_final['Ticker'].astype(int))['Ticker'].head(100))

    print('AAAAAAAAAAAAAAA', df_final.groupby(df_final['Ticker'].astype(int))['Ticker'].first().astype(str))

    dataf = df_final.groupby(df_final['Ticker'].astype(int)).first()

    df_group = dataf[dataf['Ticker'].astype(str).str.endswith('1')]

    print('df_group', df_group)
    df_final = pd.concat([df_final, df_group]).drop_duplicates(keep=False)

    test_df = df_final.groupby(df_final['Ticker'].astype(int)).first()
    print('df_final', df_final['Ticker'].head(100))
    print('test_df', test_df['Ticker'].head(100))
    train_df = pd.concat([df_final, test_df]).drop_duplicates(keep=False)


    train_df = train_df.sort_values('Ticker')

    y_train = train_df['Stock_Price'].to_numpy()
    test_df = test_df[test_df['Ticker'].astype(int).apply(lambda x: x in train_df['Ticker'].astype(int).values)]
    print('CHTIETO', test_df[['Stock_Price', 'Ticker']].head(130))
    y_test = test_df['Stock_Price']

    x_train = preprocessing.normalize(train_df.drop(['Stock_Price', 'Ticker'], axis=1).to_numpy())
    x_test = preprocessing.normalize(test_df.drop(['Stock_Price', 'Ticker'], axis=1).to_numpy())

    # test_ticker_coll = test_df['Ticker']
    #test_price_coll = test_df['Stock_Price']


    # BayesianRidg(X_train, X_test, y_train, y_test)
    # GradBoostRegr(X_train, X_test, y_train, y_test)
    # CatBoost(X_train, X_test, y_train, y_test)
    # XgBoost(X_train, X_test, y_train, y_test)
    # random_forest(X_train, X_test, y_train, y_test)

    all_models = []
    try:

        #all_models.append(NeuralNetTorch(x_train, y_train, x_test, y_test, num_of_epochs, lr))
        #all_models.append(NeuralNetTorch(x_train, y_train, x_test, y_test))
        # all_models.append(BayesianRidg(X, Y))
        # all_models.append(GradBoostRegr(X, Y))
        # all_models.append(XgBoost(X, Y))
        all_models.append(random_forest(x_train, y_train, x_test, y_test, random_forest_dict))

        answ = estimate_annualy_income_test(all_models, train_df, y_test, price_discount=price_discount, bear_inv=False)
        #answ = estimate_annualy_income_test(all_models, train_df, y_test, test_ticker_coll, price_discount=0.8, bear_inv=False)
    except:
        return -1
    return answ


    #TODO Создать визуализицию прибыли каждой модели !! + Проверить где больше успех моделей на рынке быков или медведей? + Проверит зависимость качества к количество features И процента жажды выгоды к actual income !!!

    #TODO Сделать тоже самое для TEST DATA !!!


def main():
    pd.options.mode.chained_assignment = None
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
    df = pd.DataFrame()
    total = 0

    # for ticker_count, i in tqdm(enumerate(pd.read_csv(fr'C:\Program\Neural_Network\Market_Ratios_Model\Full_list.csv', encoding='utf-8').values)):
    #     answ = data(i[0], ticker_count)
    #     if answ is not 0:
    #         df = pd.concat([df, answ])
    #     else:
    #         total += 1
    # print('total erors', total)

    # preprocess(df_initial=df, nlarge=110, miss_data_column_allowed=0.46, miss_data_row_allowed=0.07)

    opt_st = optuna.create_study(study_name='NeurNet',
                                 direction='maximize')
    opt_st.optimize(preprocess, n_trials=130, n_jobs=1)

    optuna.visualization.plot_param_importances(opt_st, target_name="f1_score")

if __name__ == '__main__':
    main()