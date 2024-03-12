from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import preprocessing
from train_models import *
from estimate_func import estimate_annualy_income_test

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

def optuna_models_params(optuna, model, sector_name):

    # General
    miss_data_column_allowed = optuna.suggest_float('miss_data_column_allowed', 0.05, 0.8, step=0.01)
    miss_data_row_allowed = optuna.suggest_float('miss_data_row_allowed', 0.05, 0.8, step=0.01)
    nlarge = optuna.suggest_int('nlarge', 10, 120, 5)

    df = pd.read_csv(rf'C:\Program\Neural_Network\Market_Ratios_Model\All_lists_collected\{sector_name}_collected.csv')

    df_remove = remove_most_miss(df, miss_data_column_allowed, miss_data_row_allowed)

    Ticker_col = df_remove['Ticker']

    df_remove = df_remove.drop("Ticker", axis='columns')
    try:
        df = add_miss_values(df_remove)
    except:
        return -100
    corr_df = df_remove.corr()['Stock_Price'].abs().nlargest(n=nlarge)


    df_final = df[corr_df.index]

    df_final = pd.concat([df_final, Ticker_col], axis=1)

    dataf = df_final.groupby(df_final['Ticker'].astype(int)).first()

    df_group = dataf[dataf['Ticker'].astype(str).str.endswith('1')]

    df_final = pd.concat([df_final, df_group]).drop_duplicates(keep=False)

    test_df = df_final.groupby(df_final['Ticker'].astype(int)).first()

    train_df = pd.concat([df_final, test_df]).drop_duplicates(keep=False)


    train_df = train_df.sort_values('Ticker')

    y_train = train_df['Stock_Price'].to_numpy()
    test_df = test_df[test_df['Ticker'].astype(int).apply(lambda x: x in train_df['Ticker'].astype(int).values)]
    y_test = test_df['Stock_Price']

    x_train = preprocessing.normalize(train_df.drop(['Stock_Price', 'Ticker'], axis=1).to_numpy())
    x_test = preprocessing.normalize(test_df.drop(['Stock_Price', 'Ticker'], axis=1).to_numpy())

    model, result = model(optuna, x_train, y_train, x_test, y_test, train=True, best_params=None)

    return result


    #TODO:  Создать визуал optuna обучения для full list, bull, bear markets for all models!!  Заполнить 2 exel таблицы
    #TODO: После всех обучений сделать предсказания моделей (обучить модель снова на тех же данных, и 2 - обучить на всех возмодных данных), объединить точки совпадения и создать таблицу для инвестирования, также попробовать и bear market, посмотреть на цены этих акций на данный момент и момент публикации отчета, просчитать процент прибыли такого портфеля


def optuna_price_disc(optuna, model, sector_name, best_params, bear_inv):
    print('Best_params', best_params)
    nlarge = best_params
    miss_data_column_allowed = best_params
    miss_data_row_allowed = best_params

    price_discount = optuna.suggest_float('price_discount', 0.01, 0.99, step=0.01)

    df = pd.read_csv(rf'C:\Program\Neural_Network\Market_Ratios_Model\All_lists_collected\{sector_name}_collected.csv')

    df_remove = remove_most_miss(df, miss_data_column_allowed, miss_data_row_allowed)

    Ticker_col = df_remove['Ticker']

    df_remove = df_remove.drop("Ticker", axis='columns')
    try:
        df = add_miss_values(df_remove)
    except:
        return -100
    corr_df = df_remove.corr()['Stock_Price'].abs().nlargest(n=nlarge)

    df_final = df[corr_df.index]

    df_final = pd.concat([df_final, Ticker_col], axis=1)

    dataf = df_final.groupby(df_final['Ticker'].astype(int)).first()

    df_group = dataf[dataf['Ticker'].astype(str).str.endswith('1')]

    df_final = pd.concat([df_final, df_group]).drop_duplicates(keep=False)

    test_df = df_final.groupby(df_final['Ticker'].astype(int)).first()

    train_df = pd.concat([df_final, test_df]).drop_duplicates(keep=False)

    train_df = train_df.sort_values('Ticker')

    y_train = train_df['Stock_Price'].to_numpy()
    test_df = test_df[test_df['Ticker'].astype(int).apply(lambda x: x in train_df['Ticker'].astype(int).values)]
    y_test = test_df['Stock_Price']

    x_train = preprocessing.normalize(train_df.drop(['Stock_Price', 'Ticker'], axis=1).to_numpy())
    x_test = preprocessing.normalize(test_df.drop(['Stock_Price', 'Ticker'], axis=1).to_numpy())

    model, result = model(optuna, x_train, y_train, x_test, y_test, train=False, best_params=best_params)


    return estimate_annualy_income_test(model, train_df, y_test, price_discount=price_discount, bear_inv=bear_inv)