# TODO estimate function by annualy earning from predictions
from statistics import mean
from matplotlib import pyplot as plt
import numpy as np
def estimate_annualy_income(models, X_estimate, tickers, price_coll):

    fig, axs = plt.subplots(5)

    for ind, model in enumerate(models):
        print('Model - ', model)

        prev_ticker = ''
        total_income_percent = []
        total_income_values = []
        total_invested = 0
        total_income = 0
        invest_price = 0
        total_balance = 5000
        for X, ticker, actual_price in zip(X_estimate, tickers, price_coll):
            current_ticker = str(ticker).split('.')[0]
            current_year = str(ticker).split('.')[1]

            if invest_price == 0 or prev_ticker != current_ticker:
                prev_ticker = current_ticker

                price_pred = model.predict(X.reshape(1, -1))

                if price_pred * 0.8 > actual_price:
                    stocks = total_balance / actual_price
                    invest_price = stocks * actual_price
                    total_invested += invest_price
                    side = 1
                elif price_pred < actual_price * 0.8:
                    stocks = total_balance / actual_price
                    invest_price = stocks * actual_price
                    total_invested += invest_price
                    side = 0

                else:
                    invest_price = 0

            else:
                if side:
                    total_income_percent.append((stocks * actual_price)/invest_price)
                    total_income += stocks * actual_price - invest_price
                    total_income_values.append(total_income)
                else:
                    total_income_percent.append(invest_price / (stocks * actual_price))
                    total_income += invest_price - stocks * actual_price
                    total_income_values.append(total_income)

                price_pred = model.predict(X.reshape(1, -1))

                if price_pred * 0.8 > actual_price:
                    stocks = total_balance / actual_price
                    invest_price = stocks * actual_price
                    total_invested += invest_price
                    side = 1
                elif price_pred < actual_price * 0.8:
                    stocks = total_balance / actual_price
                    invest_price = stocks * actual_price
                    total_invested += invest_price
                    side = 0

                else:
                    invest_price = 0

        #print('INCOME_Percents', total_income_percent)
        #print('INCOME_values', total_income_values)
        print('INCOME_RESULT', total_income)
        print('TOTAL_INVESTED', total_invested)
        print('Income ratio', total_income/total_invested)
        print('MEAN', mean(total_income_percent))

        print('ind', ind)

        axs[ind].plot(np.linspace(0, 1, len(total_income_values)), total_income_values)
    plt.show()