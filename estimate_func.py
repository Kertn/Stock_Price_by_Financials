# TODO estimate function by annualy earning from predictions
from statistics import mean
def estimate_annualy_income(model, X_estimate, tickers, price_coll):
    prev_ticker = ''
    total_income_percent = []
    invest_price = 0
    for X, ticker, actual_price in zip(X_estimate, tickers, price_coll):
        current_ticker = str(ticker).split('.')[0]
        current_year = str(ticker).split('.')[1]

        if invest_price == 0 or prev_ticker != current_ticker:
            prev_ticker = current_ticker

            price_pred = model.predict(X.reshape(1, -1))

            if price_pred * 0.7 > actual_price:
                invest_price = actual_price
            else:
                invest_price = 0

            # print('ticker', ticker)
            # print('actual_price', actual_price)
            # print('pred_price', price_pred)
            # print('\n')
        else:
            total_income_percent.append(actual_price/invest_price)

            price_pred = model.predict(X.reshape(1, -1))

            if price_pred * 0.7 > actual_price:
                invest_price = actual_price
            else:
                invest_price = 0
    print('INCOME', total_income_percent)
    print('MEAN', mean(total_income_percent))