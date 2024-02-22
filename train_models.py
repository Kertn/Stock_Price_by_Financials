from sklearn.metrics import mean_absolute_percentage_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
import catboost as ctb
import xgboost as xgb

def BayesianRidg(X_train, y_train):

    #TODO Add and Tune the params!

    reg = linear_model.BayesianRidge()
    reg.fit(X_train, y_train.ravel())
    print('train_score - BayesianRidg', reg.score(X_train, y_train))
    # print('test_score - BayesianRidg', reg.score(X_test, y_test))
    print('test_mean_abs_perc_error - BayesianRidg', mean_absolute_percentage_error(reg.predict(X_train), y_train))
    print('\n\n')

def GradBoostRegr(X_train, y_train):

    # TODO Add and Tune the params!

    reg = GradientBoostingRegressor(random_state=0)
    reg.fit(X_train, y_train.ravel())
    GradientBoostingRegressor(random_state=0)
    print('train_score - GradBoostRegr', reg.score(X_train, y_train))
    #print('test_score - GradBoostRegr', reg.score(X_test, y_test))
    print('test_mean_abs_perc_error - GradBoostRegr', mean_absolute_percentage_error(reg.predict(X_train), y_train))
    print('\n\n')

def CatBoost(X_train, y_train):

    # TODO Add and Tune the params!

    #reg = ctb.CatBoostClassifier()
    #reg.fit(X_train, y_train.ravel())
    #print('train_score - CatBoost', reg.score(X_train, y_train))
    ##print('test_score - CatBoost', reg.score(X_test, y_test))
    #print('test_mean_abs_perc_error - BayesianRidg', mean_absolute_percentage_error(reg.predict(X_train), y_train))
    print()
def XgBoost(X_train, y_train):

    # TODO Add and Tune the params!

    reg = xgb.XGBRegressor()
    reg.fit(X_train, y_train)
    print('train_score - XgBoost', reg.score(X_train, y_train))
    #print('test_score - XgBoost', reg.score(X_test, y_test))
    print('test_mean_abs_perc_error - XgBoost', mean_absolute_percentage_error(reg.predict(X_train), y_train))
    print('\n\n')


def random_forest(X_train, y_train):

    # TODO Add and Tune the params!

    reg = RandomForestRegressor(random_state=0)
    reg.fit(X_train, y_train.ravel())
    print('train_score - random_forest', reg.score(X_train, y_train))
    #print('test_score - random_forest', reg.score(X_test, y_test))
    print('test_mean_abs_perc_error - random_forest', mean_absolute_percentage_error(reg.predict(X_train), y_train))
    print('\n\n')

    return reg


def NeuralNetTorch(X_train, y_train):
    return
