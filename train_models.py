from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
import catboost as ctb
import xgboost as xgb

def BayesianRidg(X_train, y_train):

    #TODO Add and Tune the params!

    reg_model = linear_model.BayesianRidge()
    reg_model.fit(X_train, y_train.ravel())
    print('train_score - BayesianRidg', reg_model.score(X_train, y_train))
    # print('test_score - BayesianRidg', reg.score(X_test, y_test))
    print('test_mean_abs_perc_error - BayesianRidg', mean_absolute_percentage_error(reg_model.predict(X_train), y_train))
    print('\n\n')
    return reg_model

def GradBoostRegr(X_train, y_train):

    # TODO Add and Tune the params!

    reg_model = GradientBoostingRegressor(random_state=0)
    reg_model.fit(X_train, y_train.ravel())
    GradientBoostingRegressor(random_state=0)
    print('train_score - GradBoostRegr', reg_model.score(X_train, y_train))
    #print('test_score - GradBoostRegr', reg.score(X_test, y_test))
    print('test_mean_abs_perc_error - GradBoostRegr', mean_absolute_percentage_error(reg_model.predict(X_train), y_train))
    print('\n\n')
    return reg_model

def XgBoost(X_train, y_train):

    # TODO Add and Tune the params!

    reg_model = xgb.XGBRegressor()
    reg_model.fit(X_train, y_train)
    print('train_score - XgBoost', reg_model.score(X_train, y_train))
    #print('test_score - XgBoost', reg.score(X_test, y_test))
    print('test_mean_abs_perc_error - XgBoost', mean_absolute_percentage_error(reg_model.predict(X_train), y_train))
    print('\n\n')
    return reg_model


def random_forest(X_train, y_train):

    # TODO Add and Tune the params!

    reg_model = RandomForestRegressor(random_state=0)
    reg_model.fit(X_train, y_train.ravel())
    print('train_score - random_forest', reg_model.score(X_train, y_train))
    #print('test_score - random_forest', reg.score(X_test, y_test))
    print('test_mean_abs_perc_error - random_forest', mean_absolute_percentage_error(reg_model.predict(X_train), y_train))
    print('\n\n')

    return reg_model


#def NeuralNetTorch(X_train, y_train, num_of_epochs, lr):
def NeuralNetTorch(X_train, y_train, X_test, y_test):

    class linearRegression(nn.Module):
        def __init__(self, input_dim):
            super(linearRegression, self).__init__()
            self.model = nn.Sequential(
                         nn.Linear(input_dim, 60),
                         nn.ReLU(),
                         nn.Linear(60, 48),
                         nn.ReLU(),
                         nn.Linear(48, 22),
                         nn.ReLU(),
                         nn.Linear(22, 11),
                         nn.ReLU(),
                         nn.Linear(11, 1))

        def forward(self, d):
            out = self.model(d)
            return out

        def predict(self, input):
            input = torch.from_numpy(input).float()
            with torch.no_grad():
                out = self.model(input)
                return out

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()

    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()


    input_dim = X_train.shape[1]
    torch.manual_seed(42)
    reg_model = linearRegression(input_dim)

    loss = nn.MSELoss()
    optimizers = optim.Adam(params=reg_model.parameters(), lr=0.01)

    num_of_epochs = 100
    for i in range(num_of_epochs):

        y_train_prediction = reg_model(X_train)
        loss_value = loss(y_train_prediction.squeeze(), y_train)
        optimizers.zero_grad()
        loss_value.backward()
        optimizers.step()

    with torch.no_grad():
        #print('train_score - linearRegression', r2_score(reg_model.forward(X_train), y_train))
        # print('test_score - linearRegression', reg_model.score(X_test, y_test))
        print('test_mean_abs_perc_error - linearRegression', mean_absolute_percentage_error(reg_model.forward(X_test), y_test))
        print('\n\n')

    return reg_model
