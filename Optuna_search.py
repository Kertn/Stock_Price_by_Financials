import optuna
from train_models import *
from functools import partial
from preprocessing import optuna_models_params, optuna_price_disc, Model, predict_invest, estim, test_estim


# Define the best params for each model and write them into a text file
def Optuna_Optim():
    all_filters_system = ['Aerospace', 'Auto_Tires_Trucks', 'Basic_Materials', 'Business_Services', 'Computer_and_Technology', 'Construction', 'Consumer_Discretionary', 'Consumer_Staples', 'Finance', 'Industrial_Products', 'Medical_Multi-Sector_Conglomerates', 'Oils_Energy', 'Retail_Wholesale', 'Transportation', 'Unclassified', 'Utilities', 'Full_list']
    all_models = [BayesianRidg_func, ElasticNet_func, random_forest_func, XgBoost_func, GradBoostRegr_func, CatBoostRegr_func, NeuralNetTorch_func]
    trials_model = [1000, 1000, 140, 120, 70, 70, 70]
    n_jobs = [4, 4, 1, 1, 1, 1, 1]

    for sector_name in all_filters_system:
        for model, trials, n_job in zip(all_models, trials_model, n_jobs):

            optuna_ = partial(optuna_models_params, model=model, sector_name=sector_name)
            opt_model = optuna.create_study(study_name='NeurNet', direction='minimize')
            opt_model.optimize(optuna_, n_trials=trials, n_jobs=n_job)

            read_model = Model(model=model, sector_name=sector_name, best_params=opt_model.best_params)

            optuna_new = partial(optuna_price_disc, model=read_model, sector_name=sector_name, best_params=opt_model.best_params)
            opt_disc = optuna.create_study(study_name='NeurNet', direction='maximize')
            opt_disc.optimize(optuna_new, n_trials=85, n_jobs=n_job)
            with open('best_models.txt', 'a') as file:
                file.write(f'{sector_name} - {model} - {opt_model.best_params} - {opt_disc.best_params} \n\n')


# Using the selected model and best params, predict the best-invested stocks
def Invest():
    model = NeuralNetTorch_func
    sector_name = 'Full_list'
    best_params_model = {'miss_data_column_allowed': 0.79, 'miss_data_row_allowed': 0.2, 'nlarge': 75, 'num_of_epochs': 65000, 'lr': 0.025}

    best_params_estim = {'bear_inv': False, 'price_discount': 0.19}

    file_invest_name = 'Invest_Full_list_Neural_Network.csv'

    predict_invest(model, sector_name, best_params_model, best_params_estim, file_invest_name)


# Using the selected model and best params, get the test result (remove last year) of estimated models
def Test_estim_model():
    f_params = [{'miss_data_column_allowed': 0.8, 'miss_data_row_allowed': 0.294, 'nlarge': 95, 'bootstrap': False, 'max_depth': 290, 'max_features': 'sqrt', 'min_samples_leaf': 3, 'min_samples_split': 6, 'n_estimators': 600}]
    s_params = [{'bear_inv': False, 'price_discount': 0.43}]
    models = [random_forest_func]
    sectors_names = ['Computer_and_Technology']

    for model, sector_name, best_params, optuna_est in zip(models, sectors_names, f_params, s_params):
        print('Estimation of the Model:')
        read_model = Model(model=model, sector_name=sector_name, best_params=best_params)
        estim(read_model, sector_name, best_params, optuna_est)
        print('\n Test Estimation of the Model:')
        print(sector_name, ' - ', model)
        test_estim(model, sector_name, best_params, optuna_est)
