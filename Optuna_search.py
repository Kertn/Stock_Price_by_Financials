import optuna
from train_models import *
from functools import partial
from preprocessing import optuna_models_params, optuna_price_disc, Model, predict_invest, estim, end_estim

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
    model = GradBoostRegr_func
    sector_name = 'Finance'
    best_params_model = {'miss_data_column_allowed': 0.77, 'miss_data_row_allowed': 0.7, 'nlarge': 115, 'loss': 'huber', 'criterion': 'squared_error', 'min_samples_split': 6, 'min_samples_leaf': 5, 'alpha': 0.7767083304304833, 'n_estimators': 390, 'learning_rate': 0.013276306407646956, 'max_depth': 11, 'subsample': 0.7595010529541397}
    best_params_estim = {'bear_inv': True, 'price_discount': 0.75}

    read_model = Model(model=model, sector_name=sector_name, best_params=best_params_model)

    predict_invest(read_model, sector_name, best_params_model, best_params_estim)

# Using the selected model and best params, get the test result (remove last year) of estimated models
def Test_estim_model():
    f_params = [{'miss_data_column_allowed': 0.77, 'miss_data_row_allowed': 0.7, 'nlarge': 115, 'loss': 'huber', 'criterion': 'squared_error', 'min_samples_split': 6, 'min_samples_leaf': 5, 'alpha': 0.7767083304304833, 'n_estimators': 390, 'learning_rate': 0.013276306407646956, 'max_depth': 11, 'subsample': 0.7595010529541397}]
    s_params = [{'bear_inv': True, 'price_discount': 0.75}]
    models = [GradBoostRegr_func]
    sectors_names = ['Finance']

    for model, sector_name, best_params, optuna_est in zip(models, sectors_names, f_params, s_params):
        read_model = Model(model=model, sector_name=sector_name, best_params=best_params)
        estim(read_model, sector_name, best_params, optuna_est)
        print(sector_name, ' - ', model)
        end_estim(model, sector_name, best_params, optuna_est)
        print('\n')
