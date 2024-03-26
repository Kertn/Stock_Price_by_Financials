import optuna
from train_models import *
from functools import partial
from Preprocessing import optuna_models_params, optuna_price_disc, Model, predict_invest

def Optuna_Optim():
    #TODO 'Auto_Tires_Trucks', Computer_and_Technology, Consumer_Discretionary, 'Finance', 'Transportation' - доделать!
    all_filters_system = ['Unclassified', 'Utilities', 'Full_list']
    all_models = [BayesianRidg_func, ElasticNet_func, random_forest_func, XgBoost_func, GradBoostRegr_func, CatBoostRegr_func, NeuralNetTorch_func]
    trials_model = [1000, 1000, 140, 120, 70, 70, 70]
    n_jobs = [1, 1, 1, 1, 1, 1, 1]

    for sector_name in all_filters_system:
        for model, trials, n_job in zip(all_models, trials_model, n_jobs):
            optuna_ = partial(optuna_models_params, model=model, sector_name=sector_name)
            opt_model = optuna.create_study(study_name='NeurNet', direction='minimize')
            opt_model.optimize(optuna_, n_trials=trials, n_jobs=n_job)
            print('opt_model.best_params', opt_model.best_params)
            read_model = Model(optuna_price_disc, model=model, sector_name=sector_name, best_params=opt_model.best_params)
            optuna_new = partial(optuna_price_disc, model=read_model, sector_name=sector_name, best_params=opt_model.best_params)
            opt_disc = optuna.create_study(study_name='NeurNet', direction='maximize')
            opt_disc.optimize(optuna_new, n_trials=85, n_jobs=n_job)
            with open('best_models.txt', 'a') as file:
                file.write(f'{sector_name} - {model} - {opt_model.best_params} - {opt_disc.best_params} \n\n')


def Invest():
    "Business_Services - <function GradBoostRegr_func at 0x000001D9ED82F550> - {'miss_data_column_allowed': 0.78, 'miss_data_row_allowed': 0.45, 'nlarge': 50, 'loss': 'quantile', 'criterion': 'friedman_mse', 'min_samples_split': 6, 'min_samples_leaf': 4, 'alpha': 0.2069788108975787, 'n_estimators': 1410, 'learning_rate': 0.011450487025558651, 'max_depth': 7, 'subsample': 0.8444677168045779} - {'bear_inv': False, 'price_discount': 0.97} "
    best_params_model = {'miss_data_column_allowed': 0.78, 'miss_data_row_allowed': 0.45, 'nlarge': 50, 'loss': 'quantile', 'criterion': 'friedman_mse', 'min_samples_split': 6, 'min_samples_leaf': 4, 'alpha': 0.2069788108975787, 'n_estimators': 1410, 'learning_rate': 0.011450487025558651, 'max_depth': 7, 'subsample': 0.8444677168045779}
    best_params_estim = {'bear_inv': False, 'price_discount': 0.97}
    predict_invest(GradBoostRegr_func, 'Business_Services', best_params_model, best_params_estim)

