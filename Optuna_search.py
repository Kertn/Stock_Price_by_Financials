import optuna
from train_models import *
from functools import partial
from Preprocessing import optuna_models_params, optuna_price_disc

def Optuna_Optim():
    all_filters_system = ['Aerospace', 'Auto_Tires_Trucks', 'Basic_Materials', 'Business_Services', 'Computer_and_Technology', 'Construction', 'Consumer_Discretionary', 'Consumer_Staples', 'Finance', 'Industrial_Products', 'Medical_Multi-Sector_Conglomerates', 'Oils_Energy', 'Retail_Wholesale', 'Transportation', 'Unclassified', 'Utilities', 'Full_list']
    all_models = [BayesianRidg_func, ElasticNet_func, random_forest_func, XgBoost_func, GradBoostRegr_func, CatBoostRegr_func, NeuralNetTorch_func]
    trials_model = [1000, 1000, 140, 500, 200, 200, 100]
    n_jobs = [5, 5, 1, 1, 1, 1, 1]

    for sector_name in all_filters_system:
        for model, trials, n_job in zip(all_models, trials_model, n_jobs):
            optuna_ = partial(optuna_models_params, model=model, sector_name=sector_name)
            opt_model = optuna.create_study(study_name='NeurNet', direction='maximize')
            opt_model.optimize(optuna_, n_trials=trials, n_jobs=n_job)
            for bear_inv in [False, True]:
                optuna_new = partial(optuna_price_disc, model=model, sector_name=sector_name, best_params=opt_model.best_params, bear_inv=bear_inv)
                opt_disc = optuna.create_study(study_name='NeurNet', direction='minimize')
                opt_disc.optimize(optuna_new, n_trials=trials, n_jobs=n_job)