# Author: Weston Hanson
# Place: Fred Hutch Cancer Center, Seattle, WA
# Date Created: 10/3/2025
# Purpose: To test how to train and test data in XGBoost

from plotting_functions import *
from data_processing_functions import *

# Imports
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score, average_precision_score, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
import time
from sklearn.utils import resample
from itertools import product
import pickle
import os
import optuna
import sys
from joblib import Parallel, delayed

def five_fold_cv(training_matrix, params, boost_rounds, nfold, metrics, verbose_eval, early_stopping, random_state):
    '''
    Parameters:
    -----------
        training_matrix : DMatrix
            Dataframe for training data.
        params : JSON
            JSON of parameters for the trainng like: type of training, tree method, etc.
        boost_rounds : int
            Max number of rounds for boosting the algorithm. 
        nfold : int
            Number of folds the algorithm will complete.
        metrics : list
            List of metrics to evaluate performance.
        verbose_eval: int
            Number of iterations the model will output performance.
        early_stopping: int
            Number of iterations to stop the model when no progress is being made on performance. 
        

    Function:
    ---------
        - Performs 5 fold cross validation on training data.

    Returns:
    --------
        cv_results:
            Dictionary of cross validated results.
    '''
    cv_results = xgb.cv(
        params,
        training_matrix,
        num_boost_round = boost_rounds,
        nfold = nfold,
        metrics = metrics, # List of metrics to evaluate performance
        verbose_eval = verbose_eval, # Outputs model's performance every 10 boosts
        early_stopping_rounds = early_stopping, # Stops model if performance doesn't increase after 50 boosts
        seed=random_state
    )

    return cv_results

def _process_outer_fold_parallel(fold_data):
    '''
    Helper function to process a single outer fold for parallelization.
    
    Parameters:
    -----------
        fold_data : dict
            Dictionary containing all necessary data for processing one fold
    
    Returns:
    --------
        result : dict
            Dictionary containing results for this fold
    '''

    # -----------------------------
    # Unpack fold data
    # -----------------------------
    train_idx = fold_data['train_idx']
    test_idx = fold_data['test_idx']
    fold_num = fold_data['fold_num']
    X = fold_data['X']
    y = fold_data['y']
    param_ranges = fold_data['param_ranges']
    objective = fold_data['objective']
    tree_method = fold_data['tree_method']
    boost_rounds = fold_data['boost_rounds']
    nfold = fold_data['nfold']
    base_score = fold_data['base_score']
    metrics = fold_data['metrics']
    verbose_eval = fold_data['verbose_eval']
    early_stopping = fold_data['early_stopping']
    feature_selection_method = fold_data['feature_selection_method']
    X_train_filtered_columns = fold_data['X_train_filtered_columns']
    version_dir = fold_data['version_dir']
    n_trials = fold_data['n_trials']
    random_state = fold_data['random_state']
    primary_metric = fold_data.get('primary_metric', None)
    maximize_metrics = fold_data['maximize_metrics']
    optimize_direction = fold_data['optimization_direction']


    # -----------------------------
    # Split outer fold
    # -----------------------------
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Flatten y from 2D column vector to 1D array
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    # -----------------------------
    # Handle class imbalance
    # -----------------------------
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    if num_pos == 0:
        raise ValueError("Training fold has no positive samples; cannot compute scale_pos_weight.")
    scale_pos_weight = num_neg / num_pos

    # -----------------------------
    # FEATURE SELECTION
    # -----------------------------
    if feature_selection_method is not None:
        if feature_selection_method == "variable_features":
            X_train_filtered = X_train[X_train_filtered_columns]
        else:
            X_train_filtered_columns = feature_selection_helper(feature_selection_method, X_train, y_train, random_state)
            X_train_filtered = X_train[X_train_filtered_columns]
    else: 
        X_train_filtered = X_train

    # Filter both the test and training data so there is no leakage
    X_train_filtered_columns_complete = X_train_filtered.columns
    X_train_filtered = X_train[X_train_filtered_columns_complete]
    X_test_filtered = X_test[X_train_filtered_columns_complete]

    assert all(dtype.name != 'object' for dtype in X_train_filtered.dtypes)
    assert all(dtype.name != 'object' for dtype in X_test_filtered.dtypes)

    # -----------------------------
    # Set up DMatrix for XGBoost
    # -----------------------------
    dtrain_filtered = xgb.DMatrix(X_train_filtered, label=y_train, enable_categorical=True)
    dtest_filtered = xgb.DMatrix(X_test_filtered, label=y_test, enable_categorical=True)

    # -----------------------------
    # Optuna objective function
    # -----------------------------
    def optuna_objective(trial):
        params = {
            'eta': trial.suggest_float('eta', param_ranges['eta'][0], param_ranges['eta'][1], log=True),
            'max_depth': trial.suggest_int('max_depth', param_ranges['max_depth'][0], param_ranges['max_depth'][1]),
            'min_child_weight': trial.suggest_int('min_child_weight', param_ranges['min_child_weight'][0], param_ranges['min_child_weight'][1]),
            'max_delta_step': trial.suggest_int('max_delta_step', param_ranges['max_delta_step'][0], param_ranges['max_delta_step'][1]),
            'subsample': trial.suggest_float('subsample', param_ranges['subsample'][0], param_ranges['subsample'][1]),
            'colsample_bytree': trial.suggest_float('colsample_bytree', param_ranges['colsample_bytree'][0], param_ranges['colsample_bytree'][1]),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', param_ranges['colsample_bylevel'][0], param_ranges['colsample_bylevel'][1]),
            'colsample_bynode': trial.suggest_float('colsample_bynode', param_ranges['colsample_bynode'][0], param_ranges['colsample_bynode'][1]),
            'gamma': trial.suggest_float('gamma', param_ranges['gamma'][0], param_ranges['gamma'][1]),
            'lambda': trial.suggest_float('lambda', param_ranges['lambda'][0], param_ranges['lambda'][1]),
            'alpha': trial.suggest_float('alpha', param_ranges['alpha'][0], param_ranges['alpha'][1]),
            'objective': objective,
            'eval_metric': metrics,
            'tree_method': tree_method,
            'base_score': base_score,
            'scale_pos_weight': scale_pos_weight,
            'enable_categorical': True,
        }

        suggested_metric = trial.suggest_categorical('optimization_metric', metrics)
        
        if objective == "multi:softprob":
            params['num_class'] = len(np.unique(y))

        # Run five fold cv
        cv_results = five_fold_cv(dtrain_filtered, params, boost_rounds, nfold, metrics, verbose_eval, early_stopping, random_state)

        # Print the learning curves
        learning_curves_data = plot_xgb_learning_curves(
            cv_results, 
            metrics, 
            trial.number, 
            version_dir, 
            maximize_metrics
        )
        trial.set_user_attr('learning_curves', learning_curves_data)

        if suggested_metric in maximize_metrics:
            best_round_for_metric = cv_results[f'test-{suggested_metric}-mean'].idxmax()
        else:
            best_round_for_metric = cv_results[f'test-{suggested_metric}-mean'].idxmin()

        # Save best round and full CV results in user attributes
        trial.set_user_attr('best_round_for_metric', best_round_for_metric)
        trial.set_user_attr('cv_results', cv_results)
        
        # best_main_model_metric_round = cv_results.loc[best_round_for_metric, f'test-{metrics[0]}-mean']
        # return best_main_model_metric_round
        return best_round_for_metric

    # -----------------------------
    # Run Optuna study
    # -----------------------------
    optuna.logging.set_verbosity(optuna.logging.WARNING) # Suppress Optuna's verbose logging
    study = optuna.create_study(direction=optimize_direction, sampler=optuna.samplers.TPESampler(seed=random_state))    
    study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=False)
    
    # Get best parameters
    best_trial = study.best_trial
    best_params = best_trial.params.copy()
    best_optimization_metric = best_params.pop("optimization_metric")
    best_params['objective'] = objective
    best_params['eval_metric'] = metrics
    best_params['tree_method'] = tree_method
    best_params['seed'] = random_state
    best_params['scale_pos_weight'] = float(scale_pos_weight)
    if objective == "multi:softprob":
        best_params['num_class'] = len(np.unique(y))
    best_score = study.best_value
    best_round = best_trial.user_attrs['best_round_for_metric'] + 1
    cv_results = best_trial.user_attrs['cv_results']

    inner_metric_scores = {}
    for metric in metrics:
        inner_metric_scores[metric] = cv_results.loc[best_round - 1, f'test-{metric}-mean']

    # Train on full training fold
    model = xgb.train(
        params=best_params,
        dtrain=dtrain_filtered,
        num_boost_round=best_round,
    )
    
    # Test on outer test fold
    y_pred_prob = model.predict(dtest_filtered)

    if objective == "multi:softprob":
        y_pred = np.argmax(y_pred_prob, axis = 1)
    else:
        median_value = np.median(y_pred_prob)
        # y_pred = (y_pred_prob > median_value).astype(int)
        y_pred = (y_pred_prob > 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    outer_metric_scores = {"accuracy": accuracy}

    if 'logloss' in metrics:
        outer_metric_scores['logloss'] = log_loss(y_test, y_pred_prob)

    if objective != "multi:softprob":
        if 'auc' in metrics:
            outer_metric_scores['auc'] = roc_auc_score(y_test, y_pred_prob)
        if 'aucpr' in metrics:
            outer_metric_scores['aucpr'] = average_precision_score(y_test, y_pred_prob)  
    else:
        if 'logloss' not in outer_metric_scores:
            outer_metric_scores['logloss'] = log_loss(y_test, y_pred_prob)

    return {
        'fold': fold_num,
        'best_params': best_params,
        'best_optimization_metric': best_optimization_metric,
        'best_round': best_round,
        'inner_metric_scores': inner_metric_scores,
        'inner_best_auc': best_score,
        'outer_metric_scores': outer_metric_scores,
        'accuracy': accuracy,
        'class_report': classification_report(y_test, y_pred, output_dict = True),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'optuna_study': study,
        'y_pred_prob': y_pred_prob,
        'y_test': y_test
    }

def nested_five_fold_cv_bayesian_with_feature_selection(
        X, y, param_ranges, objective, tree_method, boost_rounds, nfold, base_score, metrics, 
        verbose_eval, early_stopping, feature_selection_method, X_train_filtered_columns, version_dir, training_params_file_name,  
        n_trials=100, random_state=42, primary_metric=None
        ):
    '''
    Parameters:
    -----------
        X: pandas dataframe
            The predicter data.
        y: pandas dataframe
            The predicted data.
        param_ranges : dictionary
            Dictionary of parameter ranges for Bayesian optimization.
            Each value should be a tuple (min, max) for continuous params or list for categorical.
        objective: string
           Type of regression will the model will use.
        tree_method: string
            Method the model will use for building boost trees.
        boost_rounds : int
            Max number of rounds for boosting the algorithm. 
        nfold : int
            Number of folds the algorithm will complete.
        metrics : list
            List of metrics to evaluate performance.
        verbose_eval: int
            Number of iterations the model will output performance.
        early_stopping: int
            Number of iterations to stop the model when no progress is being made on performance. 
        n_trials: int
            Number of Bayesian optimization trials (default: 100)
        random_state: int
            Random seed for reproducibility (default: 42)
        primary_metric: str
            Primary metric to optimize (default: first metric in metrics list)

    Function:
    ---------
        - Performs nested 5 fold cross validation with Bayesian optimization.
        - Outer folds processed sequentially; Optuna trials within each fold run sequentially.
        - Uses Optuna for intelligent hyperparameter search instead of grid search.
        - For each outer fold, runs Bayesian optimization to find best parameters.
        - Trains final model on best parameters and evaluates on outer test fold.

    Returns:
    --------
        outer_results:
            Dictionary of cross validated results.
        file_dir:
            Timestamped file directory to save parameters and completed model.
    '''
    # Set pimary metric for optimization
    if primary_metric is None:
        primary_metric = metrics[0]

    maximize_metrics = ["auc", "aucpr", "map"]
    if primary_metric in maximize_metrics:
        optimization_direction = "maximize"
    else :
        optimization_direction = "minimize"

    # Make sure X and y are pandas objs
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.DataFrame(y)

    # Turn all non-numeric columns to categorical
    non_num_cols = X.select_dtypes(exclude=["number"]).columns
    for col in non_num_cols:
        X[col] = X[col].astype('category')

    # Assign stratify fold object
    outer_cv = StratifiedKFold(n_splits = nfold, shuffle = True, random_state = random_state)

    # Prepare fold data for processing
    fold_data_list = []
    fold_num = 1
    
    for train_idx, test_idx in outer_cv.split(X, y):
        fold_data = {
            'train_idx': train_idx,
            'test_idx': test_idx,
            'fold_num': fold_num,
            'X': X,
            'y': y,
            'param_ranges': param_ranges,
            'objective': objective,
            'tree_method': tree_method,
            'boost_rounds': boost_rounds,
            'nfold': nfold,
            'base_score': base_score,
            'metrics': metrics,
            'verbose_eval': verbose_eval,
            'early_stopping': early_stopping,
            'feature_selection_method': feature_selection_method,
            'X_train_filtered_columns': X_train_filtered_columns,
            'version_dir': version_dir,
            'n_trials': n_trials,
            'random_state': random_state,
            'primary_metric': primary_metric,
            'maximize_metrics': maximize_metrics,
            'optimization_direction': optimization_direction
        }
        fold_data_list.append(fold_data)
        fold_num += 1
    
    # Process folds
    outer_results = []
    for fold_data in fold_data_list:
        result = _process_outer_fold_parallel(fold_data)
        outer_results.append(result)

    # Sort results by fold number
    outer_results = sorted(outer_results, key=lambda x: x['fold'])

    # Dictionary to keep track of training predictions
    training_pred_dict = {}

    # Create AUC for concatenation of all folds predictions
    training_pred_dict = {r['fold']: {'pred': r['y_pred_prob'], 'test': r['y_test']} for r in outer_results}
    combined_preds = np.concatenate([training_pred_dict[key]['pred'] for key in training_pred_dict.keys()])
    combined_real = np.concatenate([training_pred_dict[key]['test'] for key in training_pred_dict.keys()])
    fpr, tpr, roc_thresholds = roc_curve(combined_real, combined_preds)
    roc_auc = auc(fpr, tpr)

    # Remove y_pred_prob and y_test from results before saving (you just need the object to reproduce)
    for r in outer_results:
        r.pop('y_pred_prob', None)
        r.pop('y_test', None)

    # Save parameters
    filename = os.path.join(version_dir, f"{training_params_file_name}")
    with open(filename, 'wb') as file:
        pickle.dump(outer_results, file)

    return outer_results, roc_auc, version_dir

