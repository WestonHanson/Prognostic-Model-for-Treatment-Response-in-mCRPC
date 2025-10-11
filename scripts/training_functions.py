# Author: Weston Hanson
# Place: Fred Hutch Cancer Center, Seattle, WA
# Date Created: 10/3/2025
# Purpose: To test how to train and test data in XGBoost

# Imports
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, average_precision_score, log_loss
from sklearn.model_selection import StratifiedKFold
import time
from sklearn.model_selection import StratifiedKFold
from itertools import product
import pickle
import os
import optuna
import sys

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

def nested_five_fold_cv_bayesian(X, y, param_ranges, objective, tree_method, boost_rounds, nfold, metrics, 
                                  verbose_eval, early_stopping, curr_time, n_trials=100, random_state=42,
                                  primary_metric=None):
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
        curr_time: string
            Timestamp to name file directory.
        n_trials: int
            Number of Bayesian optimization trials (default: 100)
        random_state: int
            Random seed for reproducibility (default: 42)

    Function:
    ---------
        - Performs nested 5 fold cross validation with Bayesian optimization.
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

    non_num_cols = X.select_dtypes(exclude=["number"]).columns
    for col in non_num_cols:
        print("has text cols")
        X[col] = X[col].astype('category')

    outer_cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = random_state)
    outer_results = []
    fold_num = 1
    
    for train_idx, test_idx in outer_cv.split(X, y):
        print(f"\n=== Outer Fold {fold_num} ===")

        # Split outer fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Flatten y from 2D column vector to 1D array
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()
        
        # Set up DMatrixes 
        dtrain = xgb.DMatrix(X_train, label = y_train, enable_categorical = True)
        dtest = xgb.DMatrix(X_test, label = y_test, enable_categorical = True)

        # Define objective function for Optuna
        def optuna_objective(trial):
            # Suggest hyperparameters
            params = {
                'eta': trial.suggest_float('eta', param_ranges['eta'][0], param_ranges['eta'][1], log=True),
                'max_depth': trial.suggest_int('max_depth', param_ranges['max_depth'][0], param_ranges['max_depth'][1]),
                'min_child_weight': trial.suggest_int('min_child_weight', param_ranges['min_child_weight'][0], param_ranges['min_child_weight'][1]),
                'max_delta_step': trial.suggest_int('max_delta_step', param_ranges['max_delta_step'][0], param_ranges['max_delta_step'][1]),
                'subsample': trial.suggest_float('subsample', param_ranges['subsample'][0], param_ranges['subsample'][1]),
                'colsample_bytree': trial.suggest_float('colsample_bytree', param_ranges['colsample_bytree'][0], param_ranges['colsample_bytree'][1]),
                'gamma': trial.suggest_float('gamma', param_ranges['gamma'][0], param_ranges['gamma'][1]),
                'lambda': trial.suggest_float('lambda', param_ranges['lambda'][0], param_ranges['lambda'][1]),
                'alpha': trial.suggest_float('alpha', param_ranges['alpha'][0], param_ranges['alpha'][1]),
                'objective': objective,
                'eval_metric': metrics,
                'tree_method': tree_method
            }

            # Suggest which metric to optimize for early stopping
            suggested_metric = trial.suggest_categorical('optimization_metric', metrics)
            
            if objective == "multi:softprob":
                params['num_class'] = len(np.unique(y))

            # Run 5-fold CV
            cv_results = five_fold_cv(dtrain, params, boost_rounds, nfold, metrics, verbose_eval, early_stopping, random_state)

            # Find best round based on suggested metric
            if suggested_metric in maximize_metrics:
                best_round_for_metric = cv_results[f'test-{suggested_metric}-mean'].idxmax()
            else:
                best_round_for_metric = cv_results[f'test-{suggested_metric}-mean'].idxmin()
            
            # Return best score based on main model metric
            best_main_model_metric_round = cv_results.loc[best_round_for_metric, f'test-{metrics[0]}-mean']
            return best_main_model_metric_round

        # Create Optuna study
        study = optuna.create_study(
            direction=optimization_direction,
            sampler=optuna.samplers.TPESampler(seed=random_state)
        )
        
        # Optimize
        print(f"Starting Bayesian optimization with {n_trials} trials...")
        study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=True)
        
        # Get best parameters
        best_params = study.best_params.copy()
        best_optimization_metric = best_params.pop("optimization_metric")
        best_params['objective'] = objective
        best_params['eval_metric'] = metrics
        best_params['tree_method'] = tree_method
        best_params['seed'] = random_state
        if objective == "multi:softprob":
            best_params['num_class'] = len(np.unique(y))
        
        best_score = study.best_value
        
        print(f"\nBest params from Bayesian optimization: {best_params}")
        print(f"Best inner {metrics[0]}: {best_score:.4f}")
        
        # Get best number of rounds by running CV one more time with best params
        cv_results = five_fold_cv(dtrain, best_params, boost_rounds, nfold, metrics, verbose_eval, early_stopping, random_state)
        
        # Find best round based on metric chosen by Bayesian optimization
        if best_optimization_metric in maximize_metrics:
            best_round = cv_results[f'test-{best_optimization_metric}-mean'].idxmax() + 1
        else:
            best_round = cv_results[f'test-{best_optimization_metric}-mean'].idxmin() + 1

        inner_metric_scores = {}
        for metric in metrics:
            inner_metric_scores[metric] = cv_results.loc[best_round - 1, f'test-{metric}-mean']

        # Train on full training fold
        model = xgb.train(
            params=best_params,
            dtrain=dtrain,
            num_boost_round=best_round,
        )
        
        # Test on outer test fold
        y_pred_prob = model.predict(dtest)

        if objective == "multi:softprob":
            y_pred = np.argmax(y_pred_prob, axis = 1)
        else:
            y_pred = (y_pred_prob > 0.5).astype(int)

        # Calculate outer fold metrics
        accuracy = accuracy_score(y_test, y_pred)
        outer_metric_scores = {"accuracy": accuracy}

        if 'logloss' in metrics:
            outer_metric_scores['logloss'] = log_loss(y_test, y_pred_prob)

        # For binary classification, calculate AUC
        if objective != "multi:softprob":
            if 'auc' in metrics:
                outer_metric_scores['auc'] = roc_auc_score(y_test, y_pred_prob)
            if 'aucpr' in metrics:
                outer_metric_scores['aucpr'] = average_precision_score(y_test, y_pred_prob)  
        else:
            if 'logloss' not in outer_metric_scores:
                outer_metric_scores['logloss'] = log_loss(y_test, y_pred_prob)
                  

        outer_results.append({
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
            'optuna_study': study  # Save the study for later analysis
        })

        fold_num += 1

    print("\n=== Overall Results ===")
    mean_accuracy = np.mean([r['accuracy'] for r in outer_results])
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Std Accuracy: {np.std([r['accuracy'] for r in outer_results]):.4f}")

    print("\n=== Metric Summary (outer folds) ===")
    all_outer_metrics = set()
    for r in outer_results:
        all_outer_metrics.update(r['outer_metric_scores'].keys())

    for metric in sorted(all_outer_metrics):
        scores = [r['outer_metric_scores'].get(metric) for r in outer_results if metric in r['outer_metric_scores']]
        if scores:
            print(f"Mean {metric}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

    # Create directory 
    file_dir = os.path.join("..", "saved-models", f"{curr_time}_model_accuracy_{mean_accuracy:.4f}")
    try:
        os.mkdir(f"{file_dir}")
        print(f"{file_dir} directory made.")
    except FileExistsError:
            print(f"{file_dir} directory already exists.")
    except FileNotFoundError:
        print(f"Error: Parent directory for {file_dir} does not exist.")
        
    # Save parameters
    filename = os.path.join(file_dir, "xgb_model_parameters.pkl")
    with open(filename, 'wb') as file:
        pickle.dump(outer_results, file)

    return outer_results, file_dir

def nested_five_fold_cv(X, y, params_dict, objective, tree_method, boost_rounds, nfold, metrics, verbose_eval, early_stopping, curr_time):
    '''
    Parameters:
    -----------
        X: pandas dataframe
            The predicter data.
        y: pandas dataframe
            The predicted data.
        params_dict : dictionary
            dictionary of combinations of parameters for the trainng (only numeric values).
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
        curr_time: string
            Timestamp to name file directory.

    Function:
    ---------
        - Performs nested 5 fold cross validation.
        - Makes sure X and y are dataframes then makes sure all columns are numeric or categorical.
        - It does grid search on the parameter list and does a 5 fold cv for every combination.
        - Finds best performing combination of parameters and saves that.
        - Repeats that process for all 5 outer folds.
        - Saves model parameters as a pickle and returns it.

    Returns:
    --------
        outer_results:
            Dictionary of cross validated results.
        file_dir:
            Timestamped file directory to save parameters and completed model.
    '''

    # Make sure X and y are pandas objs
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    
    if isinstance(y, np.ndarray):
        y = pd.DataFrame(y)

    non_num_cols = X.select_dtypes(exclude=["number"]).columns
    for col in non_num_cols:
        print("has text cols")
        X[col] = X[col].astype('category')

    outer_cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    outer_results = []
    fold_num = 1

    # Create all hyperparameter combinations
    keys, values = zip(*params_dict.items())
    params_combinations = [dict(zip(keys, v)) for v in product(*values)]

    for train_idx, test_idx in outer_cv.split(X, y):
        print(f"\n=== Outer Fold {fold_num} ===")

        # Split outer fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Flatten y from 2D column vector to 1D array
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()

        # Set up DMatrixes 
        dtrain = xgb.DMatrix(X_train, label = y_train, enable_categorical = True)
        dtest = xgb.DMatrix(X_test, label = y_test, enable_categorical = True)

        best_score = np.inf
        best_params = None
        best_round = None

        counter = 1

        # Inner 5-fold CV over hyperparameter dict
        for combo in params_combinations:
            print(f"Loop {counter}")
            print(f"combo: {combo}")
            print()
            params = combo.copy()
            params['objective'] = objective
            if objective == "multi:softprob":
                params['num_class'] = len(np.unique(y))
            params['eval_metric'] = metrics[0]
            params['tree_method'] = tree_method

            cv_results = five_fold_cv(dtrain, params, boost_rounds, nfold, metrics, verbose_eval, early_stopping)

            mean_metric = cv_results[f'test-{metrics[0]}-mean'].min()
            round_best = cv_results[f'test-{metrics[0]}-mean'].idxmin()

            # Select best round from CV
            if mean_metric < best_score:
                best_score = mean_metric
                best_params = params
                best_round = round_best + 1

            counter += 1

        
        print(f"Best params from inner CV: {best_params}")
        print(f"Best round from inner CV: {best_round}")
        print(f"Best inner {metrics[0]}: {best_score:.4f}")

        # Train on full training fold
        model = xgb.train(
            params=best_params,
            dtrain=dtrain,
            num_boost_round=best_round
            )
        
        # Test on outer test fold
        y_pred_prob = model.predict(dtest)

        if objective == "multi:softprob":
            y_pred = np.argmax(y_pred_prob, axis = 1)
        else:
            y_pred = (y_pred_prob > 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy (outer fold {fold_num}): {accuracy:.4f}")

        outer_results.append({
            'fold': fold_num,
            'best_params': best_params,
            'best_round': best_round,
            'inner_best_score': best_score,
            'accuracy': accuracy,
            'class_report': classification_report(y_test, y_pred, output_dict = True),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        })

        fold_num += 1

    print("\n=== Overall Results ===")
    mean_accuracy = np.mean([r['accuracy'] for r in outer_results])
    print(f"Mean Accuracy: {mean_accuracy}")
    print(f"Std Accuracy: {np.std([r['accuracy'] for r in outer_results]):.4f}")

    # Create directory 
    file_dir = os.path.join("..", "saved-models", f"{curr_time}_model_accuracy_{mean_accuracy:.4f}")
    try:
        os.mkdir(f"{file_dir}")
        print(f"{file_dir} directory made.")
    except FileExistsError:
            print(f"{file_dir} directory already exists.")
    except FileNotFoundError:
        print(f"Error: Parent directory for {file_dir} does not exist.")
        
    # Save parameters
    filename = os.path.join(file_dir, "xgb_model_parameters.pkl")
    with open(filename, 'wb') as file:
        pickle.dump(outer_results, file)

    return outer_results, file_dir
    
