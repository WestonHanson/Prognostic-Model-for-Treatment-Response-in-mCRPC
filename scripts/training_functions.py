# Author: Weston Hanson
# Place: Fred Hutch Cancer Center, Seattle, WA
# Date Created: 10/3/2025
# Purpose: To test how to train and test data in XGBoost

# Imports
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from itertools import product
import pickle
import os
import sys

def five_fold_cv(training_matrix, params, boost_rounds, nfold, metrics, verbose_eval, early_stopping):
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
        seed=42
    )

    return cv_results


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
    
