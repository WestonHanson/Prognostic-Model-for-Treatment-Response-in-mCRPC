# Author: Weston Hanson
# Place: Fred Hutch Cancer Center, Seattle, WA
# Date Created: 10/7/2025
# Purpose: Read in genomic data and predict response 

# Suppress warnings early (before XGBoost import)
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# Script imports
from training_functions import *
from data_processing_functions import *
from plotting_functions import *
from simple_logistic_regression import *
from debugging_functions import *
from file_management_functions import *

# Package imports
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu
import statsmodels.stats.multitest as mt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import time
import matplotlib.pyplot as plt
import random
import json
import os
import sys
import ast
from joblib import Parallel, delayed
import statistics

# Current time stamp for synchrinization for saved model and model parameters
curr_time = time.time()

# !!!!!!!!!!!!!!!!!!
# LOAD DATA
# !!!!!!!!!!!!!!!!!!

# You can change the name here and add more or less - need to update PROCESS DATA section if you change these variables 
pluvicto_master_sheet_file = "full/path/to/pluvicto_master_sheet_file"
tfbs_data_file = "full/path/to/tfbs_data_file"
FGA_data_file = "full/path/to/FGA_data_file"
clinical_data_file = "full/path/to/clinical_data_file"
proteus_gsva_data_file = "full/path/to/proteus_gsva_data_file"
entropy_data_file = "full/path/to/entropy_data_file"

pluvicto_master_sheet_og = pd.read_csv(pluvicto_master_sheet_file, index_col=0)
tfbs_data_og = pd.read_table(tfbs_data_file, sep="\t", index_col = 0)
FGA_data_og = pd.read_table(FGA_data_file, sep="\t", index_col=0)
clinical_data_og = pd.read_csv(clinical_data_file, index_col=0)
proteus_gsva_data_og = pd.read_table(proteus_gsva_data_file, sep='\t', index_col=0)
entropy_data_og = pd.read_csv(entropy_data_file, index_col=0)

# !!!!!!!!!!!!!!!!!!!!
# USER IMPUT
# !!!!!!!!!!!!!!!!!!!!

# Name model version
model_ver = "v0.0.0"
model_subver = "v0.0.1"

# ********************
# VERSION MAINTENANCE
# ********************

version_dir, use_model = model_version_directory(model_ver, model_subver)

# **************
# READ IN JSON
# **************

with open("../input/input.json", 'r') as file:
    input_json = json.load(file)

# ********************
# PRE-LOOP PROCESSING
# ********************

# Add all reponder/non-responder labels for all conditions
pluvicto_master_sheet_groups = add_responder_groupings(pluvicto_master_sheet_og)

# Setting unchanged parameters

# Hyperparameter dictionary
param_ranges = {
    'eta': (0.01, 0.05),
    'max_depth': (2, 5),
    'min_child_weight': (1, 3),
    'max_delta_step': (0, 1),
    'subsample': (0.6, 0.9),
    'colsample_bytree': (0.3, 0.7),
    'colsample_bylevel': (0.3, 0.8),
    'colsample_bynode': (0.3, 0.8),
    'gamma': (0, 0.5),
    'lambda': (1, 5),
    'alpha': (0.5, 2)
}

# Number of boosts
num_of_boosts = 500
# Number of folds
num_of_folds = 3
# Stops model if performance doesn't increase after number of boosts
num_of_early_stopping = 100 

# Base prediction
base_score = 0.5

# Number of Bayesian optimization trials per fold
n_trials = 20  # Adjust based on time constraints

# Model args
objective = "binary:logistic"
tree_method = "hist"
metrics = ['auc', 'logloss', 'error',  'aucpr']

# **************
# START OF LOOP
# **************

random_seed_list = list(range(42, 142))

inner_auc_dict = {}
outer_auc_dict = {}
testing_auc_dict = {}
conf_int = {}
if not use_model:
    for i, perm in enumerate(input_json, 1):
        try:
            
            # Select choice of genomic data to use
            # genomic_choice = input_json[perm]["genomic_choice"]
            feature_selection_methods = input_json[perm]["feature_selection_methods"]

            tfx_cutoff = input_json[perm]["tfx_cutoff"]

            responder_group = input_json[perm]["responder_group"]

            # Cycle to filter down to
            cycle_filter = input_json[perm]['cycle_filter']

            # Columns for each dataframe to filter down to
            pluvicto_master_sheet_cols = input_json[perm]["pluvicto_master_sheet_cols"]
            tfbs_data_cols = input_json[perm]["tfbs_data_cols"]
            FGA_data_cols = input_json[perm]["FGA_data_cols"]
            clinical_data_cols = input_json[perm]["clinical_data_cols"]
            proteus_gsva_data_cols = input_json[perm]["proteus_gsva_data_cols"]
            entropy_data_cols = input_json[perm]["entropy_data_cols"]

            # Boolean to subset TFBS data to only top variable features
            subset_for_top_features = input_json[perm]["subset_for_top_features"]

            # Select how many meta features you want to select for after training the model (eg. '1' would retrain the model on "TFx_C1", 2 would be retraining on "TFx_C1" and "LOH.Score_C1", etc. "-1" retrains on all meta features)
            predictor_subset_num = input_json[perm]["predictor_subset_num"]

            # Select how many important features you want to select for after training the model (eg. '1' would retrain the model on the most impotant feature from training on all/selective_features data, '2' retrains model on the top 2 features, '-1' means retraining on all top features, etc.)
            top_feature_num = input_json[perm]["top_feature_num"]

            # ********************
            # PROCESS DATA
            # ********************

            pluvicto_master_sheet = pluvicto_master_sheet_og
            tfbs_data = tfbs_data_og
            FGA_data = FGA_data_og
            clinical_data = clinical_data_og
            proteus_gsva_data = proteus_gsva_data_og
            entropy_data = entropy_data_og

            # --------------------
            # Pre-process data
            # --------------------
            # Transpose dataframes that need it
            proteus_gsva_data_T = proteus_gsva_data.T
            proteus_gsva_data = pd.DataFrame(proteus_gsva_data_T)

            # Add cycle column to dataframes that need it 
            proteus_gsva_data['cycle'] = proteus_gsva_data.index.to_series().str.extract(r'(C[0-9]+)$')

            # Set indexes to patient id column in dataframes that need it
            clinical_data = clinical_data.reset_index().set_index('Sample_ID')

            # ----------------------------
            # Standardized data processing
            # ----------------------------
            # Filter dataframes
            pluvicto_master_sheet_cleaned = clean_dataframe(
                df=pluvicto_master_sheet, 
                responder_group_col=responder_group, 
                cycle_col=None, 
                cycle_filter=None,
                tfx_col='TFx_C1',
                tfx_filter=0.03,
                strip_id_tags=False
            )
            tfbs_data_cleaned = clean_dataframe(
                df=tfbs_data, 
                responder_group_col=None, 
                cycle_col='time_point', 
                cycle_filter=1,
                tfx_col='tumor_fraction',
                tfx_filter=0.03,
                strip_id_tags=False
            )
            FGA_data_cleaned = clean_dataframe(
                df=FGA_data, 
                responder_group_col=None, 
                cycle_col='time_point', 
                cycle_filter='C1',
                tfx_col='tumor_fraction',
                tfx_filter=0.03,
                strip_id_tags=False
            )
            clinical_data_cleaned = clean_dataframe(
                df=clinical_data, 
                responder_group_col=None, 
                cycle_col=None, 
                cycle_filter=None,
                tfx_col=None,
                tfx_filter=None,
                strip_id_tags=False
            )
            proteus_gsva_data_cleaned = clean_dataframe(
                df=proteus_gsva_data, 
                responder_group_col=None, 
                cycle_col='cycle', 
                cycle_filter='C1',
                tfx_col=None,
                tfx_filter=None,
                strip_id_tags=True
            )
            entropy_data_cleaned = clean_dataframe(
                df=entropy_data, 
                responder_group_col=None, 
                cycle_col='cycle', 
                cycle_filter='C1',
                tfx_col=None,
                tfx_filter=None,
                strip_id_tags=True
            )

            # Filter columns to ones passed
            def subset_df(df, cols):
                if cols is None:
                    return df.iloc[:, 0:0]  # keep index, no columns
                if len(cols) == 0:
                    return df  # keep all columns
                return df[cols]  # subset
            
            pluvicto_master_sheet_subset = subset_df(pluvicto_master_sheet_cleaned, pluvicto_master_sheet_cols)
            tfbs_data_subset = subset_df(tfbs_data_cleaned, tfbs_data_cols)
            FGA_data_subset = subset_df(FGA_data_cleaned, FGA_data_cols)
            clinical_data_subset = subset_df(clinical_data_cleaned, clinical_data_cols)
            proteus_gsva_data_subset = subset_df(proteus_gsva_data_cleaned, proteus_gsva_data_cols)
            entropy_data_subset = subset_df(entropy_data_cleaned, entropy_data_cols)

            # Join the dataframes (subset them to the smallest patient list)
            combined_df = pd.concat(
                [pluvicto_master_sheet_subset, tfbs_data_subset, FGA_data_subset, clinical_data_subset, proteus_gsva_data_subset, entropy_data_subset],
                axis=1,
                join="inner"
            )

            # Split data 
            X_train, y_train_encoded, X_test, y_test_encoded, X_train_filtered_columns = cohort_X_y_splitting_encoding_and_transforming_v2(
                df=combined_df,
                responder_group_col=responder_group,
                responder_categories=['responder', 'non-responder'],
                split_prct=0.25,
                random_state=42
            )

            # Helper function to process a single random seed (for parallelization)
            def process_random_seed(random_seed):
                """Process model training and evaluation for a single random seed"""
                # Set all seeds
                random.seed(random_seed)
                np.random.seed(random_seed)
                os.environ['PYTHONHASHSEED'] = str(random_seed)
            
                # ******************
                # FEATURE SELECTION
                # ******************

                seed_results = {}  # Store results for this seed

                for feature_selection_method in feature_selection_methods:

                    # print("\n" + "="*50)
                    # print(f"Random Seed: {random_seed}, Feature Selection: {feature_selection_method}")
                    # print("="*50 + "\n")

                    default_pkl_file_name = "full_cv_training_parameters.pkl"

                    # *******************
                    # TRAIN MODEL
                    # *******************
                    
                    # ----------------
                    # Cross Validation
                    # ----------------

                    # Train model with nested 5 fold cross validation
                    cv_results, testing_roc_auc, file_dir = nested_five_fold_cv_bayesian_with_feature_selection(
                        X_train,
                        y_train_encoded,
                        param_ranges,
                        objective,
                        tree_method,
                        num_of_boosts, # Number of boosts
                        num_of_folds, # Number of folds
                        base_score,
                        metrics,
                        0, # Outputs model's performance every number boosts
                        num_of_early_stopping, # Stops model if performance doesn't increase after number of boosts
                        feature_selection_method,
                        X_train_filtered_columns,
                        version_dir,
                        default_pkl_file_name,
                        n_trials=n_trials,
                        random_state=random_seed,
                    )

                    # Find best parameters
                    median_params, best_boost_round = get_median_params(cv_results)

                    # -----------------------------------
                    # Finish training to get boost rounds
                    # -----------------------------------

                    # Turn all non-numeric columns to categorical
                    non_num_cols = X_train.select_dtypes(exclude=["number"]).columns
                    for col in non_num_cols:
                        X_train[col] = X_train[col].astype('category')

                    final_mc_train = xgb.DMatrix(X_train, label = y_train_encoded, enable_categorical=True)

                    # Train model
                    final_model = xgb.train(
                        dtrain = final_mc_train,
                        num_boost_round = best_boost_round,
                        params = {
                            **median_params,
                        },
                    )

                    # Save model as json
                    filename = f"./{file_dir}/xgb_final_model_mc.json"
                    final_model.save_model(filename)

                    # ***********************
                    # TEST AND EVALUATE MODEL
                    # ***********************

                    # Turn all non-numeric columns to categorical
                    non_num_cols = X_test.select_dtypes(exclude=["number"]).columns
                    for col in non_num_cols:
                        X_test[col] = X_test[col].astype('category')

                    full_test_cohort = xgb.DMatrix(X_test, label=y_test_encoded, enable_categorical=True)

                    # Make predictions on ltbx cohort
                    y_predicted_prob = final_model.predict(full_test_cohort)

                    # # Finds best prediction from models output
                    # if objective == "multi:softprob":
                    #     y_pred = np.argmax(y_predicted_prob, axis = 1)
                    # else:
                    #     median_value = np.median(y_predicted_prob)
                    #     y_pred = (y_predicted_prob > 0.5).astype(int)

                    # Compute ROC curve and AUC
                    fpr, tpr, roc_thresholds = roc_curve(y_test_encoded, y_predicted_prob)
                    roc_auc = auc(fpr, tpr)

                    # Compute Precision–Recall curve and AUC
                    precision, recall, pr_thresholds = precision_recall_curve(y_test_encoded, y_predicted_prob)
                    pr_auc = average_precision_score(y_test_encoded, y_predicted_prob)

                    file_name = "final_model_roc_auc_curve"
                    create_ROC_AUC_precision_recall_curves(fpr, tpr, roc_auc, precision, recall, pr_auc, file_dir, file_name)

                    print(f"ROC AUC: {roc_auc:.3f}")

                    seed_results[feature_selection_method] = {
                        "random_seed": random_seed,
                        "XGBoost": roc_auc
                    }
                
                return seed_results

            # Parallelize the random seed loop
            print(f"\nParallelizing model training across {len(random_seed_list)} random seeds...")
            seed_results_list = Parallel(n_jobs=-1, verbose=10)(
                delayed(process_random_seed)(seed) for seed in random_seed_list
            )

            # Combine results from all seeds
            for seed_result in seed_results_list:
                for method, results in seed_result.items():
                    random_seed = results['random_seed']
                    if method not in inner_auc_dict:
                        inner_auc_dict[method] = {}
                    inner_auc_dict[method][random_seed] = {
                        "XGBoost": results["XGBoost"]
                    }

            print(f"inner_auc_dict: {inner_auc_dict}")

            inner_xgboost_auc_list = {
                method: [list(seed_dict.values())[0] for seed_dict in seeds.values()]
                for method, seeds in inner_auc_dict.items()
            }

            xgboost_auc_list_conf_interval = {
                method: np.percentile(auc_list, [2.5,97.5]) 
                for method, auc_list in inner_xgboost_auc_list.items()
            }
            xgboost_auc_list_median = {
                method: statistics.median(auc_list) 
                for method, auc_list in inner_xgboost_auc_list.items()
            }

            conf_int[perm] = {
                "XGBoost": {
                    method: {
                        "CI": xgboost_auc_list_conf_interval[method], 
                        "median": xgboost_auc_list_median[method]
                        }
                    for method in xgboost_auc_list_conf_interval.keys()
                }
            }

        except Exception as e:
            print("Did not save AUC")
            print(f"error: {e}")
            outer_auc_dict[perm] = 0


    # Save conf_int
    completed_testing_dict_file = version_dir + "/completed_testing_dict.pkl"
    with open(completed_testing_dict_file, 'wb') as f:
        pickle.dump(conf_int, f)

# Load saved conf_int
else:
    completed_testing_dict_file = version_dir + "/completed_testing_dict.pkl"
    with open(completed_testing_dict_file, 'rb') as f:
        conf_int = pickle.load(f)

print(f"conf_int: \n{conf_int}\n")


# Below needs fixing

os.makedirs(f"{version_dir}/all_features_auc_bar_plot", exist_ok=True)

auc_graph_with_CI_publication(conf_int, version_dir, "all_features_auc_95_CI_plot")
