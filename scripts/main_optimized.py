# Author: Weston Hanson
# Place: Fred Hutch Cancer Center, Seattle, WA
# Date Created: 10/7/2025
# Purpose: Read in genomic data and predict response 

# Script imports
from training_functions import *
from data_processing_functions import *
from plotting_functions import *

# Package imports
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import time
import matplotlib.pyplot as plt
import random
import os
import sys

# Filter out warnings
warnings.filterwarnings("ignore")

# Current time stamp for synchrinization for saved model and model parameters
curr_time = time.time()

# Set all seeds
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# !!!!!!!!!!!!!!!!!!
# LOAD DATA
# !!!!!!!!!!!!!!!!!!

pluvicto_master_sheet_file = "/fh/fast/ha_g/user/whanson/PSMA_Lutetium_whanson/genome_instability/data-files/pluvicto_survival_clean_with_response_groups.csv"
rna_seq_data_file = "/fh/fast/ha_g/user/whanson/PSMA_Lutetium_whanson/genome_instability/data-files/predictions_tumor_pluvicto.tsv"
tfbs_data_file = "/fh/fast/ha_g/user/whanson/PSMA_Lutetium_whanson/genome_instability/scripts/outputs/data-tables/heatmap_data_table.tsv"
FGA_data_file = "/fh/fast/ha_g/user/whanson/PSMA_Lutetium_whanson/genome_instability/scripts/outputs/data-tables/FGA_data_table.tsv"

pluvicto_master_sheet = pd.read_csv(pluvicto_master_sheet_file, index_col=0)
rna_seq_data = pd.read_table(rna_seq_data_file, sep="\t", index_col = 0)
tfbs_data = pd.read_table(tfbs_data_file, sep="\t", index_col = 0)
FGA_data = pd.read_table(FGA_data_file, sep="\t", index_col=0)

# !!!!!!!!!!!!!!!!!!!!
# USER IMPUT
# !!!!!!!!!!!!!!!!!!!!

genomic_choice = "tfbs"

tfx_cutoff = 0.03
# progression_group_survival_days_252_cutoff	progression_group_survival_days_median	progression_group_survival_days_quartile	
# progression_group_PSA_prog_days_median	progression_group_PSA_prog_days_quartile	progression_group_PSA_Progression	
# progression_group_tfx_prog_days_median	progression_group_tfx_prog_days_quartile	progression_group_T_cycles_1_vs_6	
# progression_group_T_cycles_2_vs_6	progression_group_T_cycles_1_2_vs_5_6	progression_group_T_cycles_1-5_vs_6

responder_group = "progression_group_survival_days_252_cutoff"

predictor_values = ["TFx_C1", "LOH.Score_C1", "TMB_C1"]

# Hyperparameter dictionary
param_ranges = {
    'eta': (0.001, 0.3),  # Log scale will be used
    'max_depth': (3, 10),
    'min_child_weight': (1, 10),
    'max_delta_step': (0, 5),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0),
    'gamma': (0, 1.0),
    'lambda': (0.5, 5.0),
    'alpha': (0, 2.0)
}

# Number of Bayesian optimization trials per fold
n_trials = 50  # Adjust based on time constraints

# Model args
objective = "binary:logistic"
tree_method = "hist"
metrics = ['auc', 'logloss', 'error',  'aucpr']

# ********************
# PROCESS DATA
# ********************

# Process either rna-seq or tfbs dat
if genomic_choice == "tfbs":
    tfbs_data = process_tfbs_data(tfbs_data)
    genomic_data = tfbs_data
else:
    genomic_data = rna_seq_data

print(f"genomic_data: \n{genomic_data}")
print()

# Combine responder_group and predictor_values
predictor_values.append(responder_group)

# Remove patients at C2 and with less than 10% TFx
genomic_data, pluvicto_master_sheet = clean_patient_names(genomic_data, pluvicto_master_sheet, predictor_values, "C2", tfx_cutoff)
print(f"genomic_data: \n{genomic_data}")
print()
print(f"pluvicto_master_sheet: \n{pluvicto_master_sheet}")
print()

FGA_column_filtered = extract_FGA(FGA_data, "C1")
print(f"FGA_column_filtered: \n{FGA_column_filtered}")
print(f"length: \n{len(FGA_column_filtered)}")
print()

# Append responder group to predictor_value and Combine dataframes
combined_dfs = combine_dataframes(genomic_data, pluvicto_master_sheet, FGA_data, predictor_values)

print(f"combined_dfs: \n{combined_dfs}")
print()

# Seperate cohorts
ltbx_cohort_filter = combined_dfs.index.str.contains("FHL")
mc_cohort = combined_dfs[~ltbx_cohort_filter]
ltbx_cohort = combined_dfs[ltbx_cohort_filter]

# For reproducibility
mc_cohort = mc_cohort.sort_index()
ltbx_cohort = ltbx_cohort.sort_index()

print(f"mc_cohort: \n{mc_cohort}\n")
print(f"ltbx_cohort: \n{ltbx_cohort}\n")

print(f"mc_cohort.index: \n{mc_cohort.index}\n")

# Split the data (Automatically splits into 25:75, stratify keeps the same proporition of 0/1 in testing and training data)
mc_train_indices, mc_test_indices = train_test_split(
        mc_cohort.index, 
        test_size=0.25, 
        random_state=RANDOM_SEED, 
        stratify=mc_cohort[responder_group]
    )
print(f"mc_train_indices: \n{mc_train_indices}\n")
print(f"mc_train_indices proportion: \n{(len(mc_train_indices))/(len(mc_train_indices)+len(mc_test_indices))}\n")
print(f"mc_train_indices: \n{mc_test_indices}\n")
print(f"mc_test_indices proportion: \n{(len(mc_test_indices))/(len(mc_train_indices)+len(mc_test_indices))}\n")

mc_train = mc_cohort.loc[mc_train_indices]
mc_test = mc_cohort.loc[mc_test_indices]
print(f"mc_train: \n{mc_train}\n")
print(f"mc_test: \n{mc_test}\n")

# Save responder groups before normalizing
y_train_mc = mc_train[[responder_group]].copy()
y_test_mc = mc_test[[responder_group]].copy()
y_ltbx = ltbx_cohort[[responder_group]].copy()
print(f"y_train_mc: \n{type(y_train_mc)}\n")
print(f"y_test_mc: \n{y_test_mc}\n")
print(f"y_ltbx: \n{y_ltbx}\n")

# Save columns that are already on a 0-1 scale
save_columns = [c for c in predictor_values if c not in [responder_group, "TMB_C1"]]
save_columns.append("genomic_instability")
print(save_columns)

print(mc_train.columns)

mc_train_extra_columns_to_save = mc_train[save_columns].copy()
mc_test_extra_columns_to_save = mc_test[save_columns].copy()
ltbx_cohort_extra_columns_to_save = ltbx_cohort[save_columns].copy()
print(f"mc_train_extra_columns_to_save: \n{mc_train_extra_columns_to_save}\n")
print(f"mc_test_extra_columns_to_save: \n{mc_test_extra_columns_to_save}\n")
print(f"ltbx_cohort_extra_columns_to_save: \n{ltbx_cohort_extra_columns_to_save}\n")


# Drop responder and saved columns
columns_to_drop = save_columns + [responder_group]
X_train_mc_df = mc_train.drop(columns=columns_to_drop)
X_test_mc_df = mc_test.drop(columns=columns_to_drop)
X_ltbx_df = ltbx_cohort.drop(columns=columns_to_drop)
print(f"X_train_mc_df: \n{X_train_mc_df}\n")
print(f"X_test_mc_df: \n{X_test_mc_df}\n")
print(f"X_ltbx_df: \n{X_ltbx_df}\n")

# Normalize along the features axis 
scaler = StandardScaler()
X_train_mc = scaler.fit_transform(X_train_mc_df)
X_test_mc = scaler.transform(X_test_mc_df)
X_ltbx = scaler.transform(X_ltbx_df)
print(f"X_train_mc: \n{X_train_mc}\n")
print(f"X_test_mc: \n{X_test_mc}\n")
print(f"X_ltbx: \n{X_ltbx}\n")

# Convert back to dataframe
X_train_mc = pd.DataFrame(X_train_mc, columns=X_train_mc_df.columns, index=X_train_mc_df.index)
X_test_mc = pd.DataFrame(X_test_mc, columns=X_test_mc_df.columns, index=X_test_mc_df.index)
X_ltbx = pd.DataFrame(X_ltbx, columns=X_ltbx_df.columns, index=X_ltbx_df.index)
print(f"X_train_mc: \n{X_train_mc.shape}\n")
print(f"X_test_mc: \n{X_test_mc.shape}\n")
print(f"X_ltbx: \n{X_ltbx.shape}\n")

# Merge extra saved columns
X_train_mc = pd.merge(X_train_mc, mc_train_extra_columns_to_save, left_index=True, right_index=True, how='inner')
X_test_mc = pd.merge(X_test_mc, mc_test_extra_columns_to_save, left_index=True, right_index=True, how='inner')
X_ltbx = pd.merge(X_ltbx, ltbx_cohort_extra_columns_to_save, left_index=True, right_index=True, how='inner')
print(f"X_train_mc: \n{X_train_mc}\n")
print(f"X_test_mc: \n{X_test_mc}\n")
print(f"X_ltbx: \n{X_ltbx}\n")

# Encode responder groups as 0/1
encoder = OrdinalEncoder(categories=[["non-responder", "responder"]])
y_train_mc_encoded = encoder.fit_transform(y_train_mc)
y_test_mc_encoded = encoder.transform(y_test_mc)
y_ltbx_encoded = encoder.transform(y_ltbx)
print(f"y_train_mc_encoded: \n{y_train_mc_encoded}\n")
print(f"y_test_mc_encoded: \n{y_test_mc_encoded}\n")
print(f"y_ltbx_encoded: \n{y_ltbx_encoded}\n")

X_train_mc, X_test_mc, X_ltbx = prepare_categorical_features(X_train_mc), prepare_categorical_features(X_test_mc), prepare_categorical_features(X_ltbx)

print(f"X_train_mc: \n{X_train_mc}")
print()
print(f"X_test_mc: \n{X_test_mc}")
print()
print(f"y_train_mc_encoded: \n{y_train_mc_encoded}")
print()
print(f"y_test_mc_encoded: \n{y_test_mc_encoded}")
print()
print(f"X_ltbx: \n{X_ltbx}")
print()
print(f"y_ltbx_encoded: \n{y_ltbx_encoded}")
print()

if X_train_mc.columns.equals(X_test_mc.columns) and X_test_mc.columns.equals(X_ltbx.columns):
    print("Same")
else:
    print("different")


# *******************
# TRAIN MODEL
# *******************

# Train model with nested 5 fold cross validation
cv_results, file_dir = nested_five_fold_cv_bayesian(
    X_train_mc,
    y_train_mc_encoded,
    param_ranges,
    objective,
    tree_method,
    800, # Number of boosts
    5, # Number of folds
    metrics,
    0, # Outputs model's performance every number boosts
    30, # Stops model if performance doesn't increase after number of boosts
    curr_time, # For saving model parameters
    n_trials=n_trials,
    random_state=RANDOM_SEED,
)

# Print out results
print(f"\nResults: \n{cv_results}")
print()

# Find best parameters
median_params, best_boost_round = get_median_params(cv_results)

print(f"median_params: \n{median_params}\n")

dtrain_mc_train = xgb.DMatrix(X_train_mc, label = y_train_mc_encoded, enable_categorical = True)
print(f"dtrain_clf: \n{dtrain_mc_train}")

# Train model
model = xgb.train(
    dtrain = dtrain_mc_train,
    num_boost_round = best_boost_round,
    params = {
        **median_params,
    },
)

print("mc training model complete")

# Save model as json
filename = f"./{file_dir}/xgb_model_mc_training.json"
model.save_model(filename)
print("saved model")

# Plot feature importance
feature_importance = model.get_score(importance_type='gain')
print(f"feature_importance: {feature_importance}")

# ****************************************************
# TEST AND EVALUATE MODEL ON 1ST COHORT (HOLD OUT SET)
# ****************************************************

# Make predictions on test set
dtest_mc_cohort = xgb.DMatrix(X_test_mc, label=y_test_mc_encoded, enable_categorical=True)
y_predicted_prob = model.predict(dtest_mc_cohort)

# Finds best prediction from models output
if objective == "multi:softprob":
    y_pred = np.argmax(y_predicted_prob, axis = 1)
else:
    y_pred = (y_predicted_prob > 0.5).astype(int)

# Evaluate accuracy of model
accuracy = accuracy_score(y_test_mc_encoded, y_pred)
print(f"\nTest Accuracy: {accuracy}")
print()

classfication = classification_report(y_test_mc_encoded, y_pred)
print(f"Classification: \n{classfication}")
print()

conf_matrix = confusion_matrix(y_test_mc_encoded, y_pred)
print(f"Confusion matrix: \n{conf_matrix}")

# Compute ROC curve and AUC
fpr, tpr, roc_thresholds = roc_curve(y_test_mc_encoded, y_predicted_prob)
roc_auc = auc(fpr, tpr)

# Compute Precision–Recall curve and AUC
precision, recall, pr_thresholds = precision_recall_curve(y_test_mc_encoded, y_predicted_prob)
pr_auc = average_precision_score(y_test_mc_encoded, y_predicted_prob)

file_name = "test_mc_cohort_roc_auc_curve"
create_ROC_AUC_precision_recall_curves(fpr, tpr, roc_auc, precision, recall, pr_auc, file_dir, file_name)

# ******************************
# TRAIN MODEL ON FULL 1ST COHORT
# ******************************

# Normalize whole cohort at once
y_mc_full = mc_cohort[[responder_group]].copy()

# Drop responder column 
X_mc_full_df = mc_cohort.drop(columns=responder_group)

# Save columns and indexs before scaling
X_mc_full_df_columns = X_mc_full_df.columns
X_mc_full_df_index = X_mc_full_df.index

mc_cohort_norm = scaler.fit_transform(X_mc_full_df)

# Convert back to dataframe
X_mc_full_df = pd.DataFrame(mc_cohort_norm, columns=X_mc_full_df_columns, index=X_mc_full_df_index)

# Split data for training
y_mc_full_encoded = encoder.fit_transform(y_mc_full)

# Prepare categorical features
X_mc_full = prepare_categorical_features(X_mc_full_df)

dtest_mc_full = xgb.DMatrix(X_mc_full, label=y_mc_full_encoded, enable_categorical=True)
print(f"full_dtest_mc_cohort: \n{dtest_mc_full}")

# Train model
full_model = xgb.train(
    dtrain = dtest_mc_full,
    num_boost_round = best_boost_round,
    params = {
        **median_params,
    },
)

print("full mc model complete")

# Save model as json
filename = f"./{file_dir}/xgb_model_full_mc.json"
full_model.save_model(filename)
print("saved model")

# Plot feature importance
feature_importance = full_model.get_score(importance_type='gain')
print(f"feature_importance: {feature_importance}")

# *************************************
# TEST AND EVALUATE MODEL ON 2ND COHORT
# *************************************

# Reset order of X_ltbx columns
print(f"X_ltbx_columns_before: {X_ltbx.columns}")
X_ltbx = X_ltbx[X_mc_full.columns]
print(f"X_ltbx_columns_after: {X_ltbx.columns}")

full_ltbx_cohort = xgb.DMatrix(X_ltbx, label=y_ltbx_encoded, enable_categorical=True)

# Make predictions on ltbx cohort
y_predicted_prob = full_model.predict(full_ltbx_cohort)

print(f"\npredicted prob: {y_predicted_prob}\n")

# Finds best prediction from models output
if objective == "multi:softprob":
    y_pred = np.argmax(y_predicted_prob, axis = 1)
else:
    y_pred = (y_predicted_prob > 0.5).astype(int)

print(f"y_pred: {y_pred}\n")
print(f"y_true: {y_ltbx_encoded}\n")

# Evaluate accuracy of model
accuracy = accuracy_score(y_ltbx_encoded, y_pred)
print(f"\nTest Accuracy: {accuracy}")
print()

classfication = classification_report(y_ltbx_encoded, y_pred)
print(f"Classification: \n{classfication}")
print()

conf_matrix = confusion_matrix(y_ltbx_encoded, y_pred)
print(f"Confusion matrix: \n{conf_matrix}")

# Compute ROC curve and AUC
fpr, tpr, roc_thresholds = roc_curve(y_ltbx_encoded, y_predicted_prob)
roc_auc = auc(fpr, tpr)

# Compute Precision–Recall curve and AUC
precision, recall, pr_thresholds = precision_recall_curve(y_ltbx_encoded, y_predicted_prob)
pr_auc = average_precision_score(y_ltbx_encoded, y_predicted_prob)

file_name = "full_model_roc_auc_curve"
create_ROC_AUC_precision_recall_curves(fpr, tpr, roc_auc, precision, recall, pr_auc, file_dir, file_name)

print(f"ROC AUC: {roc_auc:.3f}")
print(f"PR AUC: {pr_auc:.3f}")


