# Author: Weston Hanson
# Place: Fred Hutch Cancer Center, Seattle, WA
# Date Created: 10/7/2025
# Purpose: Read in genomic data and predict response 

# Script imports
from training_functions import *
from data_processing_functions import *

# Package imports
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import time
import matplotlib.pyplot as plt
import sys

# Filter out warnings
warnings.filterwarnings("ignore")

# Current time stamp for synchrinization for saved model and model parameters
curr_time = time.time()

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

responder_group = "progression_group_survival_days_252_cutoff"

predictor_values = ["LOH.Score_C1", "TMB_C1"]

# Hyperparameter dictionary
param_dict = {
    'eta': [0.1, 0.3],
    'max_depth': [3, 6],
    'min_child_weight': [1, 2],
    'max_delta_step': [0, 1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Model args
objective = "binary:logistic"
tree_method = "hist"
metrics = ["logloss"] # Only supports one metric

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

# Remove patients at C2 and with less than 10% TFx
genomic_data, pluvicto_master_sheet = clean_patient_names(genomic_data, pluvicto_master_sheet, "C2", tfx_cutoff)
print(f"genomic_data: \n{genomic_data}")
print()
print(f"pluvicto_master_sheet: \n{pluvicto_master_sheet}")
print()

FGA_column_filtered = extract_FGA(FGA_data, "C1")
print(f"FGA_column_filtered: \n{FGA_column_filtered}")
print()

# Append responder group to predictor_value and Combine dataframes
predictor_values.append(responder_group)
combined_dfs = combine_dataframes(genomic_data, pluvicto_master_sheet, FGA_data, predictor_values)

print(f"combined_dfs: \n{combined_dfs}")
print()

# Seperate cohorts
ltbx_cohort_filter = combined_dfs.index.str.contains("FHL")
mc_cohort = combined_dfs[~ltbx_cohort_filter]
ltbx_cohort = combined_dfs[ltbx_cohort_filter]

# Split the data (Automatically splits into 25:75, stratify keeps the same proporition of 0/1 in testing and training data)
mc_train_indices, mc_test_indices = train_test_split(
        mc_cohort.index, 
        test_size=0.25, 
        random_state=1, 
        stratify=mc_cohort[responder_group]
    )

mc_train = mc_cohort.loc[mc_train_indices]
mc_test = mc_cohort.loc[mc_test_indices]

# Normalize
mc_train_norm = standard_scaling(mc_train, responder_group, axis=1)
mc_test_norm = standard_scaling(mc_test, responder_group, axis=1)
ltbx_norm = standard_scaling(ltbx_cohort, responder_group, axis=1)

# Extract features and target arrays
X_train_mc, y_train_mc = mc_train_norm.drop(columns = responder_group, axis = 1), mc_train_norm[[responder_group]]
X_test_mc, y_test_mc = mc_test_norm.drop(columns = responder_group, axis = 1), mc_test_norm[[responder_group]]
X_ltbx, y_ltbx = ltbx_norm.drop(columns = responder_group, axis = 1), ltbx_norm[[responder_group]]

encoder = OrdinalEncoder()
y_train_mc_encoded = encoder.fit_transform(y_train_mc)
y_test_mc_encoded = encoder.fit_transform(y_test_mc)
y_ltbx_encoded = encoder.fit_transform(y_ltbx)

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

# *******************
# TRAIN MODEL
# *******************

# Train model with nested 5 fold cross validation
cv_results, file_dir = nested_five_fold_cv(
    X_train_mc, 
    y_train_mc_encoded,
    param_dict,
    objective,
    tree_method,
    800, # Number of boosts
    5, # Number of folds
    metrics,
    0, # Outputs model's performance every number boosts
    30, # Stops model if performance doesn't increase after number of boosts
    curr_time, # For saving model parameters
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
        'objective': objective,
        'eval_metric': metrics[0],
        'tree_method': tree_method
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

# ******************************
# TRAIN MODEL ON FULL 1ST COHORT
# ******************************

# Normalize whole cohort at once
mc_cohort_norm = standard_scaling(mc_cohort, responder_group, 0)

# Split data for training
X_mc_full, y_mc_full = mc_cohort_norm.drop(columns=responder_group, axis=1), mc_cohort_norm[[responder_group]]
y_mc_full_encoded = encoder.fit_transform(y_mc_full)

# Prepare categorical features
X_mc_full = prepare_categorical_features(X_mc_full)

dtest_mc_full = xgb.DMatrix(X_mc_full, label=y_mc_full_encoded, enable_categorical=True)
print(f"full_dtest_mc_cohort: \n{dtest_mc_full}")

# Train model
full_model = xgb.train(
    dtrain = dtest_mc_full,
    num_boost_round = best_boost_round,
    params = {
        **median_params,
        'objective': objective,
        'eval_metric': metrics[0],
        'tree_method': tree_method
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

full_ltbx_cohort = xgb.DMatrix(X_ltbx, label=y_ltbx_encoded, enable_categorical=True)

# Make predictions on ltbx cohort
y_predicted_prob = full_model.predict(full_ltbx_cohort)

# Finds best prediction from models output
if objective == "multi:softprob":
    y_pred = np.argmax(y_predicted_prob, axis = 1)
else:
    y_pred = (y_predicted_prob > 0.5).astype(int)

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

# Create figure
plt.figure(figsize=(12, 5))

# --- ROC Curve ---
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

# --- Precision–Recall Curve ---
plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='green', lw=2, label=f'PR (AUC = {pr_auc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision–Recall Curve')
plt.legend(loc='lower left')

plt.tight_layout()
plt.savefig(f"{file_dir}/roc_pr_curves.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"ROC AUC: {roc_auc:.3f}")
print(f"PR AUC: {pr_auc:.3f}")