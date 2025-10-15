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

master_sheet_file = "/path/to/meta/data" 
rna_seq_data_file = "/path/to/rna/seq/data" # Data must be in CPMs
tfbs_data_file = "/path/to/tfbs/data"
FGA_data_file = "/path/to/FGA/data"

master_sheet = pd.read_csv(master_sheet_file, index_col=0)
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

# Number of boosts
num_of_boosts = 800
# Number of folds
num_of_folds = 5
# Stops model if performance doesn't increase after number of boosts
num_of_early_stopping = 30 

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
genomic_data, master_sheet = clean_patient_names(genomic_data, master_sheet, predictor_values, "C2", tfx_cutoff)
print(f"genomic_data: \n{genomic_data}")
print()
print(f"master_sheet: \n{master_sheet}")
print()

FGA_column_filtered = extract_FGA(FGA_data, "C1")
print(f"FGA_column_filtered: \n{FGA_column_filtered}")
print(f"length: \n{len(FGA_column_filtered)}")
print()

# Append responder group to predictor_value and Combine dataframes
combined_dfs = combine_dataframes(genomic_data, master_sheet, FGA_data, predictor_values)

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

# Drop responder column 
X_train_mc_df = mc_train.drop(columns=responder_group)
X_test_mc_df = mc_test.drop(columns=responder_group)
X_ltbx_df = ltbx_cohort.drop(columns=responder_group)
print(f"X_train_mc_df: \n{X_train_mc_df.index}\n")
print(f"X_test_mc_df: \n{X_test_mc_df.index}\n")
print(f"X_ltbx_df: \n{X_ltbx_df.index}\n")

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

print(f"X_train_mc: \n{X_train_mc.isna().any().any()}")
print(f"X_test_mc: \n{X_test_mc.isna().any().any()}")
print(f"y_train_mc_encoded: \n{np.isnan(y_train_mc_encoded)}")
print(f"y_test_mc_encoded: \n{np.isnan(y_test_mc_encoded)}")
print(f"X_ltbx: \n{X_ltbx.isna().any().any()}")
print(f"y_ltbx_encoded: \n{np.isnan(y_ltbx_encoded)}")

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
    num_of_boosts, # Number of boosts
    num_of_folds, # Number of folds
    metrics,
    0, # Outputs model's performance every number boosts
    num_of_early_stopping, # Stops model if performance doesn't increase after number of boosts
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
print(f"best_boost_round: \n{best_boost_round}\n")

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


# ****************
# Debugging
# ****************

print("\n" + "="*80)
print("DEEP MODEL DEBUGGING")
print("="*80)

# 1. Check what the CV actually returned
print("\n1. CROSS-VALIDATION RESULTS:")
print("-" * 60)
for i, result in enumerate(cv_results):
    print(f"\nFold {i+1}:")
    print(f"  Best params: {result['best_params']}")
    print(f"  Best round: {result['best_round']}")
    print(f"  Accuracy: {result['accuracy']:.4f}")
    print(f"  Confusion matrix:\n{result['confusion_matrix']}")
    
    # Check if CV fold is also predicting all 1's
    cm = result['confusion_matrix']
    if cm.shape[0] == 2 and cm.shape[1] == 2:
        total_pred_0 = cm[0, 0] + cm[1, 0]  # All predictions of class 0
        total_pred_1 = cm[0, 1] + cm[1, 1]  # All predictions of class 1
        print(f"  Predictions: Class 0={total_pred_0}, Class 1={total_pred_1}")
        if total_pred_0 == 0:
            print("  ⚠️  WARNING: This fold also predicted all 1's!")

# 2. Check median parameters
print("\n2. MEDIAN PARAMETERS:")
print("-" * 60)
print(f"median_params: {median_params}")
print(f"best_boost_round: {best_boost_round}")

# Check if parameters are too restrictive
if 'max_depth' in median_params and median_params['max_depth'] <= 2:
    print("⚠️  WARNING: max_depth is very low, model may be too simple")
if 'min_child_weight' in median_params and median_params['min_child_weight'] >= 10:
    print("⚠️  WARNING: min_child_weight is very high, model may be too constrained")

# 3. Test the trained model on TRAINING data
print("\n3. MODEL PERFORMANCE ON TRAINING DATA:")
print("-" * 60)
train_pred_prob = model.predict(dtrain_mc_train)
train_pred = (train_pred_prob > 0.5).astype(int)

print(f"Training set size: {len(y_train_mc_encoded)}")
print(f"Probability stats:")
print(f"  Min: {train_pred_prob.min():.4f}")
print(f"  Max: {train_pred_prob.max():.4f}")
print(f"  Mean: {train_pred_prob.mean():.4f}")
print(f"  Std: {train_pred_prob.std():.4f}")

train_actual = y_train_mc_encoded.ravel().astype(int)
print(f"\nActual distribution: {np.bincount(train_actual)}")
print(f"Predicted distribution: {np.bincount(train_pred)}")

train_acc = accuracy_score(train_actual, train_pred)
print(f"Training accuracy: {train_acc:.4f}")

if train_acc < 0.6:
    print("⚠️  WARNING: Low training accuracy - model is not learning!")

# Show some individual predictions
print("\nFirst 20 training predictions:")
print("Index | Actual | Prob   | Pred | Match")
print("-" * 50)
for i in range(min(20, len(train_actual))):
    actual = train_actual[i]
    prob = train_pred_prob[i]
    pred = train_pred[i]
    match = "✓" if actual == pred else "✗"
    print(f"{i:5d} | {actual:6d} | {prob:.4f} | {pred:4d} | {match:^5s}")

# 4. Check if model has actually learned anything
print("\n4. MODEL STRUCTURE CHECK:")
print("-" * 60)

# Get model dump
model_dump = model.get_dump()
num_trees = len(model_dump)
print(f"Number of trees in model: {num_trees}")

if num_trees == 0:
    print("⚠️  CRITICAL ERROR: Model has no trees!")
elif num_trees < 10:
    print(f"⚠️  WARNING: Very few trees ({num_trees}), model may be undertrained")

# Check first tree
if num_trees > 0:
    first_tree = model_dump[0]
    print(f"\nFirst tree structure (truncated):")
    print(first_tree[:500] if len(first_tree) > 500 else first_tree)

# 5. Get feature importance
print("\n5. FEATURE IMPORTANCE:")
print("-" * 60)
try:
    feature_importance = model.get_score(importance_type='gain')
    if len(feature_importance) == 0:
        print("⚠️  WARNING: No feature importance scores!")
    else:
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        print(f"Top 10 features by gain:")
        for feat, score in sorted_features[:10]:
            # Map feature index back to name if possible
            if feat.startswith('f') and feat[1:].isdigit():
                feat_idx = int(feat[1:])
                if feat_idx < len(X_train_mc.columns):
                    feat_name = X_train_mc.columns[feat_idx]
                    print(f"  {feat_name} ({feat}): {score:.4f}")
                else:
                    print(f"  {feat}: {score:.4f}")
            else:
                print(f"  {feat}: {score:.4f}")
except Exception as e:
    print(f"⚠️  Error getting feature importance: {e}")

# 6. Manual prediction test
print("\n6. MANUAL PREDICTION TEST:")
print("-" * 60)
print("Testing with just the first test sample...")

# Get first test sample
first_test_sample = X_test_mc.iloc[[0]]
first_test_label = y_test_mc_encoded[0]

print(f"Actual label: {first_test_label}")
print(f"Feature values (first 10):")
for col in first_test_sample.columns[:10]:
    print(f"  {col}: {first_test_sample[col].values[0]:.4f}")

dtest_single = xgb.DMatrix(first_test_sample, enable_categorical=True)
single_pred_prob = model.predict(dtest_single)
single_pred = (single_pred_prob > 0.5).astype(int)

print(f"\nPredicted probability: {single_pred_prob[0]:.4f}")
print(f"Predicted class: {single_pred[0]}")

# 7. Check for leaf values
print("\n7. LEAF VALUE ANALYSIS:")
print("-" * 60)
# Get leaf predictions for training data to see if they're all similar
train_leaves = model.predict(dtrain_mc_train, pred_leaf=True)
print(f"Shape of leaf predictions: {train_leaves.shape}")
print(f"Unique leaf patterns in first tree: {len(np.unique(train_leaves[:, 0]))}")

if train_leaves.shape[1] > 0:
    # Check if all samples land in the same leaf
    unique_patterns = len(np.unique(train_leaves, axis=0))
    print(f"Total unique leaf patterns across all trees: {unique_patterns}")
    if unique_patterns <= 2:
        print("⚠️  WARNING: Very few unique leaf patterns - model is too simple!")

print("\n" + "="*80)
print("END DEEP DEBUGGING")
print("="*80 + "\n")

# ************************


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


