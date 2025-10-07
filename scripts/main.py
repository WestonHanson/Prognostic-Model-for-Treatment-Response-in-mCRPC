# Author: Weston Hanson
# Place: Fred Hutch Cancer Center, Seattle, WA
# Date Created: 10/7/2025
# Purpose: Read in RNA-seq data and predict response 

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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys

# Filter out warnings
warnings.filterwarnings("ignore")

# !!!!!!!!!
# Load data
# !!!!!!!!!
pluvicto_master_sheet_file = "/fh/fast/ha_g/user/whanson/PSMA_Lutetium_whanson/genome_instability/data-files/pluvicto_survival_clean_with_response_groups.csv"
rna_seq_data_file = "/fh/fast/ha_g/user/whanson/PSMA_Lutetium_whanson/genome_instability/data-files/predictions_tumor_pluvicto.tsv"

pluvicto_master_sheet = pd.read_csv(pluvicto_master_sheet_file, index_col = 0)
rna_seq_data = pd.read_table(rna_seq_data_file, sep="\t", index_col = 0)

# !!!!!!!!!!!
# User imputs
# !!!!!!!!!!!
responder_group = "progression_group_survival_days_252_cutoff"

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
metrics = ["mlogloss"] # Only supports one metric

# ***************
# Data processing
# ***************

# Remove patients at C2 and with less than 10% TFx
rna_seq_data, pluvicto_master_sheet = clean_patient_names(rna_seq_data, pluvicto_master_sheet, "C2", 0.10)

# Combine dataframes
combined_dfs = combine_dataframes(rna_seq_data, pluvicto_master_sheet, ["TFx_C1", "LOH.Score_C1", "TMB_C1", responder_group])

# Seperate cohorts
ltbx_cohort_filter = combined_dfs.index.str.contains("FHL")
mc_cohort = combined_dfs[~ltbx_cohort_filter]
ltbx_cohort = combined_dfs[ltbx_cohort_filter]

# Extract features and target arrays
X_mc_cohort, y_mc_cohort = mc_cohort.drop(columns = responder_group, axis = 1), mc_cohort[[responder_group]]

print(f"X_mc_cohort: \n{X_mc_cohort}")
print()
print(f"y_mc_cohort: \n{y_mc_cohort}")

# Encode y to numeric (binary)
y_encoded = OrdinalEncoder().fit_transform(y_mc_cohort)

print(f"y_encoded: \n{y_encoded}")

# Extract text features
cats = X_mc_cohort.select_dtypes(exclude = 'number')

print(cats)

# Convert to Pandas category
for col in cats:
    X_mc_cohort[col] = X_mc_cohort[col].astype('category')

print(X_mc_cohort.dtypes)

# Split the data (Automatically splits into 25:75, stratify keeps the same proporition of 0/1 in testing and training data)
X_train, X_test, y_train, y_test = train_test_split(X_mc_cohort, y_encoded, random_state = 1, stratify = y_encoded)

print(f"X_train: \n{X_train}")
print()
print(f"X_test: \n{X_test}")
print()
print(f"y_train: \n{y_train}")
print()
print(f"y_test: \n{y_test}")
print()

# **************
# Start modeling
# **************

# Train model with nested 5 fold cross validation
cv_results = nested_five_fold_cv(
    X_train, 
    y_train,
    param_dict,
    objective,
    tree_method,
    800, # Number of boosts
    5, # Number of folds
    metrics,
    0, # Outputs model's performance every number boosts
    30 # Stops model if performance doesn't increase after number of boosts
)

# Print out results
print(f"\nResults: \n{cv_results}")
print()

# Find best parameters
param_df = pd.DataFrame([r["best_params"] for r in cv_results])

median_params = param_df.median(numeric_only = True).to_dict()

int_cols = ["max_depth", "max_delta_step", "num_class"]
for c in int_cols:
    if c in median_params:
        median_params[c] = int(round(median_params[c]))

best_boost_round = int(np.median([r['best_round'] for r in cv_results]))
print(f"median_params: \n{median_params}")
print(f"best_boost_round: \n{best_boost_round}")

dtrain_clf = xgb.DMatrix(X_train, label = y_train, enable_categorical = True)
print(f"dtrain_clf: \n{dtrain_clf}")

# Train model
model = xgb.train(
    dtrain = dtrain_clf,
    num_boost_round = best_boost_round,
    params = {
        **median_params,
        'objective': objective,
        'eval_metric': metrics[0],
        'tree_method': tree_method
    },
)

print("model complete")

# Save model to json
model.save_model("xgb_model.json")
print("saved model")

# Make predictions on test set
dtest_clf = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
y_predicted_prob = model.predict(dtest_clf)

# Finds best prediction from models output
y_pred = np.argmax(y_predicted_prob, axis = 1)

# Evaluate accuracy of model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy}")
print()

classfication = classification_report(y_test, y_pred)
print(f"Classification: \n{classfication}")
print()

conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion matrix: \n{conf_matrix}")