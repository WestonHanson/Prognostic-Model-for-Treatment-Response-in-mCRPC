# Author: Weston Hanson
# Place: Fred Hutch Cancer Center, Seattle, WA
# Date Created: 10/3/2025
# Purpose: To test how to train and test data in XGBoost

# Script imports
from training_functions import *

# Package imports
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import time
import matplotlib.pyplot as plt
import sys

# Filter out warnings
warnings.filterwarnings("ignore")

# Current time stamp for synchrinization for saved model and model parameters
curr_time = time.time()

# Load in data
diamonds = sns.load_dataset("diamonds")

print(f"diamonds dataset: {diamonds.head()}")

# Extract features and target arrays
X, y = diamonds.drop(['cut', 'color', 'clarity'], axis = 1), diamonds[['cut']]

print(f"X: \n{X}")
print()
print(f"y: \n{y}")

# Encode y to numeric (binary)
y_encoded = OrdinalEncoder().fit_transform(y)

print(f"y_encoded: \n{y_encoded}")


# Extract text features
cats = X.select_dtypes(exclude = 'number')

print(cats)

# Convert to Pandas category
for col in cats:
    X[col] = X[col].astype('category')

print(X.dtypes)

# Split the data (Automatically splits into 25:75, stratify keeps the same proporition of 0/1 in testing and training data)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, random_state = 1, stratify = y_encoded)

print(f"X_train: \n{X_train}")
print()
print(f"X_test: \n{X_test}")
print()
print(f"y_train: \n{y_train}")
print()
print(f"y_test: \n{y_test}")
print()

# Hyperparameter dictionary
param_dict = {
    'eta': [0.1],
    # 'max_depth': [3, 6],
    # 'min_child_weight': [1, 2],
    # 'max_delta_step': [0, 1],
    # 'subsample': [0.8, 1.0],
    # 'colsample_bytree': [0.8, 1.0]
}

objective = "multi:softprob"
tree_method = "hist"
metrics = ["mlogloss"]

# Train model with nested 5 fold cross validation
cv_results, file_dir = nested_five_fold_cv(
    X_train, 
    y_train,
    param_dict,
    objective,
    tree_method,
    1, # Number of boosts
    3, # Number of folds
    metrics,
    0, # Outputs model's performance every number boosts
    30, # Stops model if performance doesn't increase after number of boosts
    curr_time
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

# Save model as json
filename = f"./{file_dir}/xgb_model.json"
model.save_model(filename)
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

# Compute ROC curve and AUC
y_true_bin = label_binarize(y_test, classes=np.arange(5))
fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_predicted_prob.ravel())
roc_auc = auc(fpr, tpr)

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

plt.tight_layout()
plt.savefig(f"{file_dir}/roc_pr_curves.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"ROC AUC: {roc_auc:.3f}")
