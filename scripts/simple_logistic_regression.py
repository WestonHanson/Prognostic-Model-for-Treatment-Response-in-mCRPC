from plotting_functions import *

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import sys
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import os
import math

def logistic_regression(X_train_mc, y_train_mc_encoded, X_ltbx, y_ltbx_encoded, predictor_values, responder_group, version_dir, model_name, use_elastic_net, random_seed):

    # Create directory inside version_dir
    logreg_version_dir = version_dir + f"/logistic-regression/{model_name}"

    # Make directory
    os.makedirs(logreg_version_dir, exist_ok=True)

    # Prepare groupings for logistic regression
    mc_cohort_TFx = X_train_mc
    mc_cohort_response = y_train_mc_encoded

    ltbx_cohort_TFx = X_ltbx
    ltbx_cohort_response = y_ltbx_encoded

    # Turn predictor values into a title
    seperator = "_"
    predictor_values_title = seperator.join(str(item) for item in predictor_values)

    # Instantiate logistic regression 
    if use_elastic_net:
        logreg = LogisticRegressionCV(cv=3, penalty='elasticnet', l1_ratios=[0.1, 0.5, 0.9], solver='saga')
    else:
        logreg = LogisticRegression(penalty="l2", solver="liblinear", random_state=random_seed)

    print(f"\nmc_cohort_TFx: \n{mc_cohort_TFx}")
    print(f"\nmc_cohort_response: \n{mc_cohort_response}")

    # Fit the model
    logreg.fit(mc_cohort_TFx, mc_cohort_response)

    # Find feature names and coefficientes
    feature_names = mc_cohort_TFx.columns  # assuming it's a pandas DataFrame
    coefficients = logreg.coef_[0]         # shape (1, n_features) for binary classification
    feature_importance_abs = dict(zip(feature_names, abs(coefficients))) # Captures absolute importatnce (magnitude, not direction)

    # Sort by featuer importance
    feature_importance_sorted = dict(
        sorted(feature_importance_abs.items(), key=lambda x: x[1], reverse=True)
    )

    # plot freature importantce
    fi_df = pd.DataFrame(list(feature_importance_sorted.items()), columns=["Feature", "Importance"])
    fi_df = fi_df.sort_values(by="Importance")
    plt.figure(figsize=(12, 5))
    sns.barplot(data=fi_df, x="Feature", y="Importance")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{logreg_version_dir}/logistic_regression_feature_importance_bargraph.png", dpi=300)

    # Get logistic regression predictions
    y_pred = logreg.predict_proba(mc_cohort_TFx)[:, 1]

    # Converts dataframe of "responder" and "non-responder" into a list of 1's and 0's
    # ltbx_cohort_response[responder_group] = (ltbx_cohort_response[responder_group] == "responder").astype(int)

    fpr, tpr, _ = roc_curve(mc_cohort_response, y_pred)
    roc_auc = roc_auc_score(mc_cohort_response, y_pred)

    precision, recall, pr_thresholds = precision_recall_curve(ltbx_cohort_response, y_pred)
    pr_auc = average_precision_score(ltbx_cohort_response, y_pred)

    file_name = f"logistic_regression_ROC_PR_Curves_for_{predictor_values_title}"
    create_ROC_AUC_precision_recall_curves(fpr, tpr, roc_auc, precision, recall, pr_auc, logreg_version_dir, file_name)

    return feature_importance_sorted, roc_auc

def logistic_regression_general(X_train_mc, y_train_mc_encoded, X_ltbx, y_ltbx_encoded, version_dir, model_name, feature_selection_method, random_seed):

    # Create directory inside version_dir
    logreg_version_dir = version_dir + f"/logistic-regression/{feature_selection_method}/{random_seed}/{model_name}"

    # Make directory
    os.makedirs(logreg_version_dir, exist_ok=True)

    # Prepare groupings for logistic regression
    mc_cohort_TFx = X_train_mc
    mc_cohort_response = y_train_mc_encoded

    ltbx_cohort_TFx = X_ltbx
    ltbx_cohort_response = y_ltbx_encoded

    # Instantiate logistic regression 
    logreg = LogisticRegression(penalty="l2", solver="liblinear", random_state=random_seed)

    # print(f"\nmc_cohort_TFx: \n{mc_cohort_TFx}")
    # print(f"\nmc_cohort_response: \n{mc_cohort_response}")

    # Fit the model
    logreg.fit(mc_cohort_TFx, mc_cohort_response)

    # Find feature names and coefficientes
    feature_names = mc_cohort_TFx.columns  # assuming it's a pandas DataFrame
    coefficients = logreg.coef_[0]         # shape (1, n_features) for binary classification
    feature_importance_abs = dict(zip(feature_names, abs(coefficients))) # Captures absolute importatnce (magnitude, not direction)

    # Sort by featuer importance
    feature_importance_sorted = dict(
        sorted(feature_importance_abs.items(), key=lambda x: x[1], reverse=True)
    )

    # plot freature importantce
    fi_df = pd.DataFrame(list(feature_importance_sorted.items()), columns=["Feature", "Importance"])
    fi_df = fi_df.sort_values(by="Importance")
    plt.figure(figsize=(12, 5))
    sns.barplot(data=fi_df, x="Feature", y="Importance")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{logreg_version_dir}/logistic_regression_feature_importance_bargraph.png", dpi=300)

    # Get logistic regression predictions
    y_pred = logreg.predict_proba(ltbx_cohort_TFx)[:, 1]
    
    fpr, tpr, _ = roc_curve(ltbx_cohort_response, y_pred)
    roc_auc = roc_auc_score(ltbx_cohort_response, y_pred)

    precision, recall, pr_thresholds = precision_recall_curve(ltbx_cohort_response, y_pred)
    pr_auc = average_precision_score(ltbx_cohort_response, y_pred)

    file_name = f"logistic_regression_ROC_PR_Curves_for_{feature_selection_method}"
    create_ROC_AUC_precision_recall_curves(fpr, tpr, roc_auc, precision, recall, pr_auc, logreg_version_dir, file_name)

    return feature_importance_sorted, roc_auc
