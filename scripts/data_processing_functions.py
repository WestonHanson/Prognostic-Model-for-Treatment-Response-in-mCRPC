# Author: Weston Hanson
# Place: Fred Hutch Cancer Center, Seattle, WA
# Date Created: 10/7/2025
# Purpose: To clean data for use in XGBoost

from simple_logistic_regression import *

import re
import pandas as pd
import statistics
import numpy as np
import sys
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.model_selection import StratifiedKFold, train_test_split
from boruta import BorutaPy
import math
from mrmr import mrmr_classif


def add_responder_groupings(pluvicto_master_sheet):
    """
    Parameters:
    -----------
        pluvicto_master_sheet: pandas DataFrame
            Data containing meta data.

    Function:
    ---------
        - Adds responder/non-responder label based on certain conditions listed in columns list.
        
    Returns:
    --------
        pluvicto_master_sheet: pandas DataFrame
            Data containing meta data and responder groupings.
    """
    columns = ["survival_days", "PSA_prog_days", "PSA_Progression", "tfx_prog_days", "T_cycles"]

    # Set Sample_ID as index
    try:
        pluvicto_master_sheet.set_index('Sample_ID', inplace=True)
    except Exception as e:
        print("\nSample_ID is already the index\n")
    
    for column in columns:
        # For binary indicators
        if column == "PSA_Progression":
            pluvicto_master_sheet[f"progression_group_{column}"] = pluvicto_master_sheet[column].apply(
                lambda x: "non-responder" if x == 1 else "responder"
            )
            continue
        
        if column == "T_cycles":
            column_temp = f"{column}_1_vs_6"
            pluvicto_master_sheet[f"progression_group_{column_temp}"] = pluvicto_master_sheet['T_cycles'].apply(
                lambda x: "responder" if x == 6 else ("non-responder" if x == 1 else np.nan)
            )
            
            column_temp = f"{column}_2_vs_6"
            pluvicto_master_sheet[f"progression_group_{column_temp}"] = pluvicto_master_sheet['T_cycles'].apply(
                lambda x: "responder" if x == 6 else ("non-responder" if x == 2 else np.nan)
            )
            
            column_temp = f"{column}_1_2_vs_5_6"
            pluvicto_master_sheet[f"progression_group_{column_temp}"] = pluvicto_master_sheet['T_cycles'].apply(
                lambda x: "responder" if x in [5, 6] else ("non-responder" if x in [1, 2] else np.nan)
            )
            
            column_temp = f"{column}_1-5_vs_6"
            pluvicto_master_sheet[f"progression_group_{column_temp}"] = pluvicto_master_sheet['T_cycles'].apply(
                lambda x: "responder" if x == 6 else ("non-responder" if x in [1, 2, 3, 4, 5] else np.nan)
            )
            continue
        
        if column == "survival_days":
            pluvicto_master_sheet[f"progression_group_{column}_252_cutoff"] = pluvicto_master_sheet[column].apply(
                lambda x: "non-responder" if x < 252 else "responder"
            )
        
        # Calculate quantiles on non-NA values
        quantile_df = pluvicto_master_sheet[pluvicto_master_sheet[column].notna()]
        quantile_vector = quantile_df[column].quantile([0.25, 0.5, 0.75])
        print(f"{column} {quantile_vector.values}")
        
        pluvicto_master_sheet[f"progression_group_{column}_median"] = pluvicto_master_sheet[column].apply(
            lambda x: "non-responder" if x < quantile_vector[0.5] else "responder"
        )
        
        pluvicto_master_sheet[f"progression_group_{column}_quartile"] = pluvicto_master_sheet[column].apply(
            lambda x: "non-responder" if x <= quantile_vector[0.25] else ("responder" if x >= quantile_vector[0.75] else np.nan)
        )
    
    return pluvicto_master_sheet

def clean_dataframe(df, responder_group_col, cycle_col, cycle_filter, tfx_col, tfx_filter, strip_id_tags):
    '''
    Parameters:
    -----------
        df: pandas DataFrame
            DataFrame positioned as patient_id x features (patient_id is index).

        responder_group_col: String
            Name of column to remove rows with no outcome data.

        cycle_col: String
            Name of cycle column.

        cycle_filter: String or int 
            Cycle to filter rows down to.

        tfx_col: String
            Name of TFx column.

        tfx_filter: float
            Filter out any rows less than tfx_filter.

        strip_id_tags: Bool
            True/False to strip any lagging tags on index ("patient_id column").

    Function:
    ---------
        - 
        - If cycle_filter is not None filter df down to only those rows that equal cycle_filter.
        - If tfx_filter is not None filter df down to only those rows greater than or equal to tfx_cutoff.
        - Drop columns and rows with all NAs.
        - If true remove tags on patient_id index if present
        - Return df

    Returns:
    --------
        df_copy: pandas DataFrame
            DataFrame with all filters applied. 
    '''
    # Copy df
    df_copy = df.copy()

    # Remove patients below cutoff
    if tfx_filter is not None:
        df_copy = df_copy[df_copy[tfx_col] >= tfx_filter]
    
    # Remove any patients that have TFx NA and columns with all NAs
    if responder_group_col is not None:
        df_copy = df_copy.dropna(subset=responder_group_col)

    # Remove patients based on timepoint
    if cycle_filter is not None:
        df_copy = df_copy[df_copy[cycle_col] == cycle_filter]

    # Drop columns with all NAs
    df_copy = df_copy.dropna(axis=1, how='all')

    # Drop rows with all NAs
    df_copy = df_copy.dropna(how='all')
    
    # Strip names of tags
    if strip_id_tags:
        original_names = df_copy.index.tolist()
        cleaned_names = [re.sub(r'_[A-Z]_C[0-9]+$', '', name) for name in original_names]
        df_copy.index = cleaned_names

    return df_copy

def get_median_params(cv_results):
    """
    Extract median parameters from cross-validation results.
    
    Parameters:
    -----------
        cv_results: list
            List of dictionaries from nested_five_fold_cv
            
    Returns:
    --------
        median_params: dict
            Dictionary of median parameter values
        best_boost_round: int
            Median best boost round
    """
    param_df = pd.DataFrame([r["best_params"] for r in cv_results])

    print(f"param_df: \n{param_df}\n")
    
    # Get median for numeric parameters only
    median_params = param_df.select_dtypes(include=[np.number]).median().to_dict()
    
    # Convert specific parameters to integers
    int_cols = ["max_depth", "max_delta_step", "seed"]
    for c in int_cols:
        if c in median_params:
            median_params[c] = int(round(median_params[c]))
    
    # Add non-numeric parameters from first fold
    for key in param_df.columns:
        if key not in median_params:
            median_params[key] = param_df[key].iloc[0]
    
    best_boost_round = int(np.median([r['best_round'] for r in cv_results]))
    
    return median_params, best_boost_round

def cohort_X_y_splitting_encoding_and_transforming_v2(df, responder_group_col, responder_categories, split_prct, random_state):
    '''
    Parameters:
    -----------
        df: pandas DataFrame
            DataFrame in patient_id x features (patient_id is index).

        responder_group_col: String
            Name of column for survival coding.

        responder_categories: List
            List of the responder labels (eg. ['responder', 'non-responder'])

        split_prct: float
            Percentage of the data that will be tested on.

        random_state: int
            Random seed to keep reproducibility. 

    Function:
    ---------
        - Split the data with "split_prct" of the data as training data and 1 - "split_prct" as testing data.
        - Obtain y_train, y_test, X_train, and X_test.
        - Find top variables from X_train.
        - Encode y_test and y_train.
        - Return all necessary variables. 
        
    Returns:
    --------

    '''
    df_copy = df.copy()

    # Split data into test and training splits (perserving class balance)
    train_df, test_df = train_test_split(
        df_copy,
        test_size=split_prct,
        random_state=random_state,
        stratify=df_copy[responder_group_col]
    )

    # Obtain y data
    y_train_df = train_df[[responder_group_col]].copy()
    y_test_df = test_df[[responder_group_col]].copy()

    # Obtain X data
    X_train_df = train_df.drop(columns=[responder_group_col])
    X_test_df = test_df.drop(columns=[responder_group_col])

    # Extract non-numeric columns before feature selection
    non_numeric_cols = X_train_df.copy().select_dtypes(exclude='number').columns.tolist()
    X_train_numeric = X_train_df.copy().select_dtypes(include='number')

    # Find top variable features from numeric features only
    top_variable_features = feature_selection_helper("variable_features", X_train_numeric)
    top_variable_features = top_variable_features + non_numeric_cols

    # Encode responder groups as 0/1
    encoder = OrdinalEncoder(categories=[responder_categories])
    y_train_encoded = encoder.fit_transform(y_train_df)
    y_test_df_encoded = encoder.transform(y_test_df)

    # Convert from list of lists to one list
    y_train_encoded = y_train_encoded.ravel().astype(int)
    y_test_df_encoded = y_test_df_encoded.ravel().astype(int)

    return X_train_df, y_train_encoded, X_test_df, y_test_df_encoded, top_variable_features


def feature_selection_helper(feature_method, df_X, df_y_encoded=None, random_seed=42):
    '''
    Parameters:
    -----------
        feature_method: String
            Method of feature selection.

        df: pandas DataFrame
            DataFrame to filter.
        

    Function:
    ---------
        - 

    Returns:
    --------
        genes_to_keep: list
            Genes from feature selection to keep.
    '''
    if feature_method == "variable_features":
        genes_to_keep = select_top_variable_features(df_X)
    elif feature_method == "absolute_PCA":
        genes_to_keep = PCA_top_features(df_X, df_y_encoded)
    elif feature_method == "boruta_rf":
        genes_to_keep = boruta_rf_feature_selection(df_X, df_y_encoded, random_seed)
    elif feature_method == "smv_rfe":
        genes_to_keep = smv_rfe(df_X, df_y_encoded, random_seed)
    elif feature_method == "mrmr":
        genes_to_keep = mrmr(df_X, df_y_encoded)
    elif feature_method == "ElasticNet":
        genes_to_keep = elastic_net_feature_selection(df_X, df_y_encoded, 3, random_seed)

    return genes_to_keep

def select_top_variable_features(df):
    '''
    Parameters:
    -----------
        df: pandas DataFrame
            DataFrame to filter.

    Function:
    ---------
        - If the length of feature sis less than half of the length of samples, return all columns as list.
        - Finds the variances of df down the columns (0 is column, 1 is row) and sorts df decending.
        - Grabs the top 20% of df.
        - Lists out the names of the genes and returns them.

    Returns:
    --------
        genes_to_keep: list
            Genes from feature selection to keep.
    '''
    # if the length of feature sis less than half of the length of samples, return all columns as list.
    if len(df.columns) < (len(df) / 2):
        genes_to_keep = df.columns.tolist()
        return genes_to_keep
    
    # Else find top 20% variable columns and return list
    variance_df = df.var(ddof=0, axis=0).sort_values(ascending=False)
    top_variable_features = variance_df.head(math.ceil(len(variance_df)*.2))
    genes_to_keep = top_variable_features.index.tolist()

    return genes_to_keep

def PCA_top_features(df_X, df_y_encoded=None):
    """
    Parameters:
    -----------
        df_X: pandas DataFrame
            Data containing information to save

        df_y_encoded: np.array
            Data containing outcome. (Default: None)

    Function:
    ---------
        - Creates PCA model
        - Fits data to PCA
        - Transforms data
        
    Returns:
    --------
        genes_to_keep: list
            Genes from feature selection to keep.
    """
    # Regress out TFx
    print(f"df_X: \n{df_X}\n")

    for column in df_X.columns:
        X = df_X[column].values.reshape(-1, 1)
        residuals, _ = regression(X, df_y_encoded)
        df_X[column] = residuals.tolist()

    print(f"df_residuals: \n{df_X}\n")
    print(f"{type(df_X)}\n")

    # Selects amount of principle componenets
    pca = PCA(n_components=2)

    # Fit normalized data to PCA then applies the PCA transformation to the data
    pca.fit(df_X)
    pca_df = pca.transform(df_X)

    loadings = pd.DataFrame(pca.components_.T, columns=[1,2], index=df_X.columns)

    print(f"loadings: \n{loadings}\n")

    top_loadings = loadings[1].sort_values(ascending=False).head(math.ceil(len(loadings)*0.2))
    print(f"top_loadings: \n{top_loadings}\n")

    genes_to_keep = top_loadings.index.tolist()
    print(f"genes_to_keep: \n{genes_to_keep}\n")

    return genes_to_keep

def regression(x_data, y_data):
    """
    Parameters:
    -----------
        x_data : np.ndarray
            Tumor fraction values, shaped as (-1, 1).

        y_data : np.ndarray
            Central depth values, shaped as (-1, 1).

    Function:
    ---------
        - Fits a linear regression model to predict central depth from tumor fraction.
        - Calculates residuals as the difference between predicted and actual values.

    Returns:
    --------
        residuals : np.ndarray
            Residual values from the regression fit.
        model : sklearn.linear_model.LinearRegression
            Trained linear regression model.
    """

     # Linear Regression
    model = LinearRegression()

    # Fitting a line to the data
    model.fit(x_data, y_data)
    # Using that line to predict central depth from tumor fraction
    predicted = model.predict(x_data)
    # Calculate residuals
    residuals = predicted - y_data

    return residuals, model

def boruta_rf_feature_selection(df_X, df_y_encoded=None, random_seed=None):
    """
    Parameters:
    -----------
        df_X: pandas DataFrame
            Data containing information to save

        df_y_encoded: np.array
            Data containing outcome. (Default: None)

    Function:
    ---------
        - Creates a Random Forest Classifier 
        - Fits a Boruta feature selection method to the classifier 
        - Fits data to the selector
        - Filters data and returns column names
        
    Returns:
    --------
        genes_to_keep: list
            Genes from feature selection to keep.
    """
    print(f"df_X: {df_X}")
    print(f"df_y_encoded: {df_y_encoded}")
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=None, random_state=random_seed)

    # define Boruta feature selection method
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=random_seed, max_iter=200, alpha=0.1)

    # find all relevant features - 5 features should be selected
    feat_selector.fit(df_X, df_y_encoded)

    boruta_mask = feat_selector.support_ | feat_selector.support_weak_

    if boruta_mask.sum() == 0:
        print("WARNING: Boruta rejected all features. Returning all features.")
        genes_to_keep = df_X.columns
        return genes_to_keep

    df_X_filtered = df_X.loc[:, boruta_mask]
    genes_to_keep = df_X_filtered.columns
    
    return genes_to_keep

def smv_rfe(df_X, df_y_encoded=None, random_seed=42):
    """
    Parameters:
    -----------
        df_X: pandas DataFrame
            Data containing information to save

        df_y_encoded: np.array
            Data containing outcome. (Default: None)

    Function:
    ---------
        - Creates a SVR object
        - Fits a RFECV object to it
        - Fits data to the RFECV oject
        - Filters data and returns columns 
        
    Returns:
    --------
        genes_to_keep: list
            Genes from feature selection to keep.
    """
    estimator = SVR(kernel="linear")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    selector = RFECV(estimator, step=1, cv=cv)
    selector = selector.fit(df_X, df_y_encoded)
    df_X_filtered = df_X.loc[:, selector.support_]
    genes_to_keep = df_X_filtered.columns
    
    return genes_to_keep


def mrmr(df_X, df_y_encoded=None):
    """
    Parameters:
    -----------
        df_X: pandas DataFrame
            Data containing information to save

        df_y_encoded: np.array
            Data containing outcome. (Default: None)

    Function:
    ---------
        - Creates a mRMR classifier and fits data to it, taking half of the features, and returns it 
        
    Returns:
    --------
        genes_to_keep: list
            Genes from feature selection to keep.
    """
    genes_to_keep = mrmr_classif(X=df_X, y=df_y_encoded, K=math.ceil(len(df_X.columns)*0.5))
    return genes_to_keep


def elastic_net_feature_selection(df_X, df_y_encoded, cv=3, random_seed=42):
    """
    Parameters:
    -----------
        df_X: pandas DataFrame
            Data containing information to save

        df_y_encoded: np.array
            Data containing outcome. (Default: None)
        
        cv: int
            Number of cross validation loops to do.

    Function:
    ---------
        - Does elasticnet on dataframe.
        
    Returns:
    --------
        genes_to_keep: list
            Genes from feature selection to keep.
    """    
    logreg = LogisticRegressionCV(cv=cv, penalty='elasticnet', l1_ratios=[0.1, 0.5, 0.9], solver='saga', random_state=random_seed)
    logreg.fit(df_X, df_y_encoded)

    # Find feature names and coefficientes
    feature_names = df_X.columns  # assuming it's a pandas DataFrame
    coefficients = logreg.coef_[0]         # shape (1, n_features) for binary classification
    feature_importance_abs = dict(zip(feature_names, abs(coefficients))) # Captures absolute importatnce (magnitude, not direction)

    # Sort by featuer importance
    feature_importance_sorted = dict(
        sorted(feature_importance_abs.items(), key=lambda x: x[1], reverse=True)
    )

    feature_importance_sorted_df = pd.DataFrame.from_dict(feature_importance_sorted, orient="index")

    print(f"feature_importance_sorted_df: {feature_importance_sorted_df}")

    top_variable_features = feature_importance_sorted_df.head(math.ceil(len(feature_importance_sorted_df)*.2))
    genes_to_keep = top_variable_features.index.tolist()

    return genes_to_keep