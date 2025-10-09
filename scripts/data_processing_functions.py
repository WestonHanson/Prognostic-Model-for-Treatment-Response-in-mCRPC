# Author: Weston Hanson
# Place: Fred Hutch Cancer Center, Seattle, WA
# Date Created: 10/7/2025
# Purpose: To clean data for use in XGBoost

import re
import pandas as pd
import statistics
import numpy as np
import sys

def process_tfbs_data(tfbs_data):
    """
    Parameters:
    -----------
        tfbs_data_file: pandas DataFrame
            Data containing tfbs data.

    Function:
    ---------
        - Remove the first 6 columns of tfbs_data, transposes it, then turns it back into a dataframe.
        - Return tfbs_data.
        
    Returns:
    --------
        tfbs_data:
            tfbs_data with the first 6 columns removed.
    """
    tfbs_data_filtered = tfbs_data.iloc[:, 6:]
    tfbs_data_T = tfbs_data_filtered.T
    tfbs_data = pd.DataFrame(tfbs_data_T)
    
    return tfbs_data

def standard_scaling(df, responder_group, axis=0):
    """
    Parameters:
    -----------
        df: pandas DataFrame
            Data containing information to normalize
        responder_group: string
            A string representing the column name for responder/non-responder group.
        axis: int
            default: 0 (columns)
            Specifies what axis to normalize at: 0 (columns) or 1 (rows)

    Function:
    ---------
        - If there is a responder_group column it strips and saves it.
        - Then zscores the data frame.
        
    Returns:
    --------
        normalized_df:
            A normalized dataframe.
    """
    # If patient_id strip and save column
    if responder_group in df.columns:
        responder_group_df = df[responder_group]
        df_temp = df.drop(columns=[responder_group]).copy()
    else:
        responder_group_df = None
        df_temp = df.copy()

    normalized_df = df_temp.copy()

    # Removes rows that have nulls
    # normalized_df, df_temp = normalized_df.dropna(how='any', axis=0), df_temp.dropna(how='any', axis=0)
    
    # Normalize across columns
    if axis == 0:
        for row in df_temp.index:
            std = statistics.pstdev(df_temp.loc[row])
            mean = df_temp.loc[row].mean()
            normalized_df.loc[row] = (df_temp.loc[row] - mean) / std
    # Normalize across rows
    elif axis == 1:
        for column in df_temp.columns:
            std = statistics.pstdev(df_temp[column])
            mean = df_temp[column].mean()
            normalized_df[column] = (df_temp[column] - mean) / std

    if not responder_group_df is None:
        normalized_df = pd.merge(normalized_df, responder_group_df, left_index = True, right_index = True, how='inner')

    return normalized_df

def clean_patient_names(genomic_data, pluvicto_master_sheet, timepoint, cutoff):
    '''
    Parameters:
    -----------
        genomic_data: pandas dataframe
            A dataframe with genomic data.
        pluvicto_master_sheet: pandas dataframe
            A dataframe with metadata.
        timepoint: string
            Timepoint to remove.
        cutoff: string
            TFx cutoff in hundredths (typically "0.3" or "0.10").
        

    Function:
    ---------
        
        - Returns the dataframe with the rows that are are greater than or equal to the cutoff.
        - Removes all patients that have specified timepoint.
        - Strips patient tags.
        - Subset both dataframes to common patients.
        - Return both dataframes.

    Returns:
    --------
        genomic_data:
            The updated dataframe.
        pluvicto_master_sheet:
            The updated dataframe.
    '''
    # Remove patients below cutoff
    pluvicto_master_sheet = pluvicto_master_sheet[pluvicto_master_sheet['TFx_C1'] >= cutoff]
    
    # Remove any patients that have TFx NA
    pluvicto_master_sheet = pluvicto_master_sheet.dropna(subset=['TFx_C1', 'LOH.Score_C1', 'TMB_C1'])

    # Remove patients based on timepoint
    column_names_to_remove = [col for col in genomic_data.columns if timepoint in col]
    
    if len(column_names_to_remove) > 0:
        genomic_data = genomic_data.drop(columns=column_names_to_remove)

    # Strip names of tags
    original_names = genomic_data.columns.tolist()
    cleaned_names = [re.sub(r'_[A-Z]_C[0-9]+$', '', name) for name in original_names]
    genomic_data.columns = cleaned_names

    # Subset to common patients for both dataframes
    clean_colnames = [col.strip() for col in genomic_data.columns]
    clean_rownames = [idx.strip() for idx in pluvicto_master_sheet.index]

    common_patients = list(set(clean_colnames) & set(clean_rownames))

    genomic_data = genomic_data[common_patients]
    pluvicto_master_sheet = pluvicto_master_sheet.loc[common_patients]

    pluvicto_master_sheet = pluvicto_master_sheet.loc[genomic_data.columns]

    return genomic_data, pluvicto_master_sheet

def extract_FGA(FGA_data, timepoint):
    '''
    Parameters:
    -----------
        fga_data: pandas dataframe
            A dataframe with fraction genome altered data.
        timepoint: string
            Timepoint to select for.
        
    Function:
    ---------
        - 

    Returns:
    --------
        FGA_column_filtered:
            A dataframe column with just the FGA of the timepoint specified. 
    '''
    FGA_data_filtered = FGA_data[FGA_data['time_point'].str.contains(timepoint)]
    FGA_column_filtered = FGA_data_filtered["genomic_instability"]
    return FGA_column_filtered

def combine_dataframes(genomic_data, pluvicto_master_sheet, FGA_data, columns_to_keep):
    '''
    Parameters:
    -----------
        genomic_data: pandas dataframe
            A dataframe with genomic data.
        pluvicto_master_sheet: pandas dataframe
            A dataframe with metadata.
        columns_to_keep: List
            A list of columns in pluvicto_master_sheet to keep.
        

    Function:
    ---------
        - Transpose genomic_data so it is patient_id x gene and turn it back into a dataframe.
        - Then subset pluvicto_master_sheet to only the columns you want to keep.
        - Merge dataframes into one on same patient_ids and return it.

    Returns:
    --------
        combined_dfs:
            A combined dataframe of genomic_data and pluvicto_master_sheet.
    '''
    # Transpose genomic data
    genomic_data_T = genomic_data.T

    # Convert from nparray to dataframe
    genomic_data_T = pd.DataFrame(genomic_data_T)

    # Subset pluvicto_master_sheet to specific columns
    pluvicto_master_sheet = pluvicto_master_sheet[columns_to_keep]

    # Combine dataframes
    combined_dfs = pd.merge(genomic_data_T, pluvicto_master_sheet, left_index = True, right_index = True, how='inner')

    # Extract FGA as a dataframe column
    FGA_column_filtered = extract_FGA(FGA_data, "C1")

    # Combine FGA
    combined_dfs = pd.merge(combined_dfs, FGA_column_filtered, left_index = True, right_index = True, how='inner')

    return combined_dfs

def prepare_categorical_features(df):
    """
    Ensure categorical features are properly set in both train and test sets.
    
    Parameters:
    -----------
        df: pandas DataFrame
            Training features
        
    Returns:
    --------
        df: pandas DataFrames
            DataFrames with categorical features properly set
    """
    cats = df.select_dtypes(exclude='number').columns
    
    for col in cats:
        df[col] = df[col].astype('category')
    
    return df

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
    
    # Get median for numeric parameters only
    median_params = param_df.select_dtypes(include=[np.number]).median().to_dict()
    
    # Convert specific parameters to integers
    int_cols = ["max_depth", "max_delta_step"]
    for c in int_cols:
        if c in median_params:
            median_params[c] = int(round(median_params[c]))
    
    # Add non-numeric parameters from first fold
    for key in param_df.columns:
        if key not in median_params:
            median_params[key] = param_df[key].iloc[0]
    
    best_boost_round = int(np.median([r['best_round'] for r in cv_results]))
    
    return median_params, best_boost_round