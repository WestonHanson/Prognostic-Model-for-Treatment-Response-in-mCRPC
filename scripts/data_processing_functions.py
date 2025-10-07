# Author: Weston Hanson
# Place: Fred Hutch Cancer Center, Seattle, WA
# Date Created: 10/7/2025
# Purpose: To clean data for use in XGBoost

import re
import pandas as pd

def clean_patient_names(rna_seq_data, pluvicto_master_sheet, timepoint, cutoff):
    '''
    Parameters:
    -----------
        rna_seq_data: pandas dataframe
            A dataframe with rna-seq data.
        pluvicto_master_sheet: pandas dataframe
            A dataframe with metadata.
        timepoint: string
            Timepoint to remove.
        cutoff: string
            TFx cutoff in hundredths (typically "0.3" or "0.10").
        

    Function:
    ---------
        - Removes all patients that have specified timepoint.
        - Returns the dataframe with the rows that are are greater than or equal to the cutoff.

    Returns:
    --------
        rna_seq_data:
            The updated dataframe.
        pluvicto_master_sheet:
            The updated dataframe.
    '''

    # Remove patients below cutoff
    pluvicto_master_sheet = pluvicto_master_sheet[pluvicto_master_sheet['TFx_C1'] >= cutoff]

    # Remove patients based on timepoint
    column_names_to_remove = [col for col in rna_seq_data.columns if timepoint in col]
    
    if len(column_names_to_remove) > 0:
        rna_seq_data = rna_seq_data.drop(columns=column_names_to_remove)

    # Strip names of tags
    original_names = rna_seq_data.columns.tolist()
    cleaned_names = [re.sub(r'_[A-Z]_C[0-9]+$', '', name) for name in original_names]
    rna_seq_data.columns = cleaned_names

    # Subset to common patients for both dataframes
    clean_colnames = [col.strip() for col in rna_seq_data.columns]
    clean_rownames = [idx.strip() for idx in pluvicto_master_sheet.index]

    common_patients = list(set(clean_colnames) & set(clean_rownames))

    rna_seq_data = rna_seq_data[common_patients]
    pluvicto_master_sheet = pluvicto_master_sheet.loc[common_patients]

    pluvicto_master_sheet = pluvicto_master_sheet.loc[rna_seq_data.columns]

    return rna_seq_data, pluvicto_master_sheet

def combine_dataframes(rna_seq_data, pluvicto_master_sheet, columns_to_keep):
    
    # Transpose rna-seq data
    rna_seq_data_T = rna_seq_data.T

    # Convert from nparray to dataframe
    rna_seq_data_T = pd.DataFrame(rna_seq_data_T)

    # Subset pluvicto_master_sheet to specific columns
    pluvicto_master_sheet = pluvicto_master_sheet[columns_to_keep]

    print(pluvicto_master_sheet)

    combined_dfs = pd.merge(rna_seq_data_T, pluvicto_master_sheet, left_index = True, right_index = True, how='inner')

    return combined_dfs