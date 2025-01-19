from typing import List
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import numpy as np
import re
import io
import warnings
import streamlit as st


def combine_column_info(column_types, column_descriptions):
    """
    Combines two dictionaries into a nested dictionary format:
    {
        'column_name': {'type': ..., 'description': ...},
        ...
    }

    Parameters:
    -----------
    column_types : dict
        A dictionary mapping column names to their types.

    column_descriptions : dict
        A dictionary mapping column names to their descriptions.

    Returns:
    --------
    combined_info : dict
        A nested dictionary combining column types and descriptions.
    """
    combined_info = {
        col: {"type": column_types[col], "description": column_descriptions[col]}
        for col in column_types
    }
    return combined_info
    

def classify_columns(df, unique_threshold=0.9, cat_threshold=10, sparse_threshold=0.8):
    """
    classify_columns_fix(df, unique_threshold=0.9, cat_threshold=10, sparse_threshold=0.8)
    
    Classifies the columns of a pandas DataFrame into categories such as 'Unique Integer',
    'Continuous Float', 'Categorical String', 'Datetime', and more. It uses logical checks 
    on column data types, unique value counts, and other characteristics to determine the 
    most appropriate type for each column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the columns to classify.
    
    unique_threshold : float, optional, default=0.9
        The threshold for determining if a column contains mostly unique values.
        For example, a column where 90% of the values are unique will be classified 
        as 'Unique String' or 'Unique Integer'.
    
    cat_threshold : int, optional, default=10
        The threshold for determining categorical columns.
        Numeric or string columns with fewer unique values than this threshold 
        will be classified as 'Categorical Integer' or 'Categorical String'.
    
    sparse_threshold : float, optional, default=0.8
        The threshold for determining sparse columns.
        Columns with more than this proportion of missing values will be considered sparse.
    
    Returns:
    --------
    column_types : dict
        A dictionary mapping column names to their classified types. 
        Example:
        {
            'ID': 'Unique Integer',
            'gender': 'Categorical String',
            'age': 'Continuous Integer',
            'spending_score': 'Categorical String'
        }
    
    descriptions : dict
        A dictionary providing a human-readable description for each column, explaining 
        the classification. 
        Example:
        {
            'ID': "'ID' is a unique identifier (integer).",
            'gender': "'gender' has 2 unique string categories.",
            'age': "'age' contains integer values with a wide range."
        }
    
    Notes:
    ------
    - The function can handle numeric, string, boolean, and datetime-like columns.
    - It avoids incorrectly classifying numeric columns as datetime by sampling values 
      and using safe parsing with `pd.to_datetime()`.
    - Sparse columns with a high ratio of missing values are explicitly flagged.
    """

    
    column_types = {}
    descriptions = {}

    for col in df.columns:
        non_null_values = df[col].dropna()
        unique_count = non_null_values.nunique()
        total_count = len(df[col])
        unique_ratio = unique_count / total_count
        missing_ratio = df[col].isna().sum() / total_count

        # Empty Column
        if non_null_values.empty:
            column_types[col] = "Empty Column"
            descriptions[col] = f"'{col}' contains no data."
            continue

        # Numeric Columns
        if pd.api.types.is_numeric_dtype(non_null_values):
            if pd.api.types.is_integer_dtype(non_null_values):
                if unique_count == total_count:
                    column_types[col] = "Unique Integer"
                    descriptions[col] = f"'{col}' is a unique identifier (integer)."
                elif unique_count <= cat_threshold:
                    column_types[col] = "Categorical Integer"
                    descriptions[col] = f"'{col}' has {unique_count} unique integer categories. The unique values are: {sorted(map(int,list(df[col].dropna().unique())))}."
                else:
                    column_types[col] = "Continuous Integer"
                    descriptions[col] = f"'{col}' contains integer values with a wide range."
            elif pd.api.types.is_float_dtype(non_null_values):
                if unique_count <= cat_threshold:
                    column_types[col] = "Categorical Float"
                    descriptions[col] = f"'{col}' has {unique_count} unique float categories. The unique values are: {sorted(list(df[col].dropna().unique()))}."
                else:
                    column_types[col] = "Continuous Float"
                    descriptions[col] = f"'{col}' contains continuous float values."
            continue

        # Datetime Columns: Validate date-like columns
        is_datetime = False
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                sample = non_null_values.sample(min(100, len(non_null_values)))
                if pd.to_datetime(sample, errors='coerce').notna().all():
                    is_datetime = True
            except:
                pass
        
        if is_datetime:
            column_types[col] = "Datetime"
            descriptions[col] = f"'{col}' represents datetime values."
            continue

        # Boolean Columns
        if pd.api.types.is_bool_dtype(non_null_values):
            column_types[col] = "Boolean"
            descriptions[col] = f"'{col}' is a boolean column with True/False values."
            continue

        # String/Object Columns
        if pd.api.types.is_object_dtype(non_null_values) or pd.api.types.is_string_dtype(non_null_values):
            if unique_ratio >= unique_threshold:
                column_types[col] = "Unique String"
                descriptions[col] = f"'{col}' contains unique string values."
            elif unique_count <= cat_threshold:
                column_types[col] = "Categorical String"
                descriptions[col] = f"'{col}' has {unique_count} unique string categories. The unique values are: {sorted(list(df[col].dropna().unique()))}."
            else:
                column_types[col] = "High Cardinality String"
                descriptions[col] = f"'{col}' contains high-cardinality string values."
            continue

        # Default Fallback
        column_types[col] = "Other Type"
        descriptions[col] = f"'{col}' could not be categorized."

    final_output = combine_column_info(column_types, descriptions)

    return final_output

def column_frequency_analysis(df, column_types):
    """
    Computes the frequency count and percentage of total rows or unique values 
    for each column based on its classified type.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame.

    column_types : dict
        A dictionary mapping column names to their classified types, 
        as returned by the `classify_columns` function.

    Returns:
    --------
    frequency_info : dict
        A dictionary where each key is a column name and each value is 
        a DataFrame summarizing frequency counts and percentages.
    """
    frequency_info = {}

    for col, value in column_types.items():
        col_type = value['type']
        try:
            if col_type in ["Categorical String", "Categorical Integer", "Categorical Float", "Boolean"]:
                # Frequency count and percentage for categorical or boolean columns
                freq_df = (
                    df[col]
                    .value_counts(dropna=False)  # Count unique values
                    .reset_index()               # Convert to DataFrame
                    .rename(columns={"index": "Classes", col: "Count"})  # Rename columns
                )
                # Calculate percentages
                freq_df["Percentage"] = round((freq_df["Count"] / len(df)) * 100, 2).astype(str) + '%'
                frequency_info[col] = freq_df
    
            elif col_type in ["Unique Integer", "Unique String"]:
                # Unique value count for unique identifier columns
                unique_count = df[col].nunique()
                frequency_info[col] = pd.DataFrame({
                    "Metric": ["Unique Count", "Total Rows"],
                    "Value": [unique_count, len(df)],
                    "Percentage": [str(round(unique_count / len(df) * 100, 2)) + '%', '100%']
                })
    
            elif col_type in ["Continuous Integer", "Continuous Float"]:
                # Range and unique count for continuous columns
                min_val = df[col].min()
                max_val = df[col].max()
                frequency_info[col] = pd.DataFrame({
                    "Metric": ["Min Value", "Max Value", "Unique Count", "Total Rows"],
                    "Value": [min_val, max_val, df[col].nunique(), len(df)],
                    "Percentage": [None, None, str(round(df[col].nunique() / len(df) * 100, 2)) + '%', '100%']
                })
    
            elif col_type == "Datetime":
                # Frequency counts for datetime columns
                freq_df = (
                    df[col]
                    .dt.to_period("M")  # Convert to monthly periods
                    .value_counts()     # Count occurrences
                    .reset_index()      # Convert to DataFrame
                    .rename(columns={"index": "Period", col: "Count"})  # Rename columns
                    .sort_values("Period")  # Sort by Period
                )
                # Calculate percentages
                freq_df["Percentage"] = str(round((freq_df["Count"] / len(df)) * 100, 2)) + '%'
                frequency_info[col] = freq_df
    
            else:
                # Handle unclassified columns
                frequency_info[col] = pd.DataFrame({
                    "Metric": ["Unknown Type"],
                    "Value": ["No Frequency Analysis Available"],
                    "Percentage": [None]
                })

        except Exception as e:
            frequency_info[col] = f"Error processing column {col}: {e}"
            
    return frequency_info

def convert_numpy_types(obj):
    """
    Recursively converts NumPy types (np.int64, np.float64) to native Python types in a dictionary.

    Parameters:
    -----------
    obj : dict, list, or value
        The input dictionary or value to clean.

    Returns:
    --------
    obj : dict, list, or value
        The cleaned dictionary or value with all NumPy types converted to native types.
    """
    if isinstance(obj, dict):
        # If the object is a dictionary, clean each value recursively
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # If the object is a list, clean each element
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        # Convert NumPy numeric types to Python native types
        return obj.item()
    else:
        # Return the value as is for other types
        return obj


def generate_summary_statistics(df, column_info):
    """
    Generates summary statistics for each column based on its classified type.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame.

    column_info : dict
        A dictionary containing column types and descriptions.
        Expected format:
        {
            'column_name': {'type': 'type', 'description': 'description'}
        }

    Returns:
    --------
    summary_stats : dict
        A dictionary containing summary statistics for each column.
    """
    summary_stats = {}

    for col, info in column_info.items():
        col_type = info['type']
        non_null_values = df[col].dropna()
        missing_ratio = df[col].isna().sum() / len(df[col])

        if col_type in ["Continuous Integer", "Continuous Float"]:
            stats = {
                "Count": len(non_null_values),
                "Mean": round(non_null_values.mean(),4),
                "Median": round(non_null_values.median(),4),
                "Std Dev": round(non_null_values.std(),4),
                "Min": non_null_values.min(),
                "Max": non_null_values.max(),
                "Unique Values": non_null_values.nunique(),
                "Skenewss": df[col].skew(),
                "Kurtosis": df[col].kurt(),
                "Missing Values": df[col].isna().sum(),
                "Missing Percentage": f"{missing_ratio:.2%}",
            }

        elif col_type in ["Categorical Integer", "Categorical String", "High Cardinality String"]:
            stats = {
                "Count": len(non_null_values),
                "Unique Values": non_null_values.nunique(),
                "Top Value": non_null_values.mode().iloc[0] if not non_null_values.mode().empty else None,
                "Top Value Frequency": non_null_values.value_counts().iloc[0].astype(int) if not non_null_values.empty else 0,
                "Missing Values": df[col].isna().sum(),
                "Missing Percentage": f"{missing_ratio:.2%}",
            }

        elif col_type == "Datetime":
            stats = {
                "Count": len(non_null_values),
                "Min Date": non_null_values.min(),
                "Max Date": non_null_values.max(),
                "Unique Dates": non_null_values.nunique(),
                "Missing Values": df[col].isna().sum(),
                "Missing Percentage": f"{missing_ratio:.2%}",
            }

        elif col_type == "Boolean":
            stats = {
                "Count": len(non_null_values),
                "True Count": (non_null_values == True).sum(),
                "False Count": (non_null_values == False).sum(),
                "Missing Values": df[col].isna().sum(),
                "Missing Percentage": f"{missing_ratio:.2%}",
            }

        elif col_type == "Sparse String" or col_type == "Sparse Column":
            stats = {
                "Count": len(non_null_values),
                "Missing Values": df[col].isna().sum(),
                "Missing Percentage": f"{missing_ratio:.2%}",
            }

        else:  # For other or mixed types
            stats = {
                "Count": len(non_null_values),
                "Unique Values": non_null_values.nunique(),
                "Missing Values": df[col].isna().sum(),
                "Missing Percentage": f"{missing_ratio:.2%}",
            }

        summary_stats[col] = stats

    corrected_stats = convert_numpy_types(obj = summary_stats)

    return corrected_stats

def create_summary_dataframe(summary_stats):
    """
    Converts the summary statistics dictionary into a pandas DataFrame.

    Parameters:
    -----------
    summary_stats : dict
        A dictionary containing summary statistics for each column.

    Returns:
    --------
    summary_df : pd.DataFrame
        A DataFrame representation of the summary statistics.
    """
    # Initialize an empty list to collect rows
    summary_data = []

    # Iterate through the summary stats dictionary
    for column, stats in summary_stats.items():
        # Flatten each column's stats into a row
        row = {"Column": column}
        row.update(stats)
        summary_data.append(row)

    # Convert the list of rows into a DataFrame
    summary_df = pd.DataFrame(summary_data)

    return summary_df

def calculate_combined_correlation(df, column_type, method="pearson", word_threshold=3):
    """
    Computes the correlation matrix for numerical and encoded categorical columns in a DataFrame.
    Filters out textual columns where the values exceed a specified word count threshold.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing numerical and categorical columns.

    column_type: dict
        The input dictionary containing column type and description
        

    method : str, optional, default="pearson"
        The method to compute correlation. Options are:
        - "pearson": Standard correlation coefficient.
        - "kendall": Kendall Tau correlation coefficient.
        - "spearman": Spearman rank correlation.

    word_threshold : int, optional, default=4
        The maximum allowed word count for values in a textual column. Columns with longer
        values will be excluded from the analysis.

    Returns:
    --------
    correlation_matrix : pandas.DataFrame
        A DataFrame containing the correlation matrix for numerical and encoded categorical columns.
    """
    # Identify numerical and textual columns
    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    # Start with numerical columns
    processed_df = df[numeric_cols].copy()

    # Filter categorical columns based on word threshold
    for col in categorical_cols:
        col_type = column_type[col]['type']
        if df[col].nunique() > 1:  # Ignore single-value columns
            # Check if values have <= word_threshold words
            max_word_count = df[col].dropna().apply(lambda x: len(str(x).split())).max()
            if max_word_count <= word_threshold:
                # Encode valid categorical columns using label encoding
                processed_df[col] = df[col].astype("category").cat.codes

    if processed_df.empty:
        raise ValueError("No valid numerical or encodable categorical columns found.")

    excluded_types = ["Unique Integer", "Unique String"]
    excluded_columns = [col for col, col_type in column_type.items() if col_type['type'] in excluded_types]
    
    # Filter the DataFrame to exclude these columns
    filtered_columns = [col for col in processed_df.columns if col not in excluded_columns]
    filtered_df = processed_df[filtered_columns]

    # Compute the correlation matrix
    correlation_matrix = filtered_df.corr(method=method)

    return correlation_matrix

def get_all_summary_stats(df):
    # Get column type description
    column_type_desc_output = classify_columns(df)

    # Get a dictionary of frequency tables for all the columns
    freq_dict = column_frequency_analysis(df = df, column_types = column_type_desc_output)

    # Get a dictionary of summary stats for all the columns
    summary_stat_dict = generate_summary_statistics(df = df, column_info = column_type_desc_output)
    # Convert the dictionary into a datafrane
    summary_stat_df = create_summary_dataframe(summary_stat_dict)

    # Create a correlation matrix using appropriate columns
    corr_mat = calculate_combined_correlation(df, column_type_desc_output)
    
    # Create a heatmap plot
    corr_plt, ax = plt.subplots(figsize=(8, 6))  # Create a specific figure
    sns.heatmap(
        corr_mat, 
        annot=True,         # Show the values on the heatmap
        fmt=".2f",          # Format annotation to 2 decimal places
        cmap="coolwarm",    # Choose a color map
        cbar=True,          # Show the color bar
        ax=ax               # Specify the axes for this plot
    )

    corr_plt.savefig("exports/report_figures/heatmap_plot.png", dpi=300, bbox_inches="tight")
    return column_type_desc_output, freq_dict, summary_stat_dict, summary_stat_df, corr_mat
