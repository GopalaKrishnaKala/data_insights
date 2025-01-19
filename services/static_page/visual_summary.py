from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import openai
import os
import streamlit as st

from services.static_page.summary_stats import calculate_combined_correlation

openai.api_key = os.getenv('OPENAI_API_KEY')

def get_column_type_desc(all_col_type_desc, columns):
    return {col: all_col_type_desc[col] for col in columns if col in all_col_type_desc}

def get_column_category(column_type_desc):
    type_categories = {
    'Categorical': ['Categorical String', 'Categorical Float', 'Categorical Integer', 'Boolean'],
    'Continuous': ['Continuous Integer', 'Continuous Float'],
    'Datetime': ['Datetime']
    }
    # Create a mapping of type to category
    type_to_category = {typ: cat for cat, types_list in type_categories.items() for typ in types_list}
    
    # Map each column type to a category
    col_types = {key: value['type'] for key, value in column_type_desc.items()}
    column_categories = [type_to_category.get(t, 'Unknown') for t in col_types.values()]
    
    # Check for combinations of categories
    unique_categories = set(column_categories)
    if unique_categories == {'Categorical'}:
        return "Categorical"
    elif unique_categories == {'Continuous'}:
        return "Continuous"
    elif unique_categories == {'Datetime'}:
        return "Datetime"
    elif 'Continuous' in unique_categories and 'Categorical' in unique_categories:
        return "Categorical and Continuous"
    elif 'Datetime' in unique_categories and 'Continuous' in unique_categories:
        return "Datetime and Continuous"
    else:
        return "Unknown"


def process_categorical_data(df):
    categorical_cols = df.select_dtypes(include='object').columns
    
    # Track analyzed relationships
    analyzed_pairs = set()
    categorical_col_data = []
    
    # Generate summary counts for combinations of categorical columns
    for col1 in categorical_cols:
        col_summary = {'column': col1, 'value_counts': df[col1].value_counts().to_dict()}
        relationships = []
        
        for col2 in categorical_cols:
            if col1 != col2:
                # Use a tuple to check if the relationship has already been analyzed
                pair = tuple(sorted((col1, col2)))
                if pair not in analyzed_pairs:
                    analyzed_pairs.add(pair)  # Mark as analyzed
                    crosstab_counts = pd.crosstab(df[col1], df[col2]).to_dict()
                    crosstab_percentages = pd.crosstab(df[col1], df[col2], normalize='index') 
                    crosstab_percentages = (crosstab_percentages.apply(lambda x: round(x * 100, 2))).to_dict()
                    relationships.append({'related_column': col2, 'relationship': {'counts': crosstab_counts, 'percentages': crosstab_percentages}})
        
        col_summary['relationships'] = relationships
        categorical_col_data.append(col_summary)
    
    return categorical_col_data


def process_continuous_data(columns_to_extract, summary_stats):
    return {col: summary_stats[col] for col in columns_to_extract if col in summary_stats}

def process_cat_cont_data(df, column_type_desc):
    categorical_columns = [col for col, col_info in column_type_desc.items() if 'Categorical' in col_info['type']]
    continuous_columns = [col for col, col_info in column_type_desc.items() if 'Continuous' in col_info['type']]
    
    cat_cont_data = {}
    
    for cat_col in categorical_columns:
        cat_cont_data[cat_col] = {}
        grouped = df.groupby(cat_col)
        total_rows = len(df)
        
        for category, group in grouped:
            cat_cont_data[cat_col][category] = {}

            category_percentage = len(group) / total_rows * 100
            
            for cont_col in continuous_columns:
                non_null_values = group[cont_col].dropna()
                missing_ratio = group[cont_col].isna().mean()
                
                cat_cont_data[cat_col][category][cont_col] = {
                    "Count": len(non_null_values),
                    "Mean": round(non_null_values.mean(), 4) if not non_null_values.empty else None,
                    "Median": round(non_null_values.median(), 4) if not non_null_values.empty else None,
                    "Std Dev": round(non_null_values.std(), 4) if not non_null_values.empty else None,
                    "Min": non_null_values.min() if not non_null_values.empty else None,
                    "Max": non_null_values.max() if not non_null_values.empty else None,
                    "Unique Values": non_null_values.nunique() if not non_null_values.empty else 0,
                    "Skewness": round(non_null_values.skew(), 4) if not non_null_values.empty else None,
                    "Kurtosis": round(non_null_values.kurt(), 4) if not non_null_values.empty else None,
                    "Missing Values": group[cont_col].isna().sum(),
                    "Missing Percentage": f"{missing_ratio:.2%}",
                    "Category Percentage": round(category_percentage, 2)
                }
    
    return cat_cont_data

def data_summary(df, column_cat, cols, summaries_output, column_type_desc):
    data_summary = None
    if column_cat == "Categorical":
        data_summary = process_categorical_data(df[cols])
    elif column_cat == "Continuous":
        data_summary = process_continuous_data(cols, summaries_output)
        data_summary['correlation'] = calculate_combined_correlation(df[cols], column_type_desc)
    elif column_cat == "Categorical and Continuous":
        data_summary = process_cat_cont_data(df[cols], column_type_desc)
    elif column_cat == "Datetime and Continuous":
        data_summary = process_continuous_data(cols)
    else:
        data_summary
    return data_summary


def get_visual_summary(df, chart_metadata, all_col_type_desc, summary_stats):
    
    llm = ChatOpenAI(
        model="gpt-4-turbo",
        openai_api_key=st.session_state.api_key,
        temperature=0,
        streaming=True
    )
    
    
    prompt_template = """
    # ROLE
    You are an AI assistant that provides insights based on datasets or statistical summaries and chart metadata.
    
    # TASK
    Use the provided input (which may be a full dataset or a statistical summary) and chart metadata to generate a brief, comprehensive summary of the key
    insights. Assume the input represents the data visualized as described in the metadata. Your description should focus on the trends, comparisons 
    and relationships observable in the chart. Structure the response as if explaining the key takeaways of the chart, including references to the x-axis,
    y-axis and notable patterns or outliers, without directly referencing the raw dataset. Ensure the explanation feels natural and engaging.
    
    # INPUT        
    Data: {data} – The input data, which can either be a full dataset with detailed entries, a statistical summary with aggregate metrics
    (e.g., mean, median, max, min, percentiles) or categorical statistics.
    Chart Metadata: {chart_metadata} - A structured description of the chart, dynamically adaptable for varying numbers of columns and roles.
    
    # CHART_METADATA DESCRIPTION
    The `chart_metadata` object provides a dynamic description of the chart. It includes:
    - `type`: The type of chart, such as "Line Chart" or "Bar Chart".
    - `variables`: A list of column names from the dataset used in the chart.
    - For each column in `variables`, an entry specifies its role, such as "x-axis", "y-axis", or other roles relevant to the chart type.
    This structure is adaptable to any dataset and chart configuration.
    
    # INSTRUCTIONS
    If the Data is a full dataset, analyze trends, relationships and patterns between variables as visualized in the chart.
    If the Data is a statistical summary, extrapolate trends and distributions as if the full dataset were visualized in the chart.
    Describe the insights as though you're explaining a visual chart to someone, anchoring your explanation to axis labels and chart type.
    Do not provide insights other than what is visualized in the chart. (for ex. don't write insights derived from data that might not be visualised)
    Provide description in a paragraph.
    Avoid explicitly mentioning whether the input is raw data or a summary. Ensure the explanation feels complete and engaging.
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["data", "chart_metadata"],
    )
    
    chain = prompt | llm

    # Collect variables of interest and their type/description
    cols = chart_metadata.get("variables", [])
    chart_columns = [col for col in cols if col in df.columns]
    
    col_type_desc = get_column_type_desc(all_col_type_desc, chart_columns)
    # print(col_type_desc)

    # Get a high level type of column such as categorical, continuous, etc...
    column_cat = get_column_category(col_type_desc)

    # Get the appropriate summary stats based on the type of column
    data_summ = data_summary(df, column_cat, chart_columns, summary_stats, col_type_desc)
    # print(data_summ)

    # Invoke the LLM to return the summary of the visual
    result = chain.invoke({"data": data_summ, "chart_metadata": chart_metadata})
    # print(result_1.content)

    visual_summary = result.content

    return visual_summary
