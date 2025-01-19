from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import List
import openai
import os
import json
import re
import warnings
import streamlit as st

openai.api_key = os.getenv('OPENAI_API_KEY')

def get_insight(df, col_type_desc, freq_dict, summary_stats, corr_mat):
    prompt_template = """
    # CONTEXT
    You are a Data Insights Expert GPT specializing in data interpretation and predictive modeling. Your role is to analyze datasets and generate comprehensive, detailed insights. You excel at identifying patterns, trends, and relationships while suggesting practical ways to enhance data understanding or apply it in predictive modeling.
    
    # GOAL
    Generate a detailed report based on the following data inputs:
    - Data shape - {df_shape}
    - Column types and descriptions - {col_type_desc}
    - Frequency counts - {freq_dict}
    - Summary statistics - {summary_stats}
    - Correlation matrix - {corr_mat}
    - Head of the data - {df_head}
    
    The report should be extremely detailed, and it must include:
    - In-depth observations about the data's structure, patterns, and anomalies.
    - A detailed exploration of relationships between variables based on the correlation matrix.
    - Observations of out-of-ordinary or nonsensical data points that could indicate errors, anomalies, or the need for further cleaning.
    - Recommendations for potential predictive (dependent) variables and independent variables for building predictive models.
    - Suggestions for further exploratory data analysis to improve understanding of the data.
    
    # REPORT STRUCTURE
    Data Overview: Start with a concise summary of the data structure using column types and descriptions. Highlight key categories, numerical ranges, or unique attributes.
    Frequency Analysis: Discuss notable frequency counts, including categories or ranges with unexpectedly high or low occurrences. Mention potential biases or patterns that may need further investigation.
    Statistical Summary: Analyze summary statistics such as mean, median, standard deviation, and outliers for numerical columns. Discuss what the statistics reveal about the data's central tendency, variability, and skewness.
    Correlation Analysis: Interpret the correlation matrix, identifying strong positive or negative relationships between variables. Suggest why these relationships exist and their potential implications.
    Unusual Data Observations: Identify and describe any data points or patterns that do not make sense or fall outside the expected range for specific columns (e.g., negative ages, extreme values like 0 for income or 200 for age, or inconsistent categorical values). Explain the potential implications of these anomalies and recommend approaches to address or further investigate them.
    Predictive Modeling Suggestions: Identify one or more promising dependent variables for predictive modeling, providing reasons based on data insights. Recommend independent variables with strong predictive potential based on their relationships and relevance to the dependent variables. Recommended predictive modeling techniques that will work the best.
    Further Exploration Recommendations: Propose additional exploratory analysis steps, such as feature engineering, handling missing values, or visualizations to uncover deeper insights. Suggest techniques like clustering, PCA, or other dimensionality reduction methods if the dataset is large or complex.
    
    # OUTPUT FORMAT:
    **Data Overview**: ...
    **Frequency Analysis**: ...
    **Statistical Summary**: ...
    **Correlation Analysis**: ...
    **Unusual Data Observations**: ...
    **Predictive Modeling Suggestions**: ...
    **Further Exploration Recommendations**: ...

    # OUTPUT INSTRUCTIONS
    Make sure to output every bullet point from the instructions with detailed information.
    Do not hesitate to include as much information as possible to give a 360 degree view of the data.
    Do not output any pre-sentences like "Here's a overview" instead directly jump in to the output
    Ensure the report is structured logically with headings and subheadings for each section. Use professional language to maintain clarity and precision.
    """

    llm = ChatOpenAI(
        model="gpt-4-turbo",#"gpt-3.5-turbo-16k-0613",
        openai_api_key=st.session_state.api_key,
        temperature=0.4,
        streaming=True
    )
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["df_shape", "col_type_desc", "freq_dict", "summary_stats", "corr_mat", "df_head"],
    )

    chain = prompt | llm 
    
    response = chain.invoke(
        {
            "df_shape": df.shape,
            "col_type_desc": col_type_desc,
            "freq_dict": freq_dict,
            "summary_stats": summary_stats,
            "corr_mat": corr_mat,
            "df_head": df.head(20)
        }
    )
        
    return response.content

def extract_insights_sections(text):
    """
    Extracts structured insights from a text string containing data overview, frequency analysis, 
    statistical summary, correlation analysis, predictive modeling suggestions, and further exploration recommendations.

    Args:
        text (str): The input text string containing various sections of analysis.

    Returns:
        dict: A dictionary containing structured insights from the text.
    """
    sections = [
        "Data Overview", 
        "Frequency Analysis", 
        "Statistical Summary",
        "Correlation Analysis", 
        "Unusual Data Observations",
        "Predictive Modeling Suggestions", 
        "Further Exploration Recommendations"
    ]
    
    insights = {}
    
    # Adjust regex pattern for robust matching
    for section in sections:
        pattern = rf"\*\*{section}\*\*:(.*?)(?=\n\*\*|\Z)"
        match = re.search(pattern, text, re.S)
        if match:
            content = match.group(1).strip()
            insights[section] = content if content else "Not found"
        else:
            insights[section] = "Not found"
    
    return insights

def generate_insight_llm(df, col_type_desc, freq_dict, summary_stats, corr_mat):
    insight = get_insight(df = df, col_type_desc = col_type_desc, freq_dict = freq_dict, summary_stats = summary_stats, corr_mat = corr_mat)
    insights_dict = extract_insights_sections(insight)

    return insights_dict
