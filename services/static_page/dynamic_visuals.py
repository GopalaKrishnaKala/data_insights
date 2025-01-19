import pandas as pd
import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import openai
import os
from llama_index.core.readers.json import JSONReader
from llama_index.core import VectorStoreIndex
import json
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools import  ToolMetadata
from llama_index.llms.openai import OpenAI
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
import streamlit as st
from services.static_page.visual_summary import get_visual_summary

#openai.api_key = os.getenv('OPENAI_API_KEY')

def initialize_chat_openai(api_key):
    return ChatOpenAI(
        temperature=0.7,
        model_name="gpt-4",
        openai_api_key=api_key
    )

def convert_dict(df, column_type_desc):
    output_dict = {}

    for column, details in column_type_desc.items():
        # Extract type and description
        column_type = details['type']
        description = details['description']

        # Determine the type field for dict2
        if 'Unique' in column_type:
            type_field = column_type
            vals = "Unique"
        elif "Categorical" in column_type:
            type_field = "categorical"
            # Extract unique values from the description
            vals = list(df[column].unique())
        else:
            type_field = "continuous"
            # For continuous columns, set 'vals' as 'integers' or 'float' based on column type
            if "Integer" in column_type:
                vals = "integers"
            elif 'Float' in column_type:
                vals = "floats"
            else:
                vals = "strings"

        # Add the entry to dict2
        output_dict[column] = {
            "column_name": column,
            "type": type_field,
            "vals": vals,
            "desc": description
        }

    return output_dict

def make_json_serializable(input_dict):
    def convert_vals(value):
        if isinstance(value, list):
            return [int(v) if isinstance(v, np.int64) else v for v in value]
        return value

    for key, value in input_dict.items():
        if "vals" in value:
            value["vals"] = convert_vals(value["vals"])
    return input_dict

def get_col_desc(df):    
    prompt_template = """
    # CONTEXT
    You are Dataset Description GPT, a world-class expert in data analysis and dataset comprehension. Your role is to analyze the first 100 rows of a given dataset and generate clear, concise descriptions for each column in the dataset.
    
    # GOAL
    Describe each column in the dataset in plain language, ensuring that the descriptions are intuitive for a non-technical audience. Use a dictionary format with column names as keys and their descriptions as values.
    
    # TASK INSTRUCTIONS
    1. Analyze the first 100 rows of the dataset provided.
    2. Identify the role and meaning of each column.
    3. Create descriptions that explain the purpose or type of data in each column in simple terms.

    # INPUT
    Data: {data}
    
    # EXAMPLES OF DESCRIPTIONS
    'salary': 'Represents the annual income of an individual in USD.'
    'location': 'Indicates the geographical region or city of the record.'
    'customer_id': 'A unique identifier assigned to each customer.'
    
    # OUTPUT FORMAT
    Return the output in a Python dictionary format:
    
    {{
        'column1': 'Description of column1',
        'column2': 'Description of column2',
        ...
    }}
    
    # ADDITIONAL REQUIREMENTS:
    1. Be concise but clear in the descriptions.
    2. If a column has numerical data, indicate its measurement unit if evident (e.g., "Age in years").
    3. Avoid assumptions beyond the provided data. Base descriptions only on observable patterns in the data.
    """
    
    llm = ChatOpenAI(
        model="gpt-4-turbo",#"gpt-3.5-turbo-16k-0613",
        openai_api_key=st.session_state.api_key,
        temperature=0,
        streaming=True
    )
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["data"],
    )

    chain = prompt | llm 
    
    response = chain.invoke(
        {
            "data": df.head(100)
        }
    )

    desc_dict = eval(response.content)
        
    return desc_dict

def update_description(col_type_desc, llm_description):
    for col in col_type_desc.keys():
        llm_desc = llm_description.get(col)
        col_type_desc[col]['desc'] = llm_desc
        
    return col_type_desc


def user_query_generation_pmt(col_desc):
    prompt=f"""
    #ROLE
    You are an expert in generating instructions for an agent to create Plotly visualizations. All your instructions should be precise and clear.
    #INPUT
    Your input is a column dictionary {col_desc}. Take this into account while creating your instructions. Use exact column names and mention the exact
    x and y coordinates. If either of the coordinates need to be calculated, specify that. 
    #INSTRUCTIONS
    - Make sure you mention the exact column names for x and y coordinates.
    - Make sure you start ALL your instructions with "Give Plotly code".
    - Make sure you mention the type of plot.
    - Make sure you create simple, medium and complex plots with a range of columns.
    - Create 3 plot instructions for each difficulty level.
    #OUTPUT
    You will output instructions in natural language.
    ###Simple Plots:
    1. <Instruction 1>
    2. <Instruction 2>
    3. <Instruction 3>
     ###Medium Plots:
    1. <Instruction 1>
    2. <Instruction 2>
    3. <Instruction 3>
     ###Complex Plots:
    1. <Instruction 1>
    2. <Instruction 2>
    3. <Instruction 3>
    """
    return prompt
    
#Parsing instructuctions form LLM output
def parse_plot_instructions(text):
    instructions = []
    
    # Split text into lines and process
    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and section headers
        if not line or line.startswith('###'):
            continue
            
        # Process numbered instructions
        if line[0].isdigit():
            # Remove the number and period at the start
            instruction = line.split('.', 1)[1].strip()
            instructions.append(instruction)
    
    return instructions
    
def agent_prompts(df_info):
    llm = ChatOpenAI(
        model="gpt-4-turbo",
        openai_api_key=st.session_state.api_key,
        temperature=0.7
    )
    pmt=user_query_generation_pmt(df_info)
    resp=llm.invoke(pmt)
    try:
        prompts = parse_plot_instructions(resp.content)
    except:
        print("Prompt generation for agent failed!")
    return prompts


def df_index_creation(path):
    reader = JSONReader()
    documents = reader.load_data(input_file=path)
    dataframe_index =  VectorStoreIndex.from_documents(documents)
    return dataframe_index

def styling_index_creation():
    styling_instructions =[Document(text="""
  Dont ignore any of these instructions.
        For a line chart always use plotly_white template, use colors that stand out, reduce x axes & y axes line to 0.2 & x & y grid width to 1. 
        Always give a title and make bold using html tag axis label and try to use multiple colors if more than one line
        Annotate the min and max of the line
        Display numbers in thousand(K) or Million(M) if larger than 1000/100000 
        Show percentages in 2 decimal points with '%' sign
        """
        ),
        Document(text="""
  Dont ignore any of these instructions.
        For a histogram always use plotly_white template,use colors that stand out, reduce x axes & y axes line to 0.2 & x & y grid width to 1. 
        Always give a title and make bold using html tag axis label and try to use multiple colors if more than one line
        Annotate the min and max of the line
        Display numbers in thousand(K) or Million(M) if larger than 1000/100000 
        Show percentages in 2 decimal points with '%' sign
        """
        ),   
        Document(text="""
  Dont ignore any of these instructions.
        For a scatter plot always use plotly_white template,use colors that stand out, reduce x axes & y axes line to 0.2 & x & y grid width to 1. 
        Always give a title and make bold using html tag axis label and try to use multiple colors if more than one line
        Annotate the min and max of the line
        Display numbers in thousand(K) or Million(M) if larger than 1000/100000 
        Show percentages in 2 decimal points with '%' sign
        """
        ),
        Document(text="""
  Dont ignore any of these instructions.
        For a box plot always use plotly_white template,use colors that stand out, reduce x axes & y axes line to 0.2 & x & y grid width to 1. 
        Always give a title and make bold using html tag axis label and try to use multiple colors if more than one line
        Annotate the min and max of the line
        Display numbers in thousand(K) or Million(M) if larger than 1000/100000 
        Show percentages in 2 decimal points with '%' sign
        """
        )
        , Document(text="""
        Dont ignore any of these instructions.
        For a bar chart always use plotly_white template,use colors that stand out, reduce x axes & y axes line to 0.2 & x & y grid width to 1. 
        Always give a title and make bold using html tag axis label and try to use multiple colors if more than one line
        Always display numbers in thousand(K) or Million(M) if larger than 1000/100000. Add annotations x values
        Annotate the values on the y variable
        If variable is a percentage show in 2 decimal points with '%' sign.
        """)
       , Document(text=
          """ General chart instructions
        Do not ignore any of these instructions
         always use plotly_white template,use colors that stand out, reduce x & y axes line to 0.2 & x & y grid width to 1. 
        Always give a title and make bold using html tag axis label 
        Always display numbers in thousand(K) or Million(M) if larger than 1000/100000. Add annotations x values
        If variable is a percentage show in 2 decimal points with '%'""")
         ]
    # Creating an Index
    style_index =  VectorStoreIndex.from_documents(styling_instructions)
    return style_index

def agent_modified_pmt():
    template= """You are designed to help with building data visualizations in Plotly. You may do all sorts of analyses and actions using Python
    
    ## Tools
    
    You have access to a wide variety of tools. You are responsible for using the tools in any sequence you deem appropriate to complete the task at hand.
    This may require breaking the task into subtasks and using different tools to complete each subtask.
    
    You have access to the following tools {tools}, use these tools to find information about the data and styling:
    {tool_desc}


    ## Output Format
    
    Please answer in the same language as the question and use the following format:
    
    ```
    Thought: The current language of the user is: (user's language). I need to use a tool to help me answer the question.
    Action: tool name (one of {tool_names}) if using a tool.
    Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
    ```
    
    Please ALWAYS start with a Thought. You can use {agent_scratchpad} to trace your thoughts.
    
    Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.
    
    If this format is used, the user will respond in the following format:
    
    ```
    Observation: tool response
    ```
    
    You should keep repeating the above format till you have enough information to answer the question without using any more tools. At that point, you MUST respond in the one of the following two formats:
    
    ```
    Thought: I can answer without using any more tools. I'll use the user's language to answer
    Answer: [your answer here (In the same language as the user's question)]
    ```
    
    ```
    Thought: I cannot answer the question with the provided tools.
    Answer: [your answer here (In the same language as the user's question)]
    ```
    
    ## Current Conversation
    
    Below is the current conversation consisting of interleaving human and assistant messages."""
    return template

def Agent_creation(dataframe_index,style_index):
    dataframe_engine = dataframe_index.as_query_engine(similarity_top_k=1)
    styling_engine = style_index.as_query_engine(similarity_top_k=1)
    template=agent_modified_pmt()
    prompt = PromptTemplate.from_template(template)
    # Builds the tools
    query_engine_tools = [
        QueryEngineTool(
            query_engine=dataframe_engine,
            metadata=ToolMetadata(
                name="dataframe_index",
                description="Provides information about the data in the data frame. Only use column names in this tool",
            ),
    \
        ),
        QueryEngineTool(
            query_engine=styling_engine,
            metadata=ToolMetadata(
                name="Styling",
                description="Provides instructions on how to style your Plotly plots"
                "Use a detailed plain text question as input to the tool.",
            ),
        ),
    ]
    llm=OpenAI(
        model="gpt-4-turbo",
        openai_api_key=st.session_state.api_key,
        temperature=0)

    try:
        agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True)
    except AttributeError as e:
        print(f"Error creating agent: {e}")
        print(f"LLM object: {llm.metadata}")
    return agent

def json_extra_prompt():
    pmt_add=""" Also give me a seperate output of a json string of format:
    '''{
    'type': 'Description of plot',
    'variables': ['columnA','columnB'],
    'columnA': 'x-axis',
    'columnB': 'y-axis',
    }'''
    where,
    -columnA is the exact column name for x-axis of the plot.
    -columnB is the exact column name for y-axis of the plot.
    
    """
    extra_info="Do ALL the intermediate calculations necessary for generating the plots. See if the variable is categorical or continuous to do these calculations accurately. Get information from the dataframe about the correct column names and make sure to style the plot properly and also give a title. DO NOT create sample data. ASSUME you have all the data in the dataframe 'data'."
    return pmt_add, extra_info

def parse_code_and_json(text):
    # Initialize variables
    python_code = ""
    json_data = ""
    current_block = None
    
    # Split text into lines for processing
    lines = text.strip().split('\n')
    
    for line in lines:
        # Check for code block markers
        if line.strip().startswith('```'):
            if current_block is None:  # Starting a new block
                block_type = line.strip().replace('```', '').lower()
                if block_type in ['python', 'json']:
                    current_block = block_type
            else:  # Ending current block
                current_block = None
            continue
            
        # If we're inside a block, add the line to appropriate content
        if current_block == 'python':
            python_code += line + '\n'
        elif current_block == 'json':
            json_data += line + '\n'
    
    # Clean up the extracted content
    python_code = python_code.strip()
    json_data = eval(json_data.strip())
    
    return python_code, json_data

def generate_plots_with_summary(data, column_type_desc, summary_stats):

    updated_col_type_desc = make_json_serializable(convert_dict(df = data, column_type_desc = column_type_desc))

    df_desc = get_col_desc(df = data)
    
    df_info = update_description(col_type_desc = updated_col_type_desc, llm_description = df_desc)

    path = "exports/dataframe.json"
    with open(path, "w") as fp:
        json.dump(df_info,fp) 
        
    instructions=agent_prompts(df_info)
    dataframe_index=df_index_creation(path)
    style_index=styling_index_creation()
    agent=Agent_creation(dataframe_index,style_index)
    json_extra, add_pmt=json_extra_prompt()

    plots_with_summary = []
    print(f'Total of {len(instructions)} visuals')
    for plot in instructions:
        try:
            plt_no = instructions.index(plot) + 1
            response=agent.chat(plot + json_extra + add_pmt)
            code,json_dict=parse_code_and_json(response.response)
            clean_code = code.replace('fig.show()','').strip()
            # Prepare a dictionary to hold the execution context
            exec_context = {'data': data}
            
            # Execute the dynamically generated code in the provided context
            exec(clean_code, {}, exec_context)
            
            # Extract the 'fig' object from the context
            fig = exec_context.get('fig')

            visual_summary = get_visual_summary(df = data, chart_metadata = json_dict, all_col_type_desc = column_type_desc, summary_stats = summary_stats)
            
            fig.write_image(f"exports/report_figures/plot_no_{plt_no}.png", width=800, height=600, scale=2)

            fig.write_html(f"exports/HTML_figures/plot_no_{plt_no}.html")

            plots_with_summary.append((fig, visual_summary, plt_no))
        except Exception as e:
            print(e)
    return plots_with_summary
