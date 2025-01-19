import streamlit as st

from pages.intro_page import render as render_intro
from pages.static_page import render as render_static
from pages.interactive_page import render as render_interactive

import os
import glob

# Specify the folder where PNG files are located
folder_path1 = "exports/charts"
folder_path2 = "exports/report_figures"
folder_path3 = "exports/HTML_figures"

os.makedirs(folder_path1, exist_ok=True)
os.makedirs(folder_path2, exist_ok=True)
os.makedirs(folder_path3, exist_ok=True)

def delete_html_files(directory):
    """
    Recursively deletes all .html files within the specified directory and its subdirectories.
    
    Parameters:
        directory (str): The path of the directory to search for .html files.
    """
    try:
        # List files in the specified directory
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            # Check if it's a file and ends with .html
            if os.path.isfile(file_path) and file.endswith('.html'):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
    except Exception as e:
        print(f"Error processing directory {directory}: {e}")

# Example usage
directory_path = "exports/HTML_figures"
delete_html_files(directory_path)



# Set page configuration
st.set_page_config(page_title="Data Insights Tool", layout="wide")

# Custom CSS to style the buttons and hide unnecessary sidebar items
st.markdown(
    """
    <style>
    /* Hide the Streamlit app menu and unnecessary links */
    [data-testid="stSidebarNav"] {
        display: none;
    }
    
    /* Style the sidebar buttons */
    .stSidebar .stButton button {
        width: 100%;
        background-color: #f0f0f0;
        border: 1px solid #ccc;
        border-radius: 5px;
        font-size: 16px;
        padding: 10px;
        margin-top: 10px;
    }
    .stSidebar .stButton button:hover {
        background-color: #ddd;
        border-color: #888;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation
st.sidebar.title("Data Insights Tool")
if "current_page" not in st.session_state:
    st.session_state.current_page = "Introduction"

# Sidebar buttons for navigation
if st.sidebar.button("Introduction"):
    st.session_state.current_page = "Introduction"
if st.sidebar.button("Static"):
    st.session_state.current_page = "Static"
if st.sidebar.button("Interactive"):
    st.session_state.current_page = "Interactive"

# Page Routing
if st.session_state.current_page == "Introduction":
    render_intro()
elif st.session_state.current_page == "Static":
    render_static()
elif st.session_state.current_page == "Interactive":
    render_interactive()
