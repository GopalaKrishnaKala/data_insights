import streamlit as st

def render():
    st.title("Introduction")
    st.write("Tool description...")
    st.write(
        """
        - **Static Page**: Upload CSV/Excel files and generate a PDF report.
        - **Interactive Page**: Upload CSV/Excel files and chat with a simple chatbot.
        """
    )
