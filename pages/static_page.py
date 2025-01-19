import streamlit as st
import os
import time
import zipfile
import pandas as pd

from services.static_page.summary_stats import get_all_summary_stats
from services.static_page.generate_insights import generate_insight_llm
from services.static_page.visual_summary import get_visual_summary
from services.static_page.dynamic_visuals import generate_plots_with_summary
from services.static_page.pdf_generation import create_pdf_report

def create_combined_zip(pdf_path, html_folder_path, zip_file_path):
    """
    Creates a ZIP file containing the PDF report and a folder with all HTML files.
    
    Parameters:
        pdf_path (str): Path to the PDF file.
        html_folder_path (str): Path to the folder containing HTML files.
        zip_file_path (str): Path to save the combined ZIP file.
    """
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        # Add the PDF file to the ZIP
        if os.path.exists(pdf_path):
            zipf.write(pdf_path, os.path.basename(pdf_path))  # Add PDF file with its name
        else:
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Add the HTML folder and its contents to the ZIP
        for root, _, files in os.walk(html_folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Add files while preserving the folder structure
                arcname = os.path.relpath(file_path, os.path.dirname(html_folder_path))
                zipf.write(file_path, arcname)

def render():
    st.title("Data Insights")
    st.write("Upload a CSV or Excel file, process it, and download a generated PDF report.")

    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

    # Take API Key Input
    st.session_state.api_key = st.text_input(
        "Enter your OpenAI API key:",
        type="password",
        value=st.session_state.api_key
    )

    if not st.session_state.api_key:
        st.warning("Please enter your OpenAI API key to proceed.")
        return  # Stop execution until API key is provided

    # Set API key in environment
    os.environ["OPENAI_API_KEY"] = st.session_state.api_key
    
    pdf_path = "exports/data_insight_report.pdf"
    
    uploaded_file = st.file_uploader(
        "Upload data (.csv or .xlsx)",
        type=["csv", "xlsx"],
        key="static_file_uploader"
    )

    if "static_data_processed" not in st.session_state:
        st.session_state.static_data_processed = False
    if "static_uploaded_file" not in st.session_state:
        st.session_state.static_uploaded_file = None

    if uploaded_file is not None:
        if st.session_state.static_uploaded_file != uploaded_file:
            st.session_state.static_uploaded_file = uploaded_file
            st.session_state.static_data_processed = False

        file_type = uploaded_file.type
        valid_types = [
            "text/csv",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/octet-stream"
        ]
        if file_type in valid_types:
            st.success("Valid file uploaded!")

            if not st.session_state.static_data_processed:
                #Added status block for processing messages
                with st.status("Processing your data...", expanded=True) as status:
                    
                    st.write("🔄 Reading the data...")
                    if file_type == "text/csv":
                        df = pd.read_csv(uploaded_file)
                        data_read = True
                        st.write('✅ Successfully read the data')
                    elif file_type in [
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        "application/octet-stream"
                    ]:
                        df = pd.read_excel(uploaded_file, engine="openpyxl")
                        st.write('✅ Successfully read the data')
                    else:
                        st.error("Error reading the file")
                        
                    st.write("📊 Generating summary statistics...")
                    column_type_desc, freq_dict, summary_stat_dict, summary_stat_df, corr_mat = get_all_summary_stats(df)
                    print(column_type_desc, freq_dict, summary_stat_df, corr_mat)
                    st.write('✅ Successfully generated summary statistics')

                    st.write("📊 Generating insights...")
                    insights_dict = generate_insight_llm(df=df,
                                                         col_type_desc = column_type_desc,
                                                         freq_dict = freq_dict, 
                                                         summary_stats = summary_stat_df, 
                                                         corr_mat = corr_mat)
                    print(insights_dict)
                    st.write('✅ Successfully generated insights')

                    st.write("📊 Generating visuals...")
                    plots_with_summary = generate_plots_with_summary(data = df, 
                                                                     column_type_desc = column_type_desc, 
                                                                     summary_stats = summary_stat_dict)
                    st.write('✅ Successfully generated visuals')
                    
                    st.write("📄 Creating the PDF report...")
                    
                    create_pdf_report(output_path = pdf_path, 
                                      title = "Data Insights Report", 
                                      sections = insights_dict, 
                                      plots_with_summary = plots_with_summary, 
                                      freq_dict = freq_dict, 
                                      summary_stat_df = summary_stat_df)
                    
                    status.update(label="✅ Processing complete!", state="complete", expanded=False)

                st.session_state.static_data_processed = True


            try:
                html_folder_path = "exports/HTML_figures"
                zip_file_path = "exports/combined_report.zip"
                # Check if the PDF file exists
                if os.path.exists(pdf_path):
                    st.success("PDF report is available.")
                else:
                    st.error("The specified PDF file does not exist.")
                
                # Check if the folder contains HTML files
                html_files = [f for f in os.listdir(html_folder_path) if f.endswith('.html')]
                
                if html_files:
                    # st.success(f"Found {len(html_files)} HTML file(s) in the folder.")
                    
                    # Create the combined ZIP file
                    create_combined_zip(pdf_path, html_folder_path, zip_file_path)
                    st.success("PDF and HTML plots have been combined into a ZIP file.")

                    # Provide a download button for the ZIP file
                    with open(zip_file_path, "rb") as zip_file:
                        st.download_button(
                            label="📥 Download Combined ZIP File",
                            data=zip_file,
                            file_name=os.path.basename(zip_file_path),
                            mime="application/zip",
                        )
                else:
                    st.warning("No HTML plots found in the specified folder.")
                    # Provide a download button for the ZIP file
                    with open(pdf_path, "rb") as pdf_file:
                        st.download_button(
                            label="📥 Download PDF report",
                            data=pdf_file,
                            file_name=os.path.basename(pdf_path),
                            mime="application/zip",
                        )
            except Exception as e:
                st.error(f"An error occurred: {e}")

        else:
            st.error("Invalid file type. Please upload a CSV or Excel file.")

if __name__ == "__main__":
    render()
