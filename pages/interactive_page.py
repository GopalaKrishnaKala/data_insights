import streamlit as st
import streamlit as st
import pandas as pd
import io
import os
import matplotlib.pyplot as plt
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
from openai import OpenAI as OpenAIClient
from services.interactive_page.get_completion import get_completion
from services.interactive_page.save_chart import save_chart
from services.interactive_page.create_pdf import create_pdf


def render():

    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

    st.session_state.api_key = st.text_input(
        "Enter your OpenAI API key:",
        type="password",
        value=st.session_state.api_key
    )

    if not st.session_state.api_key:
        st.warning("Please enter your OpenAI API key to proceed.")
        return

    #API_KEY = os.getenv('OPENAI_API_KEY')
    
    llm = OpenAI(api_token=st.session_state.api_key)
    client = OpenAIClient(api_key=st.session_state.api_key)

    st.title("Talk with Your Data")
    st.write("Upload a CSV or Excel file and then chat with your data using our AI assistant.")

    if "interactive_uploaded_file" not in st.session_state:
        st.session_state.interactive_uploaded_file = None
    if "interactive_chat_history" not in st.session_state:
        st.session_state.interactive_chat_history = []
    if "responses_list" not in st.session_state:
        st.session_state.responses_list = []
    if "question_count" not in st.session_state:
        st.session_state.question_count = 0
    if "chart_paths" not in st.session_state:
        st.session_state.chart_paths = []

    uploaded_file = st.file_uploader("Upload data (.csv or .xlsx)", type=["csv", "xlsx"], key="interactive_file_uploader")

    if uploaded_file is not None:
        st.session_state.interactive_uploaded_file = uploaded_file
        st.success("File uploaded successfully!")

        try:
            if uploaded_file.name.endswith(".csv"):
                st.session_state.interactive_df = pd.read_csv(uploaded_file)
            else:
                st.session_state.interactive_df = pd.read_excel(uploaded_file)

            st.write("**Below is a small preview of the data. Please change the file if this is not the intended file you want to perform analysis on.**")
            st.write("**Data Preview**", st.session_state.interactive_df.head())
            #st.write("**Columns**:", st.session_state.interactive_df.columns.tolist())

            st.session_state.sdf = SmartDataframe(
                st.session_state.interactive_df,
                config={"verbose": True, "save_logs": True, "save_charts": True, "llm": llm, "enforce_restricted_mode": False, "enable_cache": False, "security": "none"}
            )
            st.info("DataFrame loaded. You can now ask questions below!")
        except Exception as e:
            st.error(f"Error loading file: Please upload only a CSV of Excel file")

        for question, (pandasai_answer, summary_answer) in st.session_state.interactive_chat_history:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                if isinstance(pandasai_answer, plt.Figure):
                    st.pyplot(pandasai_answer)
                elif isinstance(pandasai_answer, pd.DataFrame):
                    st.dataframe(pandasai_answer)
                elif isinstance(pandasai_answer, str) and pandasai_answer.endswith(".png"):
                    st.image(pandasai_answer)
                
                    #st.write(pandasai_answer)
                if summary_answer != "" and summary_answer is not None:
                    st.write(f"**Summary**: {summary_answer}")

        user_question = st.chat_input("Ask about your data...")
        if user_question:
            with st.chat_message("user"):
                st.write(user_question)

            with st.spinner("Thinking..."):
                try:
                    pandasai_result = st.session_state.sdf.chat(user_question)
                    chart_path = save_chart(pandasai_result)
                    st.session_state.chart_paths.append(chart_path)
                except Exception as e:
                    pandasai_result = f"Error: {e}"
                    st.session_state.chart_paths.append(None)

            with st.spinner("Processing..."):
                try:
                    if isinstance(pandasai_result, plt.Figure):
                        summary_result = ""
                    elif (
                        isinstance(pandasai_result, str)
                        and "exports/charts" in pandasai_result
                        and pandasai_result.endswith(".png")
                    ):
                        summary_result = ""
                    elif isinstance(pandasai_result, pd.DataFrame):
                        summary_prompt = f"""
The user asked: "{user_question}"

The output is a table. Below is the data:

{pandasai_result.to_string()}

Please provide a summary of any interesting trends or observations in plain English. Insights that would be hepful in understanding more about the data and performning Analytics.
                        """
                        summary_result = get_completion(summary_prompt, client)
                    else:
                        summary_prompt = f"""
The user asked: "{user_question}"

The output is:

{pandasai_result}

Please provide a summary in plain English, highlighting any interesting trends or observations. Insights that would be hepful in understanding more about the data and performning Analytics.
                        """
                        summary_result = get_completion(summary_prompt, client)
                except Exception as e:
                    summary_result = f"Error while summarizing: {e}"

            st.session_state.interactive_chat_history.append(
                (user_question, (pandasai_result, summary_result))
            )
            st.session_state.responses_list.append({
                "question": user_question,
                "answer": summary_result,
                "image_path": chart_path  # Add the chart path here
            })

            
            st.session_state.question_count += 1

            if isinstance(pandasai_result, plt.Figure):
                with st.chat_message("assistant"):
                    st.pyplot(pandasai_result)
            elif isinstance(pandasai_result, pd.DataFrame):
                with st.chat_message("assistant"):
                    st.dataframe(pandasai_result)
                    st.write(f"**Summary**: {summary_result}")
            elif (
                isinstance(pandasai_result, str)
                and "exports/charts" in pandasai_result
                and pandasai_result.endswith(".png")
            ):
                with st.chat_message("assistant"):
                    st.image(pandasai_result)
            elif summary_result and summary_result.strip():  # Check if summary_result exists and is not empty
                with st.chat_message("assistant"):
                    st.write(f"**Summary**: {summary_result}")

        if st.session_state.question_count >= 2:
            st.success("You can now download the summary PDF!")
            if st.button("Download Summary PDF"):
                print("Chart paths:", st.session_state.chart_paths)  # Debugging statement
                overall_prompt = "This is the list of questions asked by the user and list of responses received by Pandas AI:\n\n"
                
                for response in st.session_state.responses_list:
                    overall_prompt += f"Q: {response['question']}\nA: {response['answer']}\n\n"

                overall_prompt += "Please summarize everything that you have found and help me understand the data better. If there is more than one dataset that the user asks Pandas AI, give a detailed sumamry of each dataset in different paragraphs."
        
                overall_summary = get_completion(overall_prompt, client)

                buffer = io.BytesIO()
                create_pdf(buffer, st.session_state.responses_list, overall_summary)
                buffer.seek(0)
                st.download_button(
                    label="📥 Download Summary PDF",
                    data=buffer,
                    file_name="data_insights_summary.pdf",
                    mime="application/pdf"
                )

    else:
        st.info("Please upload a CSV or Excel file to start chatting with your data.")


if __name__ == "__main__":
    render()
