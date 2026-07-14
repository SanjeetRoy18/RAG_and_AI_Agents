import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_ollama import ChatOllama

# 1. Load Environment Variables & API Key
load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# if not GROQ_API_KEY:
#     st.error("Please set your GROQ_API_KEY in a .env file.")
#     st.stop()

# 2. Streamlit Page Configuration
st.set_page_config(page_title="LLM Data Visualizer", page_icon="📊", layout="wide")
st.title("📊 LLM Data Analytics & Visualization Agent")
st.write("Upload a CSV file and ask the AI to generate insights, tables, or charts!")

# 3. File Upload Interface
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Display a preview of the dataset
    st.subheader("Data Preview")
    st.dataframe(df)
    
    # 4. Initialize the Groq LLM
    # We use llama-3.3-70b-versatile for complex reasoning/coding tasks
    llm = ChatOllama(
        temperature=0, 
        model="llama3.2",
        num_ctx=4096
    )
    
    # 5. Initialize the Data Agent
    # allow_dangerous_code=True lets the agent run code locally inside your venv
    agent = create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True, 
        allow_dangerous_code=True,
        handle_parsing_errors=True
    )
    
    st.divider()
    
    # 6. User Query & Interface Loop
    st.subheader("Ask the Agent")
    user_query = st.text_input(
        "What would you like to know or visualize?",
        placeholder="e.g., 'Plot a correlation matrix heatmap' or 'What is the average age of students?'"
    )
    
    if user_query:
        with st.spinner("Analyzing data and generating output..."):
            try:
                # Custom prompt tailoring to ensure it handles visual graphs nicely
                tailored_prompt = (
                    f"{user_query}. "
                    "If generating a matplotlib/seaborn chart, ensure you call 'plt.savefig(\"temp_chart.png\")' "
                    "at the end of your code script execution block so it is saved locally."
                )
                
                # Run the agent
                response = agent.run(tailored_prompt)
                
                # Display textual/tabular insights
                st.subheader("Analysis & Insights")
                st.write(response)
                
                # Check if the agent created a chart and render it
                if os.path.exists("temp_chart.png"):
                    st.subheader("Generated Visualization")
                    st.image("temp_chart.png")
                    # Clean up the image file so it doesn't leak into the next prompt
                    os.remove("temp_chart.png")
                    
            except Exception as e:
                # Check if the error is due to limits being reached
                error_message = str(e)
                if "iteration limit" in error_message or "time limit" in error_message:
                    st.subheader("Analysis & Insights")
                    st.warning(
                        "⏱️ **The query was too complex to finish in time.**\n\n"
                        "The agent tried to process your request but timed out while refining the code. "
                        "Try breaking your question down into smaller steps (e.g., ask for the data summary first, "
                        "then ask to plot it)."
                    )
                else:
                    # Handle other unexpected errors normally
                    st.error(f"An error occurred: {error_message}")