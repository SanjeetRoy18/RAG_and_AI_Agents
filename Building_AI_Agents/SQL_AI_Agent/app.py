import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from dotenv import load_dotenv

# ==========================================
# 0. LOAD ENVIRONMENT VARIABLES FROM .ENV
# ==========================================
load_dotenv()

# ==========================================
# 1. STREAMLIT UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="Groq SQL AI Agent", page_icon="📊", layout="centered")
st.title("📊 Chat with Your SQLite Database")
st.write("Powered by Groq & Llama 4 Scout. Ask any question in natural language!")

# ==========================================
# 2. CONFIGURATION & DATABASE INTEGRATION
# ==========================================
# Ensure your GROQ API key is configured
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Please set your GROQ_API_KEY in a .env file.")
    st.stop()

# Model and Local SQLite File settings
MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"
DB_FILE_PATH = "Chinook_Sqlite.sqlite"  # Assumes the file is in your project directory

@st.cache_resource
def initialize_sql_agent():
    """Initializes and caches the Groq LLM and SQL Agent Executor."""
    
    # Initialize Groq Chat Model via LangChain
    llm = ChatGroq(
        model=MODEL_ID,
        temperature=0.1,  # Kept low for accurate SQL generation
        max_tokens=1024
    )
    
    # Connect to the local SQLite Database (Option A)
    if not os.path.exists(DB_FILE_PATH):
        raise FileNotFoundError(f"Database file '{DB_FILE_PATH}' not found in the project directory.")
        
    db = SQLDatabase.from_uri(f"sqlite:///{DB_FILE_PATH}")
    
    # Create the LangChain SQL Agent Executor
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        verbose=True, 
        handle_parsing_errors=True
    )
    return agent_executor

# Initialize the agent
try:
    agent_executor = initialize_sql_agent()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# ==========================================
# 3. CHAT INTERFACE & STATE MANAGEMENT
# ==========================================
# Initialize chat history session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("e.g., Which artist has the most albums?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate agent response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.spinner("Querying database via Groq..."):
            try:
                # Invoke the agent executor with the user prompt
                result = agent_executor.invoke({"input": prompt})
                output_text = result.get("output", "I couldn't retrieve an answer.")
                response_placeholder.markdown(output_text)
                
                # Save assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": output_text})
            except Exception as e:
                error_msg = f"An error occurred while executing the query: {e}"
                response_placeholder.error(error_msg)