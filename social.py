import streamlit as st
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("API key not found. Please check your .env file.")
else:
    os.environ["OPENAI_API_KEY"] = api_key  # Set API key as environment variable for OpenAI

# Database setup
engine = create_engine("sqlite:///company.db")
db = SQLDatabase(engine=engine)
llm = ChatOpenAI(model="gpt-4o-mini")
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

# Data dictionary for context
data_dictionary = """
| Column Name         | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| un_id               | Unique identifier for each taxpayer/entity                                 |
| name                | Name of the company                                            |
| vat                 | VAT (Value Added Tax) amount associated with the taxpayer                  |
| payment             | Payment amount made by the taxpayer                                        |
| principal_debt      | Principal debt amount owed by the taxpayer                                 |
| forfeit             | Amount of penalty or forfeiture applied to the taxpayer                   |
| sanction            | Sanction amount associated with the taxpayer                              |
| payment_short_ind   | Indicator of whether there is a shortfall in payment (1 = Yes, 0 = No)     |
| VAR9                | Additional variable for which details are not provided (e.g., could be null or unspecified) |
"""


# Streamlit UI setup
st.title("Ask Ghassan")
st.write("Ask me anything!")

# Chatbot conversation state
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# User input
user_input = st.text_input("You:", key="user_input")

if user_input:
    # Add the data dictionary to the input for better context
    input_text = f"Refer to the following data dictionary for context:\n\n{data_dictionary}\n\n{user_input}"
    # Query the RAG model
    result = agent_executor.invoke({"input": input_text})["output"]
    # Append conversation history
    st.session_state.conversation.append(("User", user_input))
    st.session_state.conversation.append(("Ghassan", result))
    user_input = ""  # Clear input after submission

# Display conversation history in a container with autoscroll enabled
with st.container():
    for speaker, text in st.session_state.conversation:
        if speaker == "User":
            st.write(f"**You:** {text}")
        else:
            st.write(f"**Ghassan:** {text}")
    # Automatically scrolls to the latest conversation entry
    st_autoscroll = True
