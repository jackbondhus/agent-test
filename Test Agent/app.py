import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.financial_datasets.toolkit import FinancialDatasetsToolkit
from langchain_community.utilities.financial_datasets import FinancialDatasetsAPIWrapper
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_community.tools import TavilySearchResults, DuckDuckGoSearchRun
import streamlit as st
import pandas as pd
from langchain_community.callbacks.streamlit import (StreamlitCallbackHandler)
import requests
from datetime import datetime
import json

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["FINANCIAL_DATASETS_API_KEY"] = os.getenv("FINANCIAL_DATASETS_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

model = ChatOpenAI(model="gpt-4o")

api_wrapper = FinancialDatasetsAPIWrapper(
    financial_datasets_api_key=os.environ["FINANCIAL_DATASETS_API_KEY"])

search = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)

from io import StringIO

@tool
def get_portfolio_data() -> str:
    """
    Fetches portfolio data from a known public Google Sheet (CSV-compatible)
    and returns the data as a CSV string.

    The agent can parse this CSV to answer questions about the portfolio's 
    holdings, weightings, etc.
    """
    try:
        # Hard-coded sheet ID (assuming it's always the same)
        sheet_id = "1ggTxK91PuQHxs35-GXWE_-2DzjOw5xfOXvh5_8ij8Lw"

        # Construct the CSV URL
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv"

        # Read the CSV data into a Pandas DataFrame
        df = pd.read_csv(csv_url)

        # Convert the DataFrame to a CSV string in memory
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)

        # Return the CSV string (this is what the agent sees)
        return csv_buffer.getvalue()

    except Exception as e:
        # Return an error message if something goes wrong
        return f"Error fetching Google Sheet data: {str(e)}"




@tool
def current_date() -> str:
    """Returns the current system date."""
    return datetime.now().strftime("%Y-%m-%d")


toolkit = FinancialDatasetsToolkit(api_wrapper=api_wrapper)
tools = toolkit.get_tools() + [get_portfolio_data, search, current_date]

system_prompt = """

Role & Objective:
You are a financial research assistant for a student-managed investment fund. Your goal is to provide accurate, data-driven financial analysis and current market news 
using both quantitative data and qualitative insights.

Data Retrieval & Tool Usage:

Use the get_portfolio_data tool to fetch the latest portfolio information from our designated Google Sheet.

Call the Financial Datasets API for numerical financial metrics.

Use the Tavily Search API to retrieve current market news and sentiment analysis.

Current Date Enforcement:
At the start of every analysis or report, call the current_date tool to retrieve today’s date (YYYY-MM-DD). All data, news, and analysis must reference information that is 
current with this date. Avoid incorporating any historical data points unless explicitly stated for trend analysis. For instance, while evaluating Tesla’s performance, 
ensure that news items, performance metrics, and recommendations are drawn from data as of the current date rather than relying on previous reporting periods.

Response Structure:
Organize your output with headers and bullet points for clarity. Prioritize recent data and explain how the information supports current investment insights and strategies.



"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    ("ai", "{agent_scratchpad}")  
])

memory = ConversationBufferMemory(memory_key="chat_history")

agent = create_tool_calling_agent(model, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)

st.set_page_config(layout="wide")
st.title("Financial Analysis AI Assistant")
st.write("Ask me about financial data and recent company news")
st_callback = StreamlitCallbackHandler(st.container())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if prompt := st.chat_input():
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor.invoke(
            {"input": prompt}, {"callbacks": [st_callback]}
        )
        st.session_state.chat_history.append({"role": "assistant", "content": response["output"]})
        st.write(response["output"])

for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])
