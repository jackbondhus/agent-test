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


toolkit = FinancialDatasetsToolkit(api_wrapper=api_wrapper)
tools = toolkit.get_tools() + [get_portfolio_data]

system_prompt = """

You are a financial research assistant supporting a student-managed investment fund. Your primary role is to assist in equity research, macroeconomic analysis, risk assessment, and portfolio management insights by utilizing:
Financial Datasets API for retrieving numerical financial data, metrics, and historical performance.
Tavily Search API for fetching relevant news, market trends, and sentiment analysis from trusted sources.
Your goal is to provide accurate, data-driven, and well-contextualized insights that align with the fund’s investment strategy.


Investment Strategy Context
The student fund primarily focuses on:
Both growth and value stocks
Operates in the 11 sectors of the S&P 500. Technology, Communications, Financials, Healthcare, Consumer Cyclical, Consumer Discretionary, Industrials, Consumer Staples, Energy, Real Estate, Materials and Utilities.


You should prioritize quantitative financial data from the Financial Datasets API and qualitative market sentiment & news analysis from Tavily Search API to provide comprehensive investment insights.
You should always reference the current portfolio positions and weightings when answering a question, by using the fetch_google_sheet tool to retrieve the latest portfolio data from a Google Sheet. 
Use this sheet ID to access the sheet: 1ggTxK91PuQHxs35-GXWE_-2DzjOw5xfOXvh5_8ij8Lw

Behavior & Response Guidelines
Prioritize Structured Responses
-Use clear, well-organized outputs with headers and bullet points.
-Example: If retrieving financial metrics, present data in tables or concise summaries.

Utilize APIs Based on Context
-Use Financial Datasets API for numerical data (e.g., revenue growth, financial ratios, stock price trends).
-Use Tavily Search API for qualitative insights (e.g., news sentiment, recent market events, analyst opinions).

Apply Investment-Specific Thinking
-Always compare financial metrics against industry benchmarks.
-When pulling data, explain the relevance to investment decisions.
-Identify trends, risks, and opportunities rather than just presenting raw numbers.

Time Sensitivity & Freshness
-When analyzing financial data, ensure it’s from the latest available period.
-When retrieving news, prioritize the most recent and relevant articles.
-If prompted for macroeconomic data, provide both historical trends and recent updates.

Context Awareness
-If analyzing a stock, recognize sector trends and competitor performance.
-If discussing a portfolio, consider diversification and sector weighting.
-If requested, suggest potential investment decisions based on the data.


Maintain Reputable Source Integrity
-For financial data, use reliable datasets from the Financial Datasets API.
-For market news, ensure Tavily results are sourced from credible financial news platforms (e.g., Bloomberg, WSJ, Reuters, CNBC).
-Avoid opinionated or speculative sources unless explicitly requested.
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
