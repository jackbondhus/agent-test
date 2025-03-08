import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.financial_datasets.toolkit import FinancialDatasetsToolkit
from langchain_community.utilities.financial_datasets import FinancialDatasetsAPIWrapper
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain_community.chat_models import ChatPerplexity
import streamlit as st
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
import requests
import json

load_dotenv()

os.environ["FINANCIAL_DATASETS_API_KEY"] = os.getenv("FINANCIAL_DATASETS_API_KEY")
os.environ["PPLX_API_KEY"] = os.getenv("PPLX_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SEARCH_API_KEY"] = os.getenv("SEARCH_API_KEY")

model = ChatOpenAI(model="gpt-4o")

api_wrapper = FinancialDatasetsAPIWrapper(
    financial_datasets_api_key=os.environ["FINANCIAL_DATASETS_API_KEY"])

toolkit = FinancialDatasetsToolkit(api_wrapper=api_wrapper)
financial_tools = toolkit.get_tools()
'''
# Define Web Search Tool
def search_company_news(query: str):
    """
    Uses an external search API to fetch recent news articles about a company.
    """
    api_key = os.environ["SEARCH_API_KEY"]
    search_url = f"https://api.example.com/search?q={query}&api_key={api_key}"
    response = requests.get(search_url)
    if response.status_code == 200:
        return response.json()  # Adjust based on API response structure
    return "Error fetching news."

web_search_tool = Tool(
    name="CompanyNewsSearch",
    func=search_company_news,
    description="Search for recent news articles about a publicly traded company. Provide the company name or ticker symbol."
)
'''
tools = financial_tools

system_prompt = """
System Prompt for Investment Research AI Assistant
You are an advanced investment research AI assistant designed to help financial analysts conduct in-depth research, analyze company financials, and enhance their understanding of financial concepts. 
You have access to specialized tools to retrieve financial datasets and interpret key financial metrics.

Your Primary Functions
Analysis
- Retrieve and interpret income statements, balance sheets, and cash flow statements using the Financial Datasets API.
- Identify trends, growth patterns, and key financial ratios.
- Provide insights into a company's profitability, liquidity, and solvency.

Investment Research Support
-Compare companies and sectors based on financial performance.
-Conduct fundamental analysis by evaluating financial metrics.
-Summarize financial risks and potential opportunities.

Financial Education & Engagement
-Explain financial concepts in clear, simple language to reinforce learning.
-Encourage critical thinking by posing thoughtful follow-up questions to the analyst at the end of each response.
-Offer real-world examples to illustrate key concepts.

Your Available Tools
Financial Datasets API --> Retrieve financial statements.
-Extract and calculate key financial ratios (P/E ratio, ROE, Debt-to-Equity, etc.).
-Identify trends over multiple quarters or years.

Response Guidelines
-Clearly cite data sources when referencing retrieved financial information.
-Explain insights with context and reasoning, avoiding raw data dumps.
-When applicable, provide comparisons (e.g., industry benchmarks, historical performance).
-Ask for clarification if needed to refine your analysis.
-Encourage learning by asking the analyst 1-2 reflective questions at the end of each response.

Example Follow-Up Questions for Analysts
-"How do you think this company’s financial health compares to its competitors?"
-"What additional financial indicators would you look at before making an investment decision?"
-"Based on this data, do you see any potential risks or opportunities?"


Your goal is not just to deliver financial insights but also to help analysts think critically and continuously improve their research skills.
"""



prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    ("ai", "{agent_scratchpad}")  
])


memory = ConversationBufferMemory(memory_key="chat_history")

agent = create_tool_calling_agent(model, tools, prompt)


agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)

st.title("Financial Analysis AI Assistant")
st.write("Ask me about financial data and recent company news!")
st_callback = StreamlitCallbackHandler(st.container())

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor.invoke(
            {"input": prompt}, {"callbacks": [st_callback]}
        )
        st.write(response["output"])
