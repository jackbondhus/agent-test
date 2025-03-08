import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.financial_datasets.toolkit import FinancialDatasetsToolkit
from langchain_community.utilities.financial_datasets import FinancialDatasetsAPIWrapper
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
import requests
import json

load_dotenv()

os.environ["FINANCIAL_DATASETS_API_KEY"] = os.getenv("FINANCIAL_DATASETS_API_KEY")
os.environ["PPLX_API_KEY"] = os.getenv("PPLX_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SEARCH_API_KEY"] = os.getenv("SEARCH_API_KEY")

# Initialize Language Model
model = ChatOpenAI(model="gpt-3.5-turbo-16k")

# Initialize Financial Data API Wrapper
api_wrapper = FinancialDatasetsAPIWrapper(
    financial_datasets_api_key=os.environ["FINANCIAL_DATASETS_API_KEY"])

# Initialize Financial Toolkit
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
# Combine all tools
tools = financial_tools

# Define System Prompt
system_prompt = """
You are an advanced financial analysis AI assistant equipped with specialized tools
to access and analyze financial data and retrieve the latest company news.
Your primary function is to help users with:

1. Financial analysis by retrieving and interpreting income statements, balance sheets, and cash flow statements.
2. Searching for recent news articles about publicly traded companies to provide context for market movements.

Your available tools:

1. **Financial Datasets**: Use this tool to fetch financial statements.
2. **Company News Search**: Use this tool to fetch the latest news about a company.

Capabilities:

- Retrieve and analyze financial reports.
- Identify trends and key financial ratios.
- Explain financial concepts in simple terms.
- Fetch and summarize recent news articles related to a company.

When answering:
- Clearly specify which tool's data you're referencing.
- Provide context and reasoning in your analysis.
- Ask for clarification if needed.
"""

# Define Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Initialize Memory (Optional for Conversational Context)
memory = ConversationBufferMemory(memory_key="chat_history")

# Create Agent
agent = create_tool_calling_agent(model, tools, prompt)

# Initialize Agent Executor
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)

# Streaming Output for Real-Time Response
def stream_response(user_input):
    response = agent_executor.stream({"input": user_input})
    for chunk in response:
        print(chunk, end='', flush=True)  # Print response as it streams

# Example Usage
user_prompt = "Get the balance sheet for AAPL and recent news about it."
stream_response(user_prompt)