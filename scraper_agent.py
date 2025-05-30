from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langgraph.prebuilt import create_react_agent
from langchain_community.document_loaders.html import UnstructuredHTMLLoader
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from datamule import Portfolio
import tarfile
import json
@tool
def sec_filing(stock_name:str, ticker:str, end_date:str, start_date:str):
    """downloads sec filings for given stock and extracts files
    stock_name: name of folder to be stored (stock name)
    ticker: ticker symbol of the stock
    start_date: from filing list (yyyy-mm-dd)
    end_date: to filing list (yyyy-mm-dd)
    """
    
    # Create directories if they don't exist
    os.makedirs(f'portfolio/{stock_name}', exist_ok=True)
    
    # Download submissions
    portfolio = Portfolio(f'portfolio/{stock_name}')
    portfolio.download_submissions(ticker=ticker, filing_date=(start_date, end_date))
    
    # Process each tar file
    for tar in os.listdir(f'portfolio/{stock_name}'):
        # Create extract directory
        extract_path = f'extracts/{stock_name}/{tar}'
        os.makedirs(extract_path, exist_ok=True)
        
        # Extract the tar file
        tar_path = f'portfolio/{stock_name}/{tar}'
    # Create directories if they don't exist
    os.makedirs(f'portfolio/{stock_name}', exist_ok=True)
    
    # Download submissions
    portfolio = Portfolio(f'portfolio/{stock_name}')
    portfolio.download_submissions(ticker=ticker, filing_date=(start_date, end_date))
    
    # Process each tar file
    for tar in os.listdir(f'portfolio/{stock_name}'):
        # Create extract directory
        extract_path = f'extracts/{stock_name}/{tar}'
        os.makedirs(extract_path, exist_ok=True)
        
        # Extract the tar file
        tar_path = f'portfolio/{stock_name}/{tar}'
        load = tarfile.open(tar_path)
        load.extractall(extract_path)
    
    return f"Downloaded filings for {stock_name} and extracted files in {stock_name} directory"


tools= [TavilySearch(
    max_results=3,
    topic="finance",
    api_key=os.getenv("TAVILY_API_KEY"),
    # include_answer=False,
    # include_raw_content=False,
    # include_images=False,
    # include_image_descriptions=False,
    # search_depth="basic",
    # time_range="day",
    # include_domains=None,
    # exclude_domains=None
)]

grok_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.3-70b-versatile",api_key=grok_api_key)
tools=[sec_filing]
agent=create_react_agent(model=llm, tools=tools)
input_message = {
    "role": "user",
    "content": "1.get sec filings of apple for may 2025",
}
for step in agent.stream(
    {"messages": [input_message]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()