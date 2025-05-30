from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,AIMessage
from langchain_groq import ChatGroq
import os
import asyncio
load_dotenv()
grok_api_key = os.getenv("GROQ_API_KEY")
ALPHA_VANTAGE_API_KEY =os.getenv("ALPHA_VANTAGE_API_KEY")
alpha_client = MultiServerMCPClient(
    {
    "alphavantage": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "-e",
        "ALPHA_VANTAGE_API_KEY",
        "mcp/alpha-vantage"
      ],
      "env": {
        "ALPHA_VANTAGE_API_KEY": "{ALPHA_VANTAGE_API_KEY}"
      },
      "transport":"stdio"
    }
    }
)
yahoo_client = MultiServerMCPClient(
    {
    "yfinance": {
      "command": "uv",
      "args": [
        "--directory",
        "yahoo-finance-mcp",
        "run",
        "server.py"
      ],
       "transport":"stdio"
    }
  }
)
alpha_client0=MultiServerMCPClient(
    {
    "alphavantage": {
      "command": "uv",
      "args": [
        "--directory",
        "alphavantage",
        "run",
        "alphavantage"
      ],
      "env": {
        "ALPHAVANTAGE_API_KEY": "{ALPHA_VANTAGE_API_KEY}"
      },
      "transport":"stdio"
    }
  }
)
async def main():
    tools= await yahoo_client.get_tools()
    llm=ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",api_key=grok_api_key)
    agent=create_react_agent(model=llm,tools=tools)
    response =await agent.ainvoke({"messages": [HumanMessage(content="""get stock info of apple """),]},{"recursion_limit": 25})
    resp= response["messages"]
    for i in range(len(resp)):
       if isinstance(resp[i], AIMessage):
          print(f"AI: {resp[i].content}")
       elif isinstance(resp[i], HumanMessage):
          print(f"Human: {resp[i].content}")
       else:
          print(f"TOOL: {resp[i].content}")
asyncio.run(main())
