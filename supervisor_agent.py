from langgraph_supervisor import create_supervisor
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_mcp_adapters.client import MultiServerMCPClient
import os
from dotenv import load_dotenv
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langgraph.prebuilt import create_react_agent
from langchain_community.document_loaders.html import UnstructuredHTMLLoader
from langchain_core.messages import HumanMessage, convert_to_messages
from langchain_core.tools import tool
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langgraph.prebuilt import create_react_agent
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from datamule import Portfolio
import tarfile
import json
import asyncio
import os
import uuid
from groq import Groq
import speech_recognition as sr
# Load environment variables
load_dotenv()
grok_api_key = os.getenv("GROQ_API_KEY")
nvidia_api_key = os.getenv("NVIDIA_API_KEY")
url = os.getenv('QDRANT_URL')
api_key = os.getenv('QDRANT_API_KEY')

# Initialize Qdrant client and vector store
client = QdrantClient(url=url, api_key=api_key)
collection_name = "ragaaiv0"
id_key = 'docs_id'

if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
    )

vector_store = QdrantVectorStore(client=client, embedding=NVIDIAEmbeddings(model="nvidia/nv-embed-v1", nvidia_api_key=nvidia_api_key), collection_name=collection_name)

# Initialize Groq client
groq_client = Groq()

# Define tools
@tool
def sec_filing(stock_name: str, ticker: str, filing_date: tuple = ('2022-05-01', '2022-05-22')):
    """Downloads SEC filings for a given stock and extracts files."""
    try:
        os.makedirs(f'portfolio/{stock_name}', exist_ok=True)
        portfolio = Portfolio(f'portfolio/{stock_name}')
        portfolio.download_submissions(ticker=ticker, filing_date=filing_date)

        for tar in os.listdir(f'portfolio/{stock_name}'):
            extract_path = f'extracts/{stock_name}/{tar}'
            os.makedirs(extract_path, exist_ok=True)

            tar_path = f'portfolio/{stock_name}/{tar}'
            with tarfile.open(tar_path) as load:
                load.extractall(extract_path)

        return f"Downloaded filings for {stock_name} and extracted files in {stock_name} directory"
    except Exception as e:
        return f"Error processing SEC filings: {e}"

@tool
async def create_summary(stock_name: str) -> str:
    """Creates summaries of files in folder stock_name and stores them in vector DB."""
    path = f'extracts/{stock_name}'
    summaries = []

    try:
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            for file in os.listdir(folder_path):
                if file.endswith(".htm") and not file.startswith("R"):
                    file_path = os.path.join(folder_path, file)
                    loader = UnstructuredHTMLLoader(file_path)
                    docs = loader.load()
                    if not docs:
                        continue

                    content = docs[0].page_content
                    prompt = f"""
                    You are responsible for concisely summarizing a financial table or text chunk:

                    {content}

                    Extract the key financial insights and present them in a clear summary.
                    Focus on metrics, trends, and important facts.
                    """
                    response = await ChatNVIDIA(model="meta/llama-3.3-70b-instruct", nvidia_api_key=nvidia_api_key).ainvoke([HumanMessage(content=prompt)])
                    summaries.append(response.content)

        doc_ids = [str(uuid.uuid4()) for _ in summaries]
        summary_docs = [Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(summaries)]
        vector_store.add_documents(summary_docs)

        return f"Stored {len(summaries)} summaries from folder '{stock_name}' in vector database."
    except Exception as e:
        return f"Error creating summaries: {e}"

@tool
async def search_documents(query: str) -> str:
    """Search for relevant documents using the query."""
    try:
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        results = await retriever.ainvoke(query)
        content = ''.join([doc.page_content for doc in results])
        return f"Here are the most relevant documents:\n\n{content}"
    except Exception as e:
        return f"Error searching documents: {e}"

# Utility functions
async def initialize_agents():
    """Initialize agents asynchronously."""
    api_tools = await MultiServerMCPClient({
        "yfinance": {
            "command": "uv",
            "args": ["--directory", "yahoo-finance-mcp", "run", "server.py"],
            "transport": "stdio"
        }
    }).get_tools()

    api_llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", nvidia_api_key=nvidia_api_key,temperature=0.5)
    return create_react_agent(name='api_agent', model=api_llm, tools=api_tools)

def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)

def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")

def transcribe_audio(file_path: str) -> str:
    """Transcribe audio using Groq Whisper model."""
    try:
        with open(file_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(file_path, file.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
            )
            return transcription.text
    except Exception as e:
        return f"Error during transcription: {e}"

def capture_audio_prompt() -> str:
    """Capture audio from the microphone and convert it to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for prompt...")
        try:
            audio = recognizer.listen(source)
            prompt_text = recognizer.recognize_google(audio)
            print("Captured Prompt:", prompt_text)
            return prompt_text
        except sr.UnknownValueError:
            print("Could not understand the audio.")
            return ""
        except sr.RequestError as e:
            print(f"Error with speech recognition service: {e}")
            return ""

async def main():
    """Main function to run the supervisor."""
    api_agent = await initialize_agents()
    scraper_agent = create_react_agent(name='scraper_agent', model=ChatNVIDIA(model='meta/llama-3.3-70b-instruct', nvidia_api_key=nvidia_api_key,temperature=0.2), tools=[sec_filing])
    retriever_agent = create_react_agent(name='retriever_agent', model=ChatNVIDIA(model='meta/llama-3.3-70b-instruct', nvidia_api_key=nvidia_api_key,temperature=0.3), tools=[search_documents])

    supervisor = create_supervisor(
        model=ChatNVIDIA(model='meta/llama-3.3-70b-instruct', nvidia_api_key=nvidia_api_key,temperature=0.2),
        agents=[api_agent, scraper_agent,retriever_agent],
        prompt=(
            "You are a supervisor managing three agents:\n"
            "- an API agent: ask about a stock ticker\n"
            "- a scraper agent: downloads SEC filings for a ticker \n"
            "- a retriever agent: retrieve filing documents from vectordb \n"
            "Use all the agents."
        ),
        add_handoff_back_messages=True,
        output_mode="last_message",
    ).compile()

    # Capture audio prompt
    # audio_prompt = capture_audio_prompt()
    # if not audio_prompt:
    #     print("No valid prompt captured. Exiting.")
    #     return
    async for chunk in supervisor.astream({"messages": [{"role": "user", "content": "find information about apple stocks "}]}):
        pretty_print_messages(chunk, last_message=True)

    final_message_history = chunk["supervisor"]["messages"]
    print("Final Message History:", final_message_history)
    print( final_message_history[-1].content)

if __name__ == "__main__":
    asyncio.run(main())

