import os
import uuid
import asyncio
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.schema.document import Document
from langchain_community.document_loaders.html import UnstructuredHTMLLoader
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langgraph.prebuilt import create_react_agent
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
# Load environment variables
load_dotenv()
grok_api_key = os.getenv("GROQ_API_KEY")
nvidia_api_key = os.getenv("NVIDIA_API_KEY")
url=os.getenv('QDRANT_URL')
api_key=os.getenv('QDRANT_API_KEY')
# LLMs and embeddings
groq_model = ChatGroq(model="gemma2-9b-it", api_key=grok_api_key)
nvidia_model = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", nvidia_api_key=nvidia_api_key)
embedding_model = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", nvidia_api_key=nvidia_api_key)
client=QdrantClient(
    url=url,
    api_key=api_key,
)

# MultiVectorRetriever setup
id_key = "doc_id"
retriever = MultiVectorRetriever(
    vectorstore=Chroma(collection_name="summaries", embedding_function=embedding_model),
    docstore=InMemoryStore(),
    id_key=id_key,
)
if client.collection_exists('ragaaiv0')==False:
    client.create_collection(
    collection_name="ragaaiv0",
    vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
)
vector_store = QdrantVectorStore(client=client, embedding=embedding_model, collection_name="ragaaiv0")

# Prompt template
prompt_text = """
You are responsible for concisely summarizing a financial table or text chunk:

{element}

Extract the key financial insights and present them in a clear summary.
Focus on metrics, trends, and important facts.
"""

@tool
async def create_summary(stock_name: str) -> str:
    """Creates summaries of files in folder stock_name and stores them in vector DB."""
    path = f'extracts/{stock_name}'
    summaries = []
    original_contents = []

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
                original_contents.append(content)

                prompt = prompt_text.format(element=content)
                response = await nvidia_model.ainvoke([HumanMessage(content=prompt)])
                summaries.append(response.content)

                print(f"Processed {folder}/{file}")

    # Store in retriever
    doc_ids = [str(uuid.uuid4()) for _ in summaries]
    summary_docs = [Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(summaries)]

    
    vector_store.add_documents(summary_docs)
    return f"Final Answer: Stored {len(summaries)} summaries from folder '{stock_name}' in vector database."

@tool
async def search_documents(query: str) -> str:
    """Search for relevant documents using the query."""
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3,})
    results = await retriever.ainvoke(query)
    content=''
    for doc in results:
        content+=doc.page_content
    return f"Final Answer: Here are the most relevant documents:\n\n{content}"


# Create the agent
agent = create_react_agent(model=groq_model, tools=[create_summary, search_documents])

# Run the agent (asynchronously)
async def main():
    user_message = "search for cashflow "
    response = await agent.ainvoke(
        {"messages": [HumanMessage(content=user_message)]},
        {"recursion_limit": 25}
    )
    print("Agent Final Output:\n", response.get("output", response))

# Run async main
if __name__ == "__main__":
    asyncio.run(main())
