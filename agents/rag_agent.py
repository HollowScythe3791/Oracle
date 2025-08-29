import os
from pathlib import Path
import ssl
from dotenv import load_dotenv
import asyncio

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.embeddings.openai import OpenAIEmbedding
from elasticsearch import AsyncElasticsearch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# ------------------------------------------------------------------------------
# Environment & Globals
# ------------------------------------------------------------------------------
load_dotenv()
ES_PASSWORD = os.getenv("ELASTIC_PASSWORD")
SSL_PATH = os.getenv("CERT_PATH")
DATA_DIR = "data"
ES_INDEX = "oracle"

# ------------------------------------------------------------------------------
# Global llama_index Settings config (embeddings only)
# ------------------------------------------------------------------------------
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# ------------------------------------------------------------------------------
# SSL & Elasticsearch Setup
# ------------------------------------------------------------------------------
ssl_context = ssl.create_default_context(cafile=SSL_PATH)
es_client = AsyncElasticsearch(
    "https://localhost:9200",
    http_auth=("elastic", ES_PASSWORD), ssl_context=ssl_context,
    verify_certs=True,
)

vector_store = ElasticsearchStore(
    es_client=es_client,
    index_name=ES_INDEX,
    is_async=True,
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# ------------------------------------------------------------------------------
# LLM Initialization (LangChain ChatOpenAI)
# ------------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# ------------------------------------------------------------------------------
# Document Ingestion
# ------------------------------------------------------------------------------
async def ingest_documents():
    data_path = Path(DATA_DIR)
    if not any(data_path.rglob("*")):
        print("No documents found to ingest.")
        return None

    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    if not documents:
        print("No documents loaded from SimpleDirectoryReader.")
        return None

    print(f"Ingesting {len(documents)} documents into vector index...")
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    # Delete ingested files
    for filepath in data_path.rglob("*"):
        if filepath.is_file():
            try:
                os.remove(filepath)
                print(f"Deleted ingested file: {filepath}")
            except Exception as e:
                print(f"Failed to delete {filepath}: {e}")

    return index

# ------------------------------------------------------------------------------
# Load Existing Index
# ------------------------------------------------------------------------------
async def load_existing_index():
    try:
        # Just wrap the existing vector store into a VectorStoreIndex
        index = VectorStoreIndex(storage_context=storage_context)
        print("Loaded vector store index interface from storage context.")
        return index
    except Exception as e:
        print(f"Could not load vector store index from storage context: {e}")
        return None

# ------------------------------------------------------------------------------
# Direct Elasticsearch Query
# ------------------------------------------------------------------------------
async def query_elasticsearch_direct(query: str):
    try:
        response = await es_client.search(
            index=ES_INDEX,
            body={
                "query": {
                    "match": {
                        "content": query  # Adjust to your doc field
                    }
                },
                "size": 5,
            }
        )
        hits = response["hits"]["hits"]
        return [hit["_source"].get("content", "") for hit in hits if "content" in hit["_source"]]
    except Exception as e:
        print(f"Error querying Elasticsearch directly: {e}")
        return []

# ------------------------------------------------------------------------------
# Summarize/Answer with ChatOpenAI
# ------------------------------------------------------------------------------
def answer_with_llm(query: str, retrieved_docs: list[str]) -> str:
    if not retrieved_docs:
        return "No relevant text was retrieved from Elasticsearch."

    context = "\n\n".join(retrieved_docs)
    prompt = (
        f"You have the following context:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        "Using ONLY the context above if relevant, provide your best possible answer."
    )

    messages = [
        SystemMessage(content="You are a helpful assistant that answers using the provided context."),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    return response.content

# ------------------------------------------------------------------------------
# RAG Agent Node (Async)
# ------------------------------------------------------------------------------
async def rag_agent_node(state):
    user_query = (
        state["messages"][-1].content
        if hasattr(state["messages"][-1], "content")
        else str(state["messages"][-1])
    )

    # Try ingestion first
    index = await ingest_documents()
    if index is None:
        index = await load_existing_index()

    # If we have an index, use it via llama_index query engine
    if index:
        query_engine = index.as_query_engine()
        response = query_engine.query(user_query)
        return {"messages": [AIMessage(content=str(response))]}

    # Fallback: direct ES query + LLM
    es_results = await query_elasticsearch_direct(user_query)
    answer = answer_with_llm(user_query, es_results)
    return {"messages": [AIMessage(content=answer)]}

