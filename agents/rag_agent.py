import ssl
from langchain_core.messages import AIMessage
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.embeddings.openai import OpenAIEmbedding
from elasticsearch import AsyncElasticsearch
from dotenv import load_dotenv
import os

# Set your OpenAI API key securely
load_dotenv()
es_password = os.getenv("ELASTIC_PASSWORD")
ssl_path = os.getenv("CERT_PATH")

# Use OpenAI embeddings
Settings.embed_model = OpenAIEmbedding()

# Load documents from local data directory
documents = SimpleDirectoryReader("data/input").load_data()

# Create an SSLContext that trusts your Elasticsearch CA certificate
# Make sure you replace the path below with the absolute path of your CA cert (http_ca.crt)
ssl_context = ssl.create_default_context(cafile=ssl_path)

# Create an Elasticsearch client with TLS and authentication
es_client = AsyncElasticsearch(
    "https://localhost:9200",
    http_auth=("elastic", es_password),  # username, password from env
    ssl_context=ssl_context,  # This enables cert verification against your CA
    verify_certs=True,
)

# Pass the Elasticsearch client instance to ElasticsearchStore
vector_store = ElasticsearchStore(
    es_client=es_client,
    index_name="oracle",
    is_async=True
)

# Create storage context using Elasticsearch vector store
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Build the vector index using documents and storage context
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Create a query engine from the index
query_engine = index.as_query_engine()

def rag_agent_node(state):
    user_query = (
        state["messages"][-1].content
        if hasattr(state["messages"][-1], "content")
        else str(state["messages"][-1])
    )
    response = query_engine.query(user_query)
    return {"messages": [AIMessage(content=str(response))]}
