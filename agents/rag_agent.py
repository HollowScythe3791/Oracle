from langchain_core.messages import AIMessage
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv

# Set your OpenAI API key securely
load_dotenv()

# Use OpenAI embeddings
Settings.embed_model = OpenAIEmbedding()

# Load documents from local data directory
documents = SimpleDirectoryReader("data/input").load_data()

# Setup Elasticsearch vector store connection
vector_store = ElasticsearchStore(
    es_url="https://localhost:9200",
    index_name="oracle",
    es_user="elastic",                # default superuser
    es_password="EbQylHp199nwQRxM388H",
    verify_certs=False                # disable SSL cert verification for local dev (not recommended for production)
)
# Create storage context using Elasticsearch vector store
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Build the vector index using documents and storage context
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Create a query engine from the index
query_engine = index.as_query_engine()

def rag_agent_node(state):
    user_query = state["messages"][-1].content if hasattr(state["messages"][-1], "content") else str(state["messages"][-1])
    response = query_engine.query(user_query)
    return {"messages": [AIMessage(content=str(response))]}

