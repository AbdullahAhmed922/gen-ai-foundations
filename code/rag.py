from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 2 Step Rag

# Index
chunk = RecursiveCharacterTextSplitter().split_documents(
    documents=TextLoader(file_path="doc.txt").load()
)
vs = InMemoryVectorStore.from_documents(
    documents=chunk,
    embedding=init_embeddings(model="nomic-embed-text:latest",model_provider="ollama")
)
llm = init_chat_model(model="qwen2.5:3b",model_provider="ollama")

query = input("Enter your query: ")

# Step 1: Retrieval
ctx=""

# Step 2: Generation