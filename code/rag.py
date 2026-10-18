# from langchain.chat_models import init_chat_model
# from langchain.embeddings import init_embeddings
# from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# # 1. Indexing
# loader = TextLoader(file_path="doc.txt")
# documents = loader.load()
# text_splitter = RecursiveCharacterTextSplitter()
# chunks = text_splitter.split_documents(documents)

# vs = InMemoryVectorStore.from_documents(
#     documents=chunks,
#     embedding=init_embeddings(model="nomic-embed-text:latest", model_provider="ollama")
# )

# llm = init_chat_model(model="qwen2.5:3b", model_provider="ollama")

# query = input("Enter your query: ")

# # Step 1: Retrieval
# # Search for the most relevant documents based on the query
# docs = vs.similarity_search(query, k=3)
# ctx = "\n\n".join([doc.page_content for doc in docs])

# # Step 2: Generation
# # Create a prompt that includes the retrieved context
# prompt = f"Answer the question based only on the following context:\n\n{ctx}\n\nQuestion: {query}"
# response = llm.invoke(prompt)

# print("\nAnswer:")
# print(response.content)


# from langchain.chat_models import init_chat_model #LLM
# from langchain.embeddings import init_embeddings #Embedding model
# # from langchain_community.document_loaders import PyPDFLoader
# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.vectorstores import InMemoryVectorStore

# text = open(file="doc.txt",encoding="utf-8").read()

# doc = Document(page_content=text, metadata={"source": "doc.txt"})

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=600, chunk_overlap=100
# )

# chunks = splitter.split_documents(documents=[doc])

# len(chunks)

# embeddings = init_embeddings(
#     model="nomic-embed-text:latest",
#     provider = "ollama"
# )

# vs = InMemoryVectorStore.from_documents(
#     documents=chunks,
#     embedding=embeddings
# )

# llm = init_chat_model(
#     model="gemma4:31b-cloud",
#     model_provider="ollama"
# )

# query = "How refund are paid"

# texts = vs.similarity_search(query, k=3)

# ctx = "\n\n".join(d.page_content for d in texts)

# llm.invoke(input=f"Answer the question based on the context below:\nContext: {ctx}\n\nQuestion: {query}\n\nAnswer:").text
# prompt_template = [
#     {
#         "role": "system",
#         "content": "You are a helpful assistant that answers questions based on the context provided."
#     },
#     {
#         "role": "user",
#         "content": f"Answer the question based on the context below:\n"
#                    f"Context: {ctx}\n\nQuestion: {query}\n\nAnswer:"
#     }
# ]
# print(llm.invoke(input=prompt_template).text)


from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

# 1. Load and split the document
with open(file="doc.txt", encoding="utf-8") as f:
    text = f.read()

doc = Document(page_content=text, metadata={"source": "doc.txt"})

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600, chunk_overlap=100
)
chunks = splitter.split_documents(documents=[doc])

# 2. Create embeddings and vector store
embeddings = init_embeddings(
    model="nomic-embed-text:latest",
    provider="ollama"
)

vs = InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings
)

# 3. Initialize the LLM
llm = init_chat_model(
    model="gemma4:31b-cloud",
    model_provider="ollama"
)

# 4. Ask your question dynamically
query = input("Enter your question: ")

# 5. Retrieve relevant chunks and generate the answer
texts = vs.similarity_search(query, k=3)
ctx = "\n\n".join(d.page_content for d in texts)

prompt_template = [
    {
        "role": "system",
        "content": "You are a helpful assistant that answers questions based on the context provided."
    },
    {
        "role": "user",
        "content": f"Answer the question based on the context below:\n"
                   f"Context: {ctx}\n\nQuestion: {query}\n\nAnswer:"
    }
]

response = llm.invoke(input=prompt_template)
print(f"\nAnswer: {response.content.strip()}")

