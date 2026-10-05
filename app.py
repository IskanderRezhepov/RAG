import os
os.environ.pop("SSLKEYLOGFILE", None)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

# 1. Load PDF
loader = PyPDFLoader("data/test.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages")

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

# 3. Create embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# 4. Store in ChromaDB
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("PDF stored in ChromaDB successfully")

# 5. Ask a question
question = "What are the examples of abnormal situations in the document?"

results = vector_store.similarity_search(question, k=3)

# 6. Combine retrieved chunks
context = "\n\n".join(result.page_content for result in results)

# 7. Connect to Llama 3
llm = ChatOllama(
    model="llama3",
    temperature=0
)

prompt = f"""
You are answering questions using only the PDF context below.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""

response = llm.invoke(prompt)

print("\nQUESTION:")
print(question)

print("\nANSWER:")
print(response.content)