import os
os.environ.pop("SSLKEYLOGFILE", None)

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

st.title("Local PDF Chat with RAG")
st.write("Ask questions about the PDF using Llama 3, Ollama and ChromaDB.")

pdf_path = "data/test.pdf"

@st.cache_resource
def create_vector_store():
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_store

vector_store = create_vector_store()

question = st.text_input("Ask a question about the PDF:")

if question:
    results = vector_store.similarity_search(question, k=3)

    context = "\n\n".join(
        result.page_content for result in results
    )

    llm = ChatOllama(
        model="llama3",
        temperature=0
    )

    prompt = f"""
    Answer the question using only the provided PDF context.

    Context:
    {context}

    Question:
    {question}

    Give a clear and concise answer.
    """

    response = llm.invoke(prompt)

    st.subheader("Answer")
    st.write(response.content)