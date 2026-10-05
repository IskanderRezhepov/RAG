import os

os.environ.pop("SSLKEYLOGFILE", None)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma


def build_vector_store(pdf_path):
    """
    Loads a PDF, splits it into chunks,
    creates embeddings, and stores them in ChromaDB.
    """

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_store, len(documents), len(chunks)


def answer_question(vector_store, question):
    """
    Retrieves relevant chunks and sends them to Llama 3.
    """

    results = vector_store.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        result.page_content
        for result in results
    )

    llm = ChatOllama(
        model="llama3",
        temperature=0
    )

    prompt = f"""
You are a question-answering assistant.

Use ONLY the context taken from the uploaded PDF.

If the answer cannot be found in the context, say:

"The answer is not available in the document."

Do not invent information.

CONTEXT:

{context}

QUESTION:

{question}

ANSWER:
"""

    response = llm.invoke(prompt)

    return response.content, results