import os

os.environ.pop("SSLKEYLOGFILE", None)

import tempfile
import streamlit as st

from rag_pipeline import (
    build_vector_store,
    answer_question
)

from evaluation import evaluate_rag_answer


st.set_page_config(
    page_title="Local PDF RAG Assistant",
    page_icon="📄",
    layout="wide"
)


st.title("📄 Local PDF RAG Assistant")

st.write(
    """
    Upload a PDF and ask questions about its content.

    This application uses:

    - LangChain
    - Llama 3
    - Ollama
    - ChromaDB
    - Nomic Embeddings
    - DeepEval
    """
)


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "answer" not in st.session_state:
    st.session_state.answer = None

if "sources" not in st.session_state:
    st.session_state.sources = None

if "question" not in st.session_state:
    st.session_state.question = None


# --------------------------------------------------
# PDF Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    if st.button("Process PDF"):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(
                uploaded_file.getvalue()
            )

            pdf_path = tmp_file.name

        with st.spinner(
            "Reading and indexing PDF..."
        ):

            vector_store, pages, chunks = (
                build_vector_store(
                    pdf_path
                )
            )

            st.session_state.vector_store = (
                vector_store
            )

        st.success(
            "PDF indexed successfully."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "PDF Pages",
                pages
            )

        with col2:
            st.metric(
                "Text Chunks",
                chunks
            )


# --------------------------------------------------
# Question section
# --------------------------------------------------

if st.session_state.vector_store is not None:

    st.divider()

    st.subheader(
        "Ask a Question"
    )

    question = st.text_input(
        "Enter your question about the PDF"
    )

    if st.button("Ask"):

        if question.strip():

            with st.spinner(
                "Searching document and generating answer..."
            ):

                answer, sources = (
                    answer_question(
                        st.session_state.vector_store,
                        question
                    )
                )

                st.session_state.question = (
                    question
                )

                st.session_state.answer = (
                    answer
                )

                st.session_state.sources = (
                    sources
                )

        else:

            st.warning(
                "Please enter a question."
            )


# --------------------------------------------------
# Answer
# --------------------------------------------------

if st.session_state.answer:

    st.divider()

    st.subheader(
        "Answer"
    )

    st.write(
        st.session_state.answer
    )


# --------------------------------------------------
# Retrieved sources
# --------------------------------------------------

if st.session_state.sources:

    st.subheader(
        "Retrieved Sources"
    )

    for i, source in enumerate(
        st.session_state.sources,
        start=1
    ):

        page = source.metadata.get(
            "page",
            None
        )

        if isinstance(page, int):
            page_display = page + 1
        else:
            page_display = "Unknown"

        with st.expander(
            f"Source {i} — Page {page_display}"
        ):

            st.write(
                source.page_content
            )


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

if (
    st.session_state.answer
    and st.session_state.sources
):

    st.divider()

    st.subheader(
        "RAG Evaluation"
    )

    st.write(
        """
        DeepEval evaluates whether the answer is
        relevant to the question and faithful to
        the retrieved PDF content.
        """
    )

    if st.button(
        "Evaluate Answer"
    ):

        with st.spinner(
            "Evaluating RAG response..."
        ):

            try:

                evaluation = (
                    evaluate_rag_answer(
                        st.session_state.question,
                        st.session_state.answer,
                        st.session_state.sources
                    )
                )

                col1, col2 = (
                    st.columns(2)
                )

                with col1:

                    st.metric(
                        "Answer Relevancy",
                        round(
                            evaluation[
                                "answer_relevancy"
                            ],
                            3
                        )
                    )

                with col2:

                    st.metric(
                        "Faithfulness",
                        round(
                            evaluation[
                                "faithfulness"
                            ],
                            3
                        )
                    )

                st.subheader(
                    "Evaluation Explanation"
                )

                st.write(
                    "**Answer Relevancy:**"
                )

                st.write(
                    evaluation[
                        "answer_relevancy_reason"
                    ]
                )

                st.write(
                    "**Faithfulness:**"
                )

                st.write(
                    evaluation[
                        "faithfulness_reason"
                    ]
                )

            except Exception as error:

                st.error(
                    "Evaluation failed."
                )

                st.code(
                    str(error)
                )

                st.info(
                    """
                    The chatbot can still work even
                    if DeepEval evaluation fails.
                    Local evaluator models sometimes
                    have difficulty producing the
                    structured output expected by
                    evaluation frameworks.
                    """
                )


# --------------------------------------------------
# Architecture
# --------------------------------------------------

st.divider()

with st.expander(
    "How the RAG system works"
):

    st.markdown(
        """
        **1. PDF Upload**

        ↓

        **2. Text Extraction**

        ↓

        **3. Text Chunking**

        ↓

        **4. Nomic Embeddings**

        ↓

        **5. ChromaDB Vector Store**

        ↓

        **6. Semantic Retrieval**

        ↓

        **7. Llama 3 Answer Generation**

        ↓

        **8. DeepEval Quality Evaluation**
        """
    )