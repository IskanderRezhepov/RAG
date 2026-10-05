import os

os.environ.pop("SSLKEYLOGFILE", None)

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric
)

from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel


def evaluate_rag_answer(
    question,
    answer,
    retrieved_sources
):
    """
    Evaluate a RAG answer using DeepEval
    and a local Ollama model.
    """

    retrieval_context = [
        source.page_content
        for source in retrieved_sources
    ]

    # Local evaluator model
    evaluator = OllamaModel(
        model="llama3",
        base_url="http://localhost:11434",
        temperature=0
    )

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=retrieval_context
    )

    relevancy_metric = AnswerRelevancyMetric(
        model=evaluator,
        threshold=0.5,
        include_reason=True,
        async_mode=False
    )

    faithfulness_metric = FaithfulnessMetric(
        model=evaluator,
        threshold=0.5,
        include_reason=True,
        async_mode=False
    )

    relevancy_metric.measure(test_case)
    faithfulness_metric.measure(test_case)

    return {
        "answer_relevancy": relevancy_metric.score,
        "answer_relevancy_reason": relevancy_metric.reason,

        "faithfulness": faithfulness_metric.score,
        "faithfulness_reason": faithfulness_metric.reason
    }