"""
DeepEval test suite for the RAG pipeline.

Run with:   pytest evaluation.py -v

Requirements:
  - GOOGLE_API_KEY  : used by the RAG pipeline itself (ingestion + answering).
  - OPENAI_API_KEY  : used by the DeepEval judge model (JUDGE_MODEL below).
    DeepEval's metrics default to OpenAI models, so a Google key alone is not
    enough to score the results. Set JUDGE_MODEL to a model you have access to.

Note: this suite is Qdrant-only. The Neo4j knowledge graph was removed from
the retrieval path, so there is no graph driver here any more.
"""

import os

import pytest
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

import hybrid_search
import ingestion
import utils
import vector_db

utils.load_environment()

QDRANT_COLLECTION_NAME = "evaluation_collection"
JUDGE_MODEL = os.getenv("DEEPEVAL_JUDGE_MODEL", "gpt-4o-mini")

TEST_DATA = """
Dr. Evelyn Reed, a leading researcher at Innovate Dynamics, published a paper on Quantum RAG systems.
The paper, titled 'Quantum Leaps in AI', was co-authored by Dr. Ben Carter.
Innovate Dynamics is headquartered in Geneva. The project code for this research is QR-7.
"""


def _metrics():
    """Built lazily so that importing this module does not require a judge key."""
    return [
        AnswerRelevancyMetric(threshold=0.7, model=JUDGE_MODEL, include_reason=True),
        FaithfulnessMetric(threshold=0.7, model=JUDGE_MODEL, include_reason=True),
        ContextualRecallMetric(threshold=0.7, model=JUDGE_MODEL, include_reason=True),
    ]


@pytest.fixture(scope="module")
def rag_client(tmp_path_factory):
    """
    Ingests the known test document into an isolated Qdrant store.

    Uses a temporary directory so the evaluation never touches the real
    ./local_qdrant_db used by the app.
    """
    from qdrant_client import QdrantClient

    db_path = tmp_path_factory.mktemp("eval_qdrant")
    client = QdrantClient(path=str(db_path))
    vector_db.create_collection_if_not_exists(client, QDRANT_COLLECTION_NAME)

    doc = tmp_path_factory.mktemp("eval_docs") / "test_document.txt"
    doc.write_text(TEST_DATA, encoding="utf-8")

    ingestion.ingest_file(
        file_path=str(doc),
        collection_name=QDRANT_COLLECTION_NAME,
        qdrant_client=client,
    )

    yield client
    client.close()


def run_rag_pipeline(query: str, client):
    """Runs the full RAG pipeline, returning (answer, context)."""
    context = hybrid_search.hybrid_retrieval(
        query, QDRANT_COLLECTION_NAME, client
    )
    answer = hybrid_search.generate_response(query, context)
    return answer, context


CASES = [
    (
        "Who published the paper on Quantum RAG systems?",
        "Dr. Evelyn Reed published the paper on Quantum RAG systems.",
    ),
    (
        "Who works for Innovate Dynamics?",
        "Dr. Evelyn Reed works at Innovate Dynamics.",
    ),
    (
        "What is the project code for the Quantum RAG research?",
        "The project code is QR-7.",
    ),
    (
        "Where is the company that Dr. Ben Carter is associated with located?",
        "Innovate Dynamics, which Dr. Ben Carter is associated with, is in Geneva.",
    ),
]


@pytest.mark.parametrize("query,expected", CASES, ids=[c[0] for c in CASES])
def test_rag_system(rag_client, query, expected):
    """Scores one query against the relevancy/faithfulness/recall metrics."""
    answer, context = run_rag_pipeline(query, rag_client)

    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        expected_output=expected,
        retrieval_context=[context],
        context=[context],
    )

    evaluate(test_cases=[test_case], metrics=_metrics())
