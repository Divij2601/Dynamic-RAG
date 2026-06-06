"""
Plane 2 - Generation quality evaluation.

Runs each query through the full Dynamic-RAG graph
and scores the generated answers for faithfulness,
groundedness, answer relevance, completeness and a
citation-accuracy proxy.

Faithfulness/groundedness are only meaningful for
answerable, evidence-grounded queries, so those
metrics are computed over answerable examples only.
"""

from statistics import mean

from sentence_transformers import (
    SentenceTransformer
)
from sklearn.metrics.pairwise import (
    cosine_similarity
)

from evaluation.pipeline_exec import execute_dataset
from src.config import settings
from src.observability.logger import app_logger


ABSTAIN_MARKER = "could not find sufficient evidence"


class GenerationEvaluator:
    """
    Plane 2: Generation evaluation.
    """

    _embedding_model = None

    def _model(self):
        if self._embedding_model is None:
            self._embedding_model = (
                SentenceTransformer(
                    settings.EMBEDDING_MODEL
                )
            )
        return self._embedding_model

    def evaluate(
        self,
        dataset_path: str,
        results=None
    ):
        """
        Compute Plane 2 metrics. If `results`
        (from execute_dataset) is provided, reuse
        them; otherwise run the pipeline.
        """

        if results is None:
            results = execute_dataset(dataset_path)

        faithfulness_scores = []
        groundedness_scores = []
        relevance_scores = []
        completeness_scores = []
        citation_scores = []

        evaluated = 0

        for record in results:

            example = record["example"]
            response = record["response"]

            # Generation quality applies to
            # answerable, evidence-grounded answers.
            if not example.get("answerable", True):
                continue

            if response is None:
                continue

            evaluated += 1

            answer = response.answer or ""

            faith = response.faithfulness_score
            if faith is not None:
                faithfulness_scores.append(faith)

            if response.grounded is not None:
                groundedness_scores.append(
                    float(response.grounded)
                )

            relevance_scores.append(
                self._answer_relevance(
                    example["query"],
                    answer
                )
            )

            completeness_scores.append(
                self._estimate_completeness(answer)
            )

            # Citation-accuracy proxy: agreement of
            # faithfulness and groundedness signals.
            cite = (
                (faith if faith is not None else 0.0)
                + (
                    float(response.grounded)
                    if response.grounded is not None
                    else 0.0
                )
            ) / 2
            citation_scores.append(cite)

        def _avg(xs):
            return round(mean(xs), 4) if xs else 0.0

        metrics = {
            "Faithfulness": _avg(faithfulness_scores),
            "Groundedness": _avg(groundedness_scores),
            "Answer Relevance": _avg(relevance_scores),
            "Completeness": _avg(completeness_scores),
            "Citation Accuracy": _avg(citation_scores),
            "Evaluated (answerable)": evaluated
        }

        app_logger.success(
            "Plane 2 generation evaluation complete"
        )

        return metrics

    def _answer_relevance(
        self,
        query: str,
        answer: str
    ) -> float:

        if not answer.strip():
            return 0.0

        model = self._model()

        q = model.encode(query)
        a = model.encode(answer)

        return float(
            cosine_similarity([q], [a])[0][0]
        )

    def _estimate_completeness(
        self,
        answer: str
    ) -> float:

        if ABSTAIN_MARKER in answer.lower():
            return 0.3

        words = len(answer.split())

        if words > 120:
            return 1.0
        if words > 60:
            return 0.8
        if words > 30:
            return 0.6
        return 0.4


generation_evaluator = GenerationEvaluator()
