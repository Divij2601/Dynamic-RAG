import json
from statistics import mean

from sentence_transformers import (
    SentenceTransformer
)
from sklearn.metrics.pairwise import (
    cosine_similarity
)

from src.graph.state import (
    QueryState
)

from src.planner.planner import (
    query_planner
)

from src.planner.router import (
    query_router
)

from src.generation.prompt_builder import (
    prompt_builder
)

from src.generation.generator import (
    response_generator
)

from src.generation.verifier import (
    faithfulness_verifier
)

from src.observability.logger import (
    app_logger
)


class GenerationEvaluator:
    """
    Plane 2:
    Generation evaluation
    """

    def __init__(self):

        self.embedding_model = (
            SentenceTransformer(
                "BAAI/bge-small-en-v1.5"
            )
        )

    def evaluate(
        self,
        dataset_path: str
    ):

        with open(
            dataset_path,
            "r",
            encoding="utf-8"
        ) as f:

            dataset = json.load(f)

        faithfulness_scores = []
        groundedness_scores = []
        relevance_scores = []
        completeness_scores = []
        citation_scores = []

        for example in dataset:

            query = (
                example["query"]
            )

            session_id = (
                "eval_session"
            )

            planner_output = (
                query_planner
                .plan(query)
            )

            state = QueryState(
                request_id="eval",

                session_id=
                session_id,

                query_text=query,

                planner_output=
                planner_output
            )

            state = (
                query_router
                .execute(state)
            )

            prompt = (
                prompt_builder
                .build_prompt(
                    query=query,

                    internal_evidence=(
                        getattr(
                            state,
                            "internal_evidence",
                            []
                        )
                    ),

                    web_evidence=(
                        getattr(
                            state,
                            "web_evidence",
                            []
                        )
                    ),

                    memory_context=(
                        getattr(
                            state,
                            "memory_context",
                            ""
                        )
                    )
                )
            )

            answer = (
                response_generator
                .generate(prompt)
            )

            verification = (
                faithfulness_verifier
                .verify(
                    query=query,

                    answer=answer,

                    evidence_items=(
                        getattr(
                            state,
                            "internal_evidence",
                            []
                        )
                        +
                        getattr(
                            state,
                            "web_evidence",
                            []
                        )
                    )
                )
            )

            # Faithfulness
            faithfulness = (
                verification.get(
                    "faithfulness_score",
                    0.0
                )
            )

            faithfulness_scores.append(
                faithfulness
            )

            # Groundedness
            grounded = (
                verification.get(
                    "grounded",
                    False
                )
            )

            groundedness_scores.append(
                float(grounded)
            )

            # Answer relevance
            relevance = (
                self._answer_relevance(
                    query,
                    answer
                )
            )

            relevance_scores.append(
                relevance
            )

            # Completeness
            completeness = (
                self._estimate_completeness(
                    answer
                )
            )

            completeness_scores.append(
                completeness
            )

            # Citation accuracy proxy
            citation_scores.append(
                (
                    faithfulness
                    +
                    float(grounded)
                ) / 2
            )

        metrics = {

            "Faithfulness":
            round(
                mean(
                    faithfulness_scores
                ),
                4
            ),

            "Groundedness":
            round(
                mean(
                    groundedness_scores
                ),
                4
            ),

            "Answer Relevance":
            round(
                mean(
                    relevance_scores
                ),
                4
            ),

            "Completeness":
            round(
                mean(
                    completeness_scores
                ),
                4
            ),

            "Citation Accuracy":
            round(
                mean(
                    citation_scores
                ),
                4
            )
        }

        app_logger.success(
            "Plane 2 generation "
            "evaluation complete"
        )

        return metrics

    def _answer_relevance(
        self,
        query: str,
        answer: str
    ) -> float:
        """
        Query-answer relevance
        """

        query_emb = (
            self.embedding_model
            .encode(query)
        )

        answer_emb = (
            self.embedding_model
            .encode(answer)
        )

        similarity = (
            cosine_similarity(
                [query_emb],
                [answer_emb]
            )[0][0]
        )

        return float(similarity)

    def _estimate_completeness(
        self,
        answer: str
    ) -> float:
        """
        Simple completeness
        proxy metric
        """

        if (
            "insufficient evidence"
            in answer.lower()
        ):
            return 0.3

        word_count = len(
            answer.split()
        )

        if word_count > 120:
            return 1.0

        elif word_count > 60:
            return 0.8

        elif word_count > 30:
            return 0.6

        return 0.4


generation_evaluator = (
    GenerationEvaluator()
)