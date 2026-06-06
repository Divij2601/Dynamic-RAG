import json
import time
import numpy as np
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

from src.memory.retriever import (
    memory_retriever
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

from src.generation.response_builder import (
    response_builder
)

from src.observability.logger import (
    app_logger
)


class SystemEvaluator:
    """
    Plane 3:
    System evaluation
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

        latencies = []
        accuracies = []
        abstentions = []
        failures = []

        for example in dataset:

            start = (
                time.perf_counter()
            )

            try:

                query = (
                    example["query"]
                )

                ground_truth = (
                    example[
                        "ground_truth_answer"
                    ]
                )

                answerable = (
                    example[
                        "answerable"
                    ]
                )

                planner_output = (
                    query_planner
                    .plan(query)
                )

                state = QueryState(
                    request_id="eval",

                    session_id=
                    "eval_session",

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

                final_response = (
                    response_builder
                    .build(
                        answer=answer,

                        route=(
                            state
                            .selected_route
                        ),

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
                        ),

                        verification=(
                            verification
                        )
                    )
                )

                latency = (
                    time.perf_counter()
                    - start
                ) * 1000

                latencies.append(
                    latency
                )

                accuracy = (
                    self
                    ._answer_accuracy(
                        answer,
                        ground_truth
                    )
                )

                accuracies.append(
                    accuracy
                )

                abstained = (
                    "insufficient evidence"
                    in answer.lower()
                )

                if not answerable:

                    abstentions.append(
                        float(
                            abstained
                        )
                    )

            except Exception as e:

                failures.append(
                    str(e)
                )

        metrics = {

            "Mean Latency (ms)":
            round(
                mean(latencies),
                2
            ),

            "P95 Latency (ms)":
            round(
                np.percentile(
                    latencies,
                    95
                ),
                2
            ),

            "End-to-End Accuracy":
            round(
                mean(
                    accuracies
                ),
                4
            ),

            "Rejection Rate":
            round(
                mean(
                    abstentions
                )
                if abstentions
                else 1.0,
                4
            ),

            "Estimated Cost / Query":
            0.0,

            "Failure Count":
            len(failures)
        }

        app_logger.success(
            "Plane 3 system "
            "evaluation complete"
        )

        return metrics

    def _answer_accuracy(
        self,
        answer: str,
        ground_truth: str
    ) -> float:
        """
        End-to-end answer similarity
        """

        if not ground_truth:
            return 1.0

        gt_emb = (
            self.embedding_model
            .encode(
                ground_truth
            )
        )

        ans_emb = (
            self.embedding_model
            .encode(answer)
        )

        similarity = (
            cosine_similarity(
                [gt_emb],
                [ans_emb]
            )[0][0]
        )

        return float(similarity)


system_evaluator = (
    SystemEvaluator()
)