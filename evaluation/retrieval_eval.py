import json
from statistics import mean

from evaluation.utils.metrics import (
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    hit_rate,
    context_precision,
    context_recall
)

from src.retrieval.hybrid import (
    hybrid_retriever
)

from src.retrieval.reranker import (
    reranker
)

from src.retrieval.evidence import (
    evidence_builder
)

from src.observability.logger import (
    app_logger
)


class RetrievalEvaluator:
    """
    Plane 1:
    Retrieval evaluation
    """

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

        recall_scores = []
        mrr_scores = []
        ndcg_scores = []
        hit_scores = []
        precision_scores = []
        context_recall_scores = []

        for example in dataset:

            query = (
                example["query"]
            )

            gold_chunks = (
                example[
                    "relevant_chunk_ids"
                ]
            )

            retrieval = (
                hybrid_retriever
                .retrieve(query)
            )

            reranked = (
                reranker
                .rerank(
                    query=query,

                    retrieved_chunks=(
                        retrieval[
                            "results"
                        ]
                    )
                )
            )

            evidence = (
                evidence_builder
                .build(
                    reranked[
                        "results"
                    ]
                )
            )

            retrieved_ids = [
                e.chunk_id
                for e in evidence
                if e.chunk_id
            ]

            recall_scores.append(
                recall_at_k(
                    retrieved_ids,
                    gold_chunks
                )
            )

            mrr_scores.append(
                reciprocal_rank(
                    retrieved_ids,
                    gold_chunks
                )
            )

            ndcg_scores.append(
                ndcg_at_k(
                    retrieved_ids,
                    gold_chunks
                )
            )

            hit_scores.append(
                hit_rate(
                    retrieved_ids,
                    gold_chunks
                )
            )

            precision_scores.append(
                context_precision(
                    retrieved_ids,
                    gold_chunks
                )
            )

            context_recall_scores.append(
                context_recall(
                    retrieved_ids,
                    gold_chunks
                )
            )

        metrics = {
            "Recall@K":
            round(
                mean(
                    recall_scores
                ),
                4
            ),

            "MRR":
            round(
                mean(
                    mrr_scores
                ),
                4
            ),

            "NDCG@K":
            round(
                mean(
                    ndcg_scores
                ),
                4
            ),

            "Hit Rate":
            round(
                mean(
                    hit_scores
                ),
                4
            ),

            "Context Precision":
            round(
                mean(
                    precision_scores
                ),
                4
            ),

            "Context Recall":
            round(
                mean(
                    context_recall_scores
                ),
                4
            )
        }

        app_logger.success(
            "Plane 1 retrieval "
            "evaluation complete"
        )

        return metrics


retrieval_evaluator = (
    RetrievalEvaluator()
)