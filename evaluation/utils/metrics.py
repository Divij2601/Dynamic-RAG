import math
from typing import List


def recall_at_k(
    retrieved: List[str],
    relevant: List[str]
) -> float:
    """
    Recall@K
    """

    if not relevant:
        return 1.0

    hits = len(
        set(retrieved)
        &
        set(relevant)
    )

    return hits / len(relevant)


def hit_rate(
    retrieved: List[str],
    relevant: List[str]
) -> float:
    """
    Hit rate
    """

    return float(
        any(
            chunk in relevant
            for chunk in retrieved
        )
    )


def reciprocal_rank(
    retrieved: List[str],
    relevant: List[str]
) -> float:
    """
    Reciprocal rank
    """

    for rank, chunk in enumerate(
        retrieved,
        start=1
    ):

        if chunk in relevant:
            return 1 / rank

    return 0.0


def ndcg_at_k(
    retrieved: List[str],
    relevant: List[str]
) -> float:
    """
    NDCG@K
    """

    dcg = 0

    for i, chunk in enumerate(
        retrieved,
        start=1
    ):

        rel = (
            1
            if chunk
            in relevant
            else 0
        )

        dcg += (
            rel
            / math.log2(i + 1)
        )

    ideal_hits = min(
        len(relevant),
        len(retrieved)
    )

    idcg = sum(
        1 / math.log2(i + 1)
        for i in range(
            1,
            ideal_hits + 1
        )
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


def context_precision(
    retrieved: List[str],
    relevant: List[str]
) -> float:
    """
    Context precision
    """

    if not retrieved:
        return 0.0

    hits = len(
        set(retrieved)
        &
        set(relevant)
    )

    return hits / len(retrieved)


def context_recall(
    retrieved: List[str],
    relevant: List[str]
) -> float:
    """
    Context recall
    """

    return recall_at_k(
        retrieved,
        relevant
    )