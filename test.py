from src.retrieval.hybrid import (
    hybrid_retriever
)

from src.retrieval.reranker import (
    reranker
)


query = (
    "What is PTOC50?"
)

hybrid_results = (
    hybrid_retriever
    .retrieve(query)
)

reranked = (
    reranker
    .rerank(
        query=query,
        retrieved_chunks=(
            hybrid_results[
                "results"
            ]
        )
    )
)

for i, chunk in enumerate(
    reranked["results"]
):

    print("\n")
    print(
        f"Rank {i+1}"
    )

    print(
        f"Rerank Score: "
        f"{chunk['rerank_score']}"
    )

    print(
        chunk["text"][:300]
    )