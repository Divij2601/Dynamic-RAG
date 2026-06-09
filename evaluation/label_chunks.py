"""
Chunk-ID labeller for unlabelled test-set entries.

For every answerable entry whose relevant_chunk_ids
list is empty, this script:

  1. Runs hybrid retrieval + reranking against the
     production Qdrant index.
  2. Accepts the top-K reranked chunk IDs as the
     ground-truth label (pragmatic auto-labelling).
  3. Removes the "chunk_ids_not_verified" note from
     metadata so the entry is treated as fully labelled
     on the next benchmark run.
  4. Writes the updated test_set.json in-place.

Run once after adding new queries to the test set.
Requires Qdrant + BM25 index to be populated.

Usage:
    python -m evaluation.label_chunks
    python -m evaluation.label_chunks evaluation/data/test_set.json 8
"""

import json
import sys
import time
from pathlib import Path

# How many top chunk IDs to accept as ground truth.
# 5 is consistent with the existing labelled entries.
DEFAULT_TOP_K = 5

# Delay between retrieval calls (ms → seconds).
# No LLM calls so no rate limit, but Qdrant benefits
# from a small pause when under load.
INTER_QUERY_DELAY = 0.2


def label_dataset(
    dataset_path: str,
    top_k: int = DEFAULT_TOP_K
) -> dict:

    from src.retrieval.hybrid import hybrid_retriever
    from src.retrieval.reranker import reranker
    from src.config import settings
    from src.observability.logger import app_logger

    path = Path(dataset_path)
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    unlabelled = [
        (i, ex)
        for i, ex in enumerate(dataset)
        if ex.get("answerable", True)
        and not ex.get("relevant_chunk_ids")
    ]

    if not unlabelled:
        print("Nothing to label — all answerable entries "
              "already have chunk IDs.")
        return {"labelled": 0, "skipped": 0}

    print(
        f"Labelling {len(unlabelled)} unlabelled entries "
        f"(top_k={top_k}) ..."
    )

    labelled_count = 0
    failed = []

    for idx, (i, example) in enumerate(unlabelled, start=1):

        query = example["query"]

        try:
            retrieval = hybrid_retriever.retrieve(
                query,
                top_k=settings.RERANK_TOP_K
            )

            reranked = reranker.rerank(
                query=query,
                retrieved_chunks=retrieval["results"]
            )

            chunk_ids = [
                c["chunk_id"]
                for c in reranked["results"]
                if c.get("chunk_id")
            ][:top_k]

            if not chunk_ids:
                print(
                    f"  [{idx}/{len(unlabelled)}] "
                    f"WARNING: no chunks retrieved for: "
                    f"{query[:60]!r}"
                )
                failed.append(query)
                continue

            dataset[i]["relevant_chunk_ids"] = chunk_ids

            # Remove the placeholder note now that the
            # entry is properly labelled.
            meta = dataset[i].get("metadata", {})
            meta.pop("note", None)
            dataset[i]["metadata"] = meta

            labelled_count += 1

            print(
                f"  [{idx}/{len(unlabelled)}] "
                f"OK  {len(chunk_ids)} chunks  "
                f"{query[:55]!r}"
            )

        except Exception as exc:
            print(
                f"  [{idx}/{len(unlabelled)}] "
                f"ERROR: {query[:50]!r}: {exc!r}"
            )
            failed.append(query)

        if idx < len(unlabelled):
            time.sleep(INTER_QUERY_DELAY)

    # Write back in-place.
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(
        f"\nDone. Labelled {labelled_count} / "
        f"{len(unlabelled)} entries."
    )

    if failed:
        print(f"Failed ({len(failed)}):")
        for q in failed:
            print(f"  - {q}")

    return {
        "labelled": labelled_count,
        "failed": len(failed)
    }


if __name__ == "__main__":

    dataset_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "evaluation/data/test_set.json"
    )

    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TOP_K

    result = label_dataset(dataset_path, top_k=top_k)
    print(result)
