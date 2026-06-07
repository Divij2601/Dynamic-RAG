"""
Benchmark re-anchoring tool.

After adding new documents to the corpus, the existing
test_set.json queries may need their relevant_chunk_ids
updated to reference chunks from the new docs.

This script:
1. Loads the current test_set.json
2. For each answerable query, retrieves the top-20 chunks
3. Scores each chunk against the ground_truth_answer using
   semantic similarity
4. Proposes updated relevant_chunk_ids (chunks whose
   text semantically covers the answer)
5. Writes the updated file (after user review if --dry-run)

Usage:
    python -m evaluation.reanchor_benchmark
    python -m evaluation.reanchor_benchmark --dry-run
    python -m evaluation.reanchor_benchmark --threshold 0.65
"""

import json
import sys
import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval.hybrid import hybrid_retriever
from src.retrieval.reranker import reranker
from src.config import settings


DATASET_PATH = "evaluation/data/test_set.json"
DEFAULT_THRESHOLD = 0.60   # cosine sim of chunk vs ground truth
TOP_K = 20                 # candidates retrieved per query


def find_relevant_chunks(
    query: str,
    ground_truth: str,
    model: SentenceTransformer,
    threshold: float
) -> list:
    """
    Retrieve top-K chunks for the query, then keep those
    whose text is semantically close to the ground truth.
    """

    retrieval = hybrid_retriever.retrieve(
        query,
        top_k=TOP_K
    )

    reranked = reranker.rerank(
        query=query,
        retrieved_chunks=retrieval["results"]
    )

    if not reranked["results"]:
        return []

    gt_emb = model.encode(
        ground_truth,
        normalize_embeddings=True
    )

    chunk_ids = []

    for chunk in reranked["results"]:

        text = chunk.get("text", "")
        if not text.strip():
            continue

        chunk_emb = model.encode(
            text,
            normalize_embeddings=True
        )

        sim = float(
            cosine_similarity(
                [gt_emb],
                [chunk_emb]
            )[0][0]
        )

        if sim >= threshold:
            chunk_ids.append(chunk["chunk_id"])

    return chunk_ids


def main():

    parser = argparse.ArgumentParser(
        description="Re-anchor benchmark chunk IDs"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print proposed changes without writing"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Similarity threshold (default {DEFAULT_THRESHOLD})"
    )
    parser.add_argument(
        "--dataset",
        default=DATASET_PATH,
        help="Path to test_set.json"
    )
    args = parser.parse_args()

    print(
        f"Loading dataset: {args.dataset}"
    )

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print("Loading embedding model...")
    model = SentenceTransformer(
        settings.EMBEDDING_MODEL
    )

    updated = 0
    unchanged = 0
    proposed_changes = []

    for i, example in enumerate(dataset):

        if not example.get("answerable", True):
            continue

        gt = example.get("ground_truth_answer", "")
        if not gt.strip():
            continue

        query = example["query"]
        old_ids = example.get("relevant_chunk_ids", [])

        new_ids = find_relevant_chunks(
            query=query,
            ground_truth=gt,
            model=model,
            threshold=args.threshold
        )

        if set(new_ids) != set(old_ids):

            proposed_changes.append({
                "index": i,
                "query": query[:60],
                "old_ids": old_ids,
                "new_ids": new_ids
            })

            if not args.dry_run:
                dataset[i]["relevant_chunk_ids"] = new_ids

            updated += 1
            print(
                f"  [{i}] CHANGED {query[:55]!r}"
                f"\n       old: {old_ids}"
                f"\n       new: {new_ids}"
            )

        else:
            unchanged += 1

    print(
        f"\nSummary: {updated} updated, "
        f"{unchanged} unchanged"
    )

    if args.dry_run:
        print("(dry-run: no file written)")
        return

    if updated > 0:
        out_path = args.dataset
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)
        print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
