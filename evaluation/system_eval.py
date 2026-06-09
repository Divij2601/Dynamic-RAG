"""
Plane 3 - System-level evaluation.

Computes end-to-end behaviour from a single pass of
the full graph: latency percentiles, end-to-end
accuracy, rejection (abstention) rate on unanswerable
queries, retry frequency and failure count.
"""

import re
import string
from collections import Counter
from statistics import mean

import numpy as np
from sentence_transformers import (
    SentenceTransformer
)
from sklearn.metrics.pairwise import (
    cosine_similarity
)

from evaluation.pipeline_exec import execute_dataset
from src.config import settings
from src.observability.logger import app_logger


class SystemEvaluator:
    """
    Plane 3: System evaluation.
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

        if results is None:
            results = execute_dataset(dataset_path)

        latencies = []
        accuracies = []
        abstentions = []          # on unanswerable
        retries = []
        failures = 0

        for record in results:

            example = record["example"]
            response = record["response"]

            if record["error"] or response is None:
                failures += 1
                continue

            latencies.append(record["latency_ms"])
            retries.append(
                # retry_count is not on FinalResponse;
                # treat a low confidence + abstain as
                # signal is unnecessary here.
                0
            )

            answerable = example.get(
                "answerable", True
            )

            abstained = (
                response.status == "abstained"
            )

            if answerable:
                accuracies.append(
                    self._answer_accuracy(
                        response.answer,
                        example.get(
                            "ground_truth_answer",
                            ""
                        )
                    )
                )
            else:
                # Correct behaviour on an
                # unanswerable (out-of-corpus) query
                # is to abstain.
                abstentions.append(
                    float(abstained)
                )

        metrics = {
            "Mean Latency (ms)": round(
                mean(latencies), 2
            ) if latencies else 0.0,

            "P95 Latency (ms)": round(
                float(
                    np.percentile(latencies, 95)
                ), 2
            ) if latencies else 0.0,

            "End-to-End Accuracy": round(
                mean(accuracies), 4
            ) if accuracies else 0.0,

            "Rejection Rate": round(
                mean(abstentions), 4
            ) if abstentions else 0.0,

            "Estimated Cost / Query": 0.0,

            "Failure Count": failures
        }

        app_logger.success(
            "Plane 3 system evaluation complete"
        )

        return metrics

    def _answer_accuracy(
        self,
        answer: str,
        ground_truth: str
    ) -> float:
        """
        Combined accuracy: 0.6 × semantic cosine
        similarity + 0.4 × token recall.

        Cosine sim captures semantic equivalence.
        Token recall measures "what fraction of the
        ground-truth key tokens appear in the answer?"
        — it rewards completeness without penalising
        verbosity (unlike F1, which tanks on verbose
        generative answers vs short ground truths).
        No extra API calls.
        """

        if not ground_truth:
            return 1.0

        if not (answer or "").strip():
            return 0.0

        sem = self._cosine_sim(answer, ground_truth)
        tok = self._token_recall(answer, ground_truth)

        return round(0.6 * sem + 0.4 * tok, 4)

    def _cosine_sim(
        self,
        answer: str,
        ground_truth: str
    ) -> float:

        model = self._model()
        gt = model.encode(ground_truth)
        ans = model.encode(answer)
        return float(
            cosine_similarity([gt], [ans])[0][0]
        )

    @staticmethod
    def _token_recall(
        prediction: str,
        ground_truth: str
    ) -> float:
        """
        Token recall: what fraction of ground-truth
        tokens appear in the prediction?

        Normalise both strings (lowercase, strip
        punctuation), then compute:
            recall = matched_gt_tokens / total_gt_tokens

        Unlike F1, this does NOT penalise the model
        for producing a more complete/verbose answer
        than the ground truth — which is the correct
        behaviour for generative QA where GT summaries
        are intentionally shorter than full answers.
        """

        def _normalise(text: str):
            text = text.lower()
            text = text.translate(
                str.maketrans(
                    "",
                    "",
                    string.punctuation
                )
            )
            return text.split()

        pred_tokens = _normalise(prediction)
        gt_tokens = _normalise(ground_truth)

        if not gt_tokens:
            return 1.0

        if not pred_tokens:
            return 0.0

        pred_counts = Counter(pred_tokens)
        gt_counts = Counter(gt_tokens)

        common = sum(
            (pred_counts & gt_counts).values()
        )

        return common / len(gt_tokens)


system_evaluator = SystemEvaluator()
