"""
Unified benchmark runner for Dynamic-RAG.

Runs all evaluation planes and saves a JSON report:

  - Plane 1 (retrieval): independent, no LLM calls.
  - Gate C  (routing):   planner route accuracy.
  - Pipeline pass:       each query run once through
                         the graph (shared by Plane 2/3).
  - Plane 2 (generation) and Plane 3 (system): computed
                         from the shared pipeline pass.
"""

import json
from datetime import datetime
from pathlib import Path

from evaluation.retrieval_eval import (
    retrieval_evaluator
)
from evaluation.route_eval import (
    route_evaluator
)
from evaluation.pipeline_exec import (
    execute_dataset
)
from evaluation.generation_eval import (
    generation_evaluator
)
from evaluation.system_eval import (
    system_evaluator
)

from src.observability.logger import app_logger


class BenchmarkRunner:
    """
    Unified benchmark runner.
    """

    def run(
        self,
        dataset_path: str,
        experiment_name: str = "dynamic_rag"
    ):

        app_logger.info("Starting benchmark run")

        # Plane 1 - retrieval (no LLM)
        retrieval_metrics = (
            retrieval_evaluator.evaluate(
                dataset_path
            )
        )

        # Gate C - routing accuracy
        route_metrics = route_evaluator.evaluate(
            dataset_path
        )

        # Single full-pipeline pass shared by 2 & 3
        app_logger.info(
            "Executing pipeline pass for "
            "generation + system planes"
        )
        results = execute_dataset(dataset_path)

        # Plane 2 - generation
        generation_metrics = (
            generation_evaluator.evaluate(
                dataset_path,
                results=results
            )
        )

        # Plane 3 - system
        system_metrics = (
            system_evaluator.evaluate(
                dataset_path,
                results=results
            )
        )

        final_report = {
            "experiment_name": experiment_name,
            "timestamp": datetime.utcnow().isoformat(),
            "dataset": dataset_path,
            "plane_1_retrieval": retrieval_metrics,
            "gate_c_routing": route_metrics,
            "plane_2_generation": generation_metrics,
            "plane_3_system": system_metrics
        }

        self._save_report(
            final_report,
            experiment_name
        )

        app_logger.success("Benchmark complete")

        return final_report

    def _save_report(
        self,
        report: dict,
        experiment_name: str
    ):

        reports_dir = Path("evaluation/reports")
        reports_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        timestamp = datetime.utcnow().strftime(
            "%Y%m%d_%H%M%S"
        )

        report_path = reports_dir / (
            f"{experiment_name}_{timestamp}.json"
        )

        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(report, f, indent=4)

        app_logger.success(
            f"Saved benchmark report: {report_path}"
        )


benchmark_runner = BenchmarkRunner()


if __name__ == "__main__":

    import sys

    dataset = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "evaluation/data/test_set.json"
    )

    report = benchmark_runner.run(dataset)

    print(json.dumps(report, indent=2))
