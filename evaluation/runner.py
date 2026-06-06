import json
from datetime import datetime
from pathlib import Path

from evaluation.retrieval_eval import (
    retrieval_evaluator
)

from evaluation.generation_eval import (
    generation_evaluator
)

from evaluation.system_eval import (
    system_evaluator
)

from src.observability.logger import (
    app_logger
)


class BenchmarkRunner:
    """
    Unified benchmark runner
    for Dynamic-RAG
    """

    def run(
        self,
        dataset_path: str,

        experiment_name:
        str = "dynamic_rag"
    ):
        """
        Run all 3 evaluation planes
        """

        app_logger.info(
            "Starting benchmark run"
        )

        retrieval_metrics = (
            retrieval_evaluator
            .evaluate(
                dataset_path
            )
        )

        generation_metrics = (
            generation_evaluator
            .evaluate(
                dataset_path
            )
        )

        system_metrics = (
            system_evaluator
            .evaluate(
                dataset_path
            )
        )

        final_report = {

            "experiment_name":
            experiment_name,

            "timestamp":
            datetime.utcnow()
            .isoformat(),

            "dataset":
            dataset_path,

            "plane_1_retrieval":
            retrieval_metrics,

            "plane_2_generation":
            generation_metrics,

            "plane_3_system":
            system_metrics
        }

        self._save_report(
            final_report,
            experiment_name
        )

        app_logger.success(
            "Benchmark complete"
        )

        return final_report

    def _save_report(
        self,
        report: dict,
        experiment_name: str
    ):
        """
        Save report
        """

        timestamp = (
            datetime.utcnow()
            .strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        report_path = Path(
            "evaluation/reports"
        ) / (
            f"{experiment_name}"
            f"_{timestamp}.json"
        )

        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                report,
                f,
                indent=4
            )

        app_logger.success(
            f"Saved benchmark "
            f"report: "
            f"{report_path}"
        )


benchmark_runner = (
    BenchmarkRunner()
)