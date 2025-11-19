"""Experiment command - run SimRAG experiments"""
from pathlib import Path
from typing import Optional

import typer

from ..utils import check_venv


def experiment(
    command: str = typer.Argument(..., help="Experiment command: run, baseline, simrag, or compare"),
    documents: Optional[str] = typer.Option(None, "--documents", "-d", help="Path to documents folder"),
    test_data: bool = typer.Option(False, "--test-data", help="Use test data for Stage 1 instead of Alpaca"),
    skip_baseline: bool = typer.Option(False, "--skip-baseline", help="Skip baseline experiment"),
    skip_simrag: bool = typer.Option(False, "--skip-simrag", help="Skip SimRAG training"),
    baseline_file: Optional[str] = typer.Option(None, "--baseline-file", help="Path to existing baseline results"),
    simrag_file: Optional[str] = typer.Option(None, "--simrag-file", help="Path to existing SimRAG results"),
) -> None:
    """Run SimRAG experiments"""
    if not check_venv():
        raise typer.Exit(1)

    if command == "run":
        _run_full_experiment(documents, test_data, skip_baseline, skip_simrag, baseline_file, simrag_file)
    elif command == "baseline":
        _run_baseline(documents)
    elif command == "simrag":
        _run_simrag(documents, test_data)
    elif command == "compare":
        _run_compare(baseline_file, simrag_file)
    else:
        typer.echo(f"Unknown experiment command: {command}", err=True)
        typer.echo("Available commands: run, baseline, simrag, compare", err=True)
        raise typer.Exit(1)


def _run_full_experiment(
    documents: Optional[str],
    test_data: bool,
    skip_baseline: bool,
    skip_simrag: bool,
    baseline_file: Optional[str],
    simrag_file: Optional[str],
) -> None:
    """Run complete experiment pipeline"""
    try:
        from simrag_reproduction.experiments.run_experiment import run_complete_experiment
        from simrag_reproduction.logging_config import setup_logging

        setup_logging()

        documents_folder = documents or "../HTML_DOCS"
        run_complete_experiment(
            documents_folder=documents_folder,
            use_real_datasets=not test_data,
            skip_baseline=skip_baseline,
            skip_simrag=skip_simrag,
            baseline_file=baseline_file,
            simrag_file=simrag_file,
        )
        raise typer.Exit(0)
    except Exception as e:
        typer.echo(f"Experiment failed: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


def _run_baseline(documents: Optional[str]) -> None:
    """Run baseline experiment"""
    try:
        from simrag_reproduction.experiments.baseline.run_baseline import run_baseline_test
        from simrag_reproduction.logging_config import setup_logging

        setup_logging()

        documents_folder = documents or "../HTML_DOCS"
        run_baseline_test(
            documents_folder=documents_folder,
            output_file="baseline_results.json",
            use_timestamp=True,
        )
        raise typer.Exit(0)
    except Exception as e:
        typer.echo(f"Baseline experiment failed: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


def _run_simrag(documents: Optional[str], test_data: bool) -> None:
    """Run SimRAG pipeline"""
    try:
        from simrag_reproduction.experiments.simrag.run_full_pipeline import run_full_pipeline
        from simrag_reproduction.logging_config import setup_logging

        setup_logging()

        documents_folder = documents or "../HTML_DOCS"
        run_full_pipeline(
            documents_folder=documents_folder,
            use_real_datasets=not test_data,
            output_file="full_pipeline_results.json",
            use_timestamp=True,
        )
        raise typer.Exit(0)
    except Exception as e:
        typer.echo(f"SimRAG pipeline failed: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


def _run_compare(baseline_file: Optional[str], simrag_file: Optional[str]) -> None:
    """Compare experiment results"""
    try:
        from simrag_reproduction.experiments.comparison.compare_results import compare_results
        from simrag_reproduction.logging_config import setup_logging
        from pathlib import Path

        setup_logging()

        results_dir = Path("simrag_reproduction/experiments")
        if not baseline_file:
            baseline_file = str(results_dir / "baseline" / "results" / "baseline_results.json")
        if not simrag_file:
            simrag_file = str(results_dir / "simrag" / "results" / "full_pipeline_results.json")

        comparison = compare_results(
            baseline_file=baseline_file,
            simrag_file=simrag_file,
            output_file="comparison_results.json",
        )

        typer.echo(f"\nImprovement: {comparison['improvement']['context_score_improvement_percent']:+.1f}%")
        raise typer.Exit(0)
    except Exception as e:
        typer.echo(f"Comparison failed: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)

