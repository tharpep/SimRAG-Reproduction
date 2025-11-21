"""Experiment command - run SimRAG experiments"""
from pathlib import Path
from typing import Optional

import typer

from ..utils import check_venv


def experiment(
    command: str = typer.Argument(..., help="Experiment command: run, stage1, stage2, baseline, simrag, compare, export, or results"),
    documents: Optional[str] = typer.Option(None, "--documents", "-d", help="Path to documents folder"),
    test_data: bool = typer.Option(False, "--test-data", help="Use test data for Stage 1 instead of Alpaca"),
    baseline_file: Optional[str] = typer.Option(None, "--baseline-file", help="Path to existing baseline results"),
    simrag_file: Optional[str] = typer.Option(None, "--simrag-file", help="Path to existing SimRAG results"),
    stage1_model: Optional[str] = typer.Option(None, "--stage1-model", help="Path to Stage 1 model (for Stage 2 only)"),
) -> None:
    """Run SimRAG experiments"""
    if not check_venv():
        raise typer.Exit(1)

    if command == "run":
        _run_full_experiment(documents, test_data)
    elif command == "stage1":
        _run_stage1_only(test_data)
    elif command == "stage2":
        _run_stage2_only(documents, stage1_model)
    elif command == "baseline":
        _run_baseline(documents)
    elif command == "simrag":
        _run_simrag(documents, test_data)
    elif command == "compare":
        _run_compare(baseline_file, simrag_file)
    elif command == "export":
        _run_export_model()
    elif command == "results":
        _run_display_results()
    else:
        typer.echo(f"Unknown experiment command: {command}", err=True)
        typer.echo("Available commands: run, stage1, stage2, baseline, simrag, compare, export, results", err=True)
        typer.echo("\nNote: 'run' trains both stages. Use 'stage1' or 'stage2' to run individually.", err=True)
        typer.echo("Testing/comparison done in Colab notebook.", err=True)
        raise typer.Exit(1)


def _run_full_experiment(
    documents: Optional[str],
    test_data: bool,
) -> None:
    """Run SimRAG training pipeline (Stage 1 -> Stage 2)"""
    try:
        from ...experiments.run_experiment import run_complete_experiment
        from ...logging_config import setup_logging

        setup_logging()

        # Resolve path relative to project root when running via CLI
        if documents:
            documents_folder = documents
        else:
            # Default: data/documents relative to project root
            # CLI file is at: simrag_reproduction/cli/commands/experiment.py
            # Project root is 4 levels up: commands -> cli -> simrag_reproduction -> project_root
            cli_file = Path(__file__)
            project_root = cli_file.parent.parent.parent.parent
            documents_folder = str(project_root / "data" / "documents")
        run_complete_experiment(
            documents_folder=documents_folder,
            use_real_datasets=not test_data,
        )
        raise typer.Exit(0)
    except Exception as e:
        typer.echo(f"Training failed: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


def _run_stage1_only(test_data: bool) -> None:
    """Run Stage 1 training only"""
    try:
        from ...experiments.simrag.run_stage1 import run_stage1_training
        from ...logging_config import setup_logging

        setup_logging()

        typer.echo("=== Stage 1 Training ===")
        typer.echo("")
        if test_data:
            typer.echo("⚠️  Using test data (not recommended for production)")
        else:
            typer.echo("Using Alpaca dataset (52K instruction-following examples)")
        typer.echo("")
        typer.echo("Running Stage 1 training...")
        
        results = run_stage1_training(
            use_real_datasets=not test_data,
            output_file="stage1_results.json",
            use_timestamp=True
        )
        
        typer.echo("\n✓ Stage 1 training complete!")
        typer.echo(f"  Version: {results['training']['version']}")
        typer.echo(f"  Training time: {results['training']['training_time_seconds']:.1f}s")
        if results['training'].get('final_loss'):
            typer.echo(f"  Final loss: {results['training']['final_loss']:.4f}")
        typer.echo(f"  Model path: {results['training']['model_path']}")
        typer.echo(f"\nNext: Run Stage 2 with this model:")
        typer.echo(f"  simrag experiment stage2")
        typer.echo(f"  (or specify model: simrag experiment stage2 --stage1-model {results['training']['model_path']})")
        
        raise typer.Exit(0)
    except Exception as e:
        typer.echo(f"Stage 1 training failed: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


def _run_stage2_only(documents: Optional[str], stage1_model: Optional[str]) -> None:
    """Run Stage 2 training only"""
    try:
        from ...experiments.simrag.run_stage2 import run_stage2_training
        from ...experiments.test_model import list_available_models
        from ...config import get_tuning_config
        from ...logging_config import setup_logging

        setup_logging()

        # Resolve documents path
        if documents:
            documents_folder = documents
        else:
            cli_file = Path(__file__)
            project_root = cli_file.parent.parent.parent.parent
            documents_folder = str(project_root / "data" / "documents")

        # If no Stage 1 model specified, show interactive menu
        if not stage1_model:
            config = get_tuning_config()
            
            typer.echo("=== Stage 2 Training ===")
            typer.echo("")
            typer.echo("Select a Stage 1 model to continue from:")
            typer.echo("")
            
            # List all Stage 1 models
            models = list_available_models("stage_1", config.model_size)
            
            if not models:
                typer.echo("❌ No Stage 1 models found!", err=True)
                typer.echo(f"   Expected models in: tuned_models/model_{'1b' if config.model_size == 'small' else '8b'}/stage_1/", err=True)
                typer.echo("   Please train a Stage 1 model first using: simrag experiment stage1", err=True)
                raise typer.Exit(1)
            
            typer.echo(f"Available Stage 1 models ({len(models)}):")
            for i, model in enumerate(models, 1):
                display_name = model.get('display_name', model['version'])
                loss_str = f", Loss: {model['final_loss']:.4f}" if model['final_loss'] else ""
                time_str = f", Time: {model['training_time']:.1f}s" if model['training_time'] else ""
                notes_str = f" - {model['notes']}" if model['notes'] else ""
                typer.echo(f"  {i}. {display_name}{loss_str}{time_str}{notes_str}")
            typer.echo("  0. Exit")
            typer.echo("")
            
            # Prompt for selection
            try:
                model_choice = typer.prompt("Enter model number", default="0").strip()
                if model_choice == "0":
                    typer.echo("Exiting...")
                    raise typer.Exit(0)
                
                model_num = int(model_choice)
                if 1 <= model_num <= len(models):
                    selected_model = models[model_num - 1]
                    stage1_model = selected_model["path"]
                    version = selected_model["version"]
                    checkpoint = selected_model.get("checkpoint")
                    
                    if checkpoint:
                        typer.echo(f"\n✓ Selected: {version} ({checkpoint})")
                    else:
                        typer.echo(f"\n✓ Selected: {version} (final)")
                else:
                    typer.echo(f"Invalid choice: {model_num}", err=True)
                    raise typer.Exit(1)
            except ValueError:
                typer.echo("Invalid input. Please enter a number.", err=True)
                raise typer.Exit(1)

        typer.echo("\nRunning Stage 2 training...")
        typer.echo(f"Using Stage 1 model: {stage1_model}")
        
        results = run_stage2_training(
            documents_folder=documents_folder,
            stage_1_model_path=stage1_model,
            output_file="stage2_results.json",
            use_timestamp=True
        )
        
        typer.echo("\n✓ Stage 2 training complete!")
        typer.echo(f"  Version: {results['training']['version']}")
        if 'total_rounds' in results['training']:
            typer.echo(f"  Rounds: {results['training']['total_rounds']}")
        typer.echo(f"  Model path: {results['training']['model_path']}")
        typer.echo(f"\nNext: Export and test in Colab notebook")
        typer.echo(f"  simrag experiment export")
        
        raise typer.Exit(0)
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Stage 2 training failed: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


def _run_baseline(documents: Optional[str]) -> None:
    """Run baseline experiment"""
    try:
        from ...experiments.baseline.run_baseline import run_baseline_test
        from ...logging_config import setup_logging

        setup_logging()

        # Resolve path relative to project root when running via CLI
        if documents:
            documents_folder = documents
        else:
            # Default: data/documents relative to project root
            # CLI file is at: simrag_reproduction/cli/commands/experiment.py
            # Project root is 4 levels up: commands -> cli -> simrag_reproduction -> project_root
            cli_file = Path(__file__)
            project_root = cli_file.parent.parent.parent.parent
            documents_folder = str(project_root / "data" / "documents")
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
        from ...experiments.simrag.run_full_pipeline import run_full_pipeline
        from ...logging_config import setup_logging

        setup_logging()

        # Resolve path relative to project root when running via CLI
        if documents:
            documents_folder = documents
        else:
            # Default: data/documents relative to project root
            # CLI file is at: simrag_reproduction/cli/commands/experiment.py
            # Project root is 4 levels up: commands -> cli -> simrag_reproduction -> project_root
            cli_file = Path(__file__)
            project_root = cli_file.parent.parent.parent.parent
            documents_folder = str(project_root / "data" / "documents")
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
        from ...experiments.comparison.compare_results import compare_results
        from ...experiments.utils import find_most_recent_results_file
        from ...logging_config import setup_logging
        from pathlib import Path

        setup_logging()

        results_dir = Path("simrag_reproduction/experiments")
        
        # Find most recent files if not specified
        if not baseline_file:
            baseline_results_dir = results_dir / "baseline" / "results"
            baseline_file = find_most_recent_results_file(
                results_dir=baseline_results_dir,
                base_filename="baseline_results.json",
                fallback_filename="baseline_results.json"
            )
            typer.echo(f"Using most recent baseline results: {baseline_file}")
        
        if not simrag_file:
            simrag_results_dir = results_dir / "simrag" / "results"
            simrag_file = find_most_recent_results_file(
                results_dir=simrag_results_dir,
                base_filename="full_pipeline_results.json",
                fallback_filename="full_pipeline_results.json"
            )
            typer.echo(f"Using most recent SimRAG results: {simrag_file}")

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


def _run_export_model() -> None:
    """Export a model to ZIP file for Colab"""
    try:
        from ...experiments.test_model import list_available_models, export_model
        from ...config import get_tuning_config
        from ...logging_config import setup_logging
        from pathlib import Path

        setup_logging()

        config = get_tuning_config()
        
        # Step 1: Choose stage
        typer.echo("=== Model Export ===")
        typer.echo("")
        typer.echo("Select stage:")
        typer.echo("  1. Stage 1 (Instruction Following)")
        typer.echo("  2. Stage 2 (Domain Adaptation)")
        typer.echo("  0. Exit")
        typer.echo("")
        
        try:
            stage_choice = typer.prompt("Enter stage number", default="0").strip()
            if stage_choice == "0":
                typer.echo("Exiting...")
                raise typer.Exit(0)
            
            stage_num = int(stage_choice)
            if stage_num == 1:
                stage = "stage_1"
            elif stage_num == 2:
                stage = "stage_2"
            else:
                typer.echo(f"Invalid choice: {stage_num}", err=True)
                raise typer.Exit(1)
        except ValueError:
            typer.echo("Invalid input. Please enter a number.", err=True)
            raise typer.Exit(1)
        
        # Step 2: List available models
        typer.echo(f"\nLoading models for {stage}...")
        models = list_available_models(stage, config.model_size)
        
        if not models:
            typer.echo(f"No models found for {stage}!", err=True)
            typer.echo(f"Expected models in: tuned_models/model_{'1b' if config.model_size == 'small' else '8b'}/{stage}/", err=True)
            raise typer.Exit(1)
        
        typer.echo(f"\nAvailable models ({len(models)}):")
        for i, model in enumerate(models, 1):
            display_name = model.get('display_name', model['version'])
            loss_str = f", Loss: {model['final_loss']:.4f}" if model['final_loss'] else ""
            time_str = f", Time: {model['training_time']:.1f}s" if model['training_time'] else ""
            notes_str = f" - {model['notes']}" if model['notes'] else ""
            typer.echo(f"  {i}. {display_name}{loss_str}{time_str}{notes_str}")
        typer.echo("  0. Exit")
        typer.echo("")
        
        # Step 3: Choose model
        try:
            model_choice = typer.prompt("Enter model number", default="0").strip()
            if model_choice == "0":
                typer.echo("Exiting...")
                raise typer.Exit(0)
            
            model_num = int(model_choice)
            if 1 <= model_num <= len(models):
                selected_model = models[model_num - 1]
                model_path = selected_model["path"]
                version = selected_model["version"]
                checkpoint = selected_model.get("checkpoint")
            else:
                typer.echo(f"Invalid choice: {model_num}", err=True)
                raise typer.Exit(1)
        except ValueError:
            typer.echo("Invalid input. Please enter a number.", err=True)
            raise typer.Exit(1)
        
        # Step 4: Export the model
        typer.echo(f"\nExporting model: {version} ({stage})")
        if checkpoint:
            typer.echo(f"Checkpoint: {checkpoint}")
        typer.echo(f"Model path: {model_path}")
        typer.echo("")
        
        zip_path = export_model(
            model_path=model_path,
            stage=stage
        )
        
        typer.echo("\n=== Export Complete ===")
        typer.echo(f"✓ ZIP file created: {zip_path}")
        typer.echo(f"\nUpload this file to Google Colab for testing.")
        
        raise typer.Exit(0)
    except KeyboardInterrupt:
        typer.echo("\nOperation cancelled.")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Model export failed: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


def _run_display_results() -> None:
    """Display experiment results with interactive selection"""
    try:
        from ...logging_config import setup_logging
        from datetime import datetime
        import json

        setup_logging()

        # Find comparison results directory (in project root for easier access)
        cli_file = Path(__file__)
        project_root = cli_file.parent.parent.parent.parent
        results_dir = project_root / "comparison_results"
        
        # Create directory if it doesn't exist
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # List all comparison result files
        result_files = sorted(
            results_dir.glob("comparison_results_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if not result_files:
            typer.echo("No comparison results found!", err=True)
            typer.echo(f"Expected results in: {results_dir}", err=True)
            typer.echo("\nRun the Colab notebook to generate comparison results.", err=True)
            raise typer.Exit(1)
        
        # Display menu
        typer.echo("=== View Comparison Results ===")
        typer.echo("")
        typer.echo(f"Available results ({len(result_files)}):")
        
        for i, result_file in enumerate(result_files, 1):
            # Try to extract timestamp from filename
            try:
                # Format: comparison_results_YYYY-MM-DD_HH-MM-SS.json
                name = result_file.stem
                if "_" in name:
                    parts = name.split("_")
                    if len(parts) >= 4:
                        date_part = "_".join(parts[2:4])  # YYYY-MM-DD_HH-MM-SS
                        timestamp = datetime.strptime(date_part, "%Y-%m-%d_%H-%M-%S")
                        formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        formatted_time = "Unknown"
                else:
                    formatted_time = "Unknown"
            except:
                formatted_time = "Unknown"
            
            size_kb = result_file.stat().st_size / 1024
            typer.echo(f"  {i}. {result_file.name}")
            typer.echo(f"     {formatted_time} ({size_kb:.1f} KB)")
        
        typer.echo("  0. Exit")
        typer.echo("")
        
        # Get user selection
        try:
            choice = typer.prompt("Enter result number", default="0").strip()
            if choice == "0":
                typer.echo("Exiting...")
                raise typer.Exit(0)
            
            result_num = int(choice)
            if 1 <= result_num <= len(result_files):
                selected_file = result_files[result_num - 1]
                _display_comparison_results(selected_file)
            else:
                typer.echo(f"Invalid choice: {result_num}", err=True)
                raise typer.Exit(1)
        except ValueError:
            typer.echo("Invalid input. Please enter a number.", err=True)
            raise typer.Exit(1)
        
        raise typer.Exit(0)
    except KeyboardInterrupt:
        typer.echo("\nOperation cancelled.")
        raise typer.Exit(1)
    except typer.Exit:
        # Re-raise Typer exit exceptions (they're not errors)
        raise
    except Exception as e:
        typer.echo(f"Failed to display results: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


def _display_comparison_results(result_file: Path) -> None:
    """Display comparison results in a clean format"""
    import json
    
    with open(result_file, 'r') as f:
        comparison = json.load(f)
    
    typer.echo("\n" + "="*70)
    typer.echo("COMPARISON RESULTS")
    typer.echo("="*70)
    
    # Validation info
    validation = comparison.get("validation", {})
    if validation.get("is_valid"):
        typer.echo("✓ Experiment validation: PASSED")
    else:
        typer.echo("⚠️  Experiment validation: FAILED")
        for issue in validation.get("issues", []):
            typer.echo(f"   - {issue}")
    
    typer.echo("")
    
    # Baseline metrics
    baseline = comparison.get("baseline", {})
    typer.echo("BASELINE (Base Model):")
    typer.echo(f"  Context Score:      {baseline.get('avg_context_score', 0.0):.3f}")
    typer.echo(f"  Answer Quality:     {baseline.get('avg_answer_quality', 0.0):.3f}")
    typer.echo(f"  Response Time:      {baseline.get('avg_response_time', 0.0):.2f}s")
    typer.echo(f"  Questions Tested:   {baseline.get('num_questions', 0)}")
    
    # Statistical info
    baseline_stats = baseline.get("context_score_stats", {})
    if baseline_stats:
        typer.echo(f"  Statistics:         {baseline_stats.get('mean', 0.0):.3f} ± {baseline_stats.get('std', 0.0):.3f}")
        typer.echo(f"  95% CI:            [{baseline_stats.get('ci_lower', 0.0):.3f}, {baseline_stats.get('ci_upper', 0.0):.3f}]")
    
    typer.echo("")
    
    # SimRAG metrics
    simrag = comparison.get("simrag", {})
    typer.echo("SIMRAG (Fine-tuned Model):")
    model_info = simrag.get("model", {})
    if model_info:
        typer.echo(f"  Model:             {model_info.get('base_model', 'N/A')}")
        typer.echo(f"  Stage:             {model_info.get('stage', 'N/A')}")
        typer.echo(f"  Version:           {model_info.get('version', 'N/A')}")
        if model_info.get("checkpoint"):
            typer.echo(f"  Checkpoint:        {model_info.get('checkpoint')}")
    typer.echo(f"  Context Score:      {simrag.get('avg_context_score', 0.0):.3f}")
    typer.echo(f"  Answer Quality:     {simrag.get('avg_answer_quality', 0.0):.3f}")
    typer.echo(f"  Response Time:      {simrag.get('avg_response_time', 0.0):.2f}s")
    typer.echo(f"  Questions Tested:   {simrag.get('num_questions', 0)}")
    
    # Statistical info
    simrag_stats = simrag.get("context_score_stats", {})
    if simrag_stats:
        typer.echo(f"  Statistics:         {simrag_stats.get('mean', 0.0):.3f} ± {simrag_stats.get('std', 0.0):.3f}")
        typer.echo(f"  95% CI:            [{simrag_stats.get('ci_lower', 0.0):.3f}, {simrag_stats.get('ci_upper', 0.0):.3f}]")
    
    typer.echo("")
    
    # Improvement metrics
    improvement = comparison.get("improvement", {})
    typer.echo("IMPROVEMENT:")
    context_improvement = improvement.get("context_score_improvement_percent", 0.0)
    quality_improvement = improvement.get("answer_quality_improvement", 0.0)
    time_change = improvement.get("response_time_change_percent", 0.0)
    
    # Color code improvements (positive = good, negative = bad)
    context_sign = "+" if context_improvement >= 0 else ""
    quality_sign = "+" if quality_improvement >= 0 else ""
    time_sign = "+" if time_change <= 0 else ""  # Negative time change is good
    
    typer.echo(f"  Context Score:      {context_sign}{context_improvement:+.1f}%")
    typer.echo(f"  Answer Quality:     {quality_sign}{quality_improvement:+.3f}")
    typer.echo(f"  Response Time:      {time_sign}{time_change:+.1f}%")
    
    # Statistical significance
    sig_info = improvement.get("statistical_significance", {})
    if sig_info:
        typer.echo("")
        typer.echo("STATISTICAL SIGNIFICANCE:")
        overlap = sig_info.get("overlap", True)
        if not overlap:
            typer.echo("  ✓ Statistically significant (p < 0.05)")
            typer.echo("    Confidence intervals do not overlap")
        else:
            typer.echo("  ⚠️  Not statistically significant")
            typer.echo("    Confidence intervals overlap")
        typer.echo(f"  Baseline CI: {sig_info.get('baseline_ci', 'N/A')}")
        typer.echo(f"  SimRAG CI:   {sig_info.get('simrag_ci', 'N/A')}")
    
    # Interpretations and Conclusions
    typer.echo("")
    typer.echo("="*70)
    typer.echo("INTERPRETATION & CONCLUSIONS")
    typer.echo("="*70)
    
    # Context score interpretation
    context_improvement = improvement.get("context_score_improvement_percent", 0.0)
    if context_improvement > 10:
        typer.echo("✓ Context Score: Strong improvement (>10%)")
        typer.echo("  The fine-tuned model retrieves more relevant documents.")
    elif context_improvement > 5:
        typer.echo("✓ Context Score: Moderate improvement (5-10%)")
        typer.echo("  The fine-tuned model shows better document retrieval.")
    elif context_improvement > 0:
        typer.echo("⚠️  Context Score: Minor improvement (<5%)")
        typer.echo("  Small improvement - may not be practically significant.")
    else:
        typer.echo("✗ Context Score: Regression")
        typer.echo("  Fine-tuning did not improve document retrieval quality.")
    
    # Answer quality interpretation
    quality_improvement = improvement.get("answer_quality_improvement", 0.0)
    if quality_improvement > 0.1:
        typer.echo("✓ Answer Quality: Strong improvement (>0.1)")
        typer.echo("  The fine-tuned model generates significantly better answers.")
    elif quality_improvement > 0.05:
        typer.echo("✓ Answer Quality: Moderate improvement (0.05-0.1)")
        typer.echo("  The fine-tuned model shows better answer quality.")
    elif quality_improvement > 0:
        typer.echo("⚠️  Answer Quality: Minor improvement (<0.05)")
        typer.echo("  Small improvement - may need more training or data.")
    else:
        typer.echo("✗ Answer Quality: Regression")
        typer.echo("  Fine-tuning did not improve answer quality.")
    
    # Response time interpretation
    time_change = improvement.get("response_time_change_percent", 0.0)
    if time_change < -10:
        typer.echo("✓ Response Time: Faster (>10% reduction)")
        typer.echo("  The fine-tuned model is significantly faster.")
    elif time_change < -5:
        typer.echo("✓ Response Time: Moderately faster (5-10% reduction)")
        typer.echo("  The fine-tuned model shows improved efficiency.")
    elif time_change < 0:
        typer.echo("⚠️  Response Time: Slightly faster (<5% reduction)")
        typer.echo("  Minor speed improvement.")
    elif time_change < 10:
        typer.echo("⚠️  Response Time: Slightly slower (<10% increase)")
        typer.echo("  Acceptable trade-off for quality improvements.")
    else:
        typer.echo("✗ Response Time: Significantly slower (>10% increase)")
        typer.echo("  Performance regression - consider optimization.")
    
    # Overall conclusion
    typer.echo("")
    typer.echo("OVERALL ASSESSMENT:")
    is_significant = not sig_info.get("overlap", True) if sig_info else False
    has_improvement = context_improvement > 5 or quality_improvement > 0.05
    
    if is_significant and has_improvement:
        typer.echo("  🎉 SUCCESS: Fine-tuning shows statistically significant improvements!")
        typer.echo("     The SimRAG approach is working as expected.")
    elif has_improvement:
        typer.echo("  ✓ PROMISING: Fine-tuning shows improvements, but not statistically significant.")
        typer.echo("     Consider: more training data, longer training, or different hyperparameters.")
    elif context_improvement > 0 or quality_improvement > 0:
        typer.echo("  ⚠️  MIXED: Some improvements, but may not be practically significant.")
        typer.echo("     Consider: reviewing training data quality or adjusting training strategy.")
    else:
        typer.echo("  ✗ NEEDS WORK: Fine-tuning did not show clear improvements.")
        typer.echo("     Consider: checking training data, hyperparameters, or model architecture.")
    
    typer.echo("")
    typer.echo("="*70)
    typer.echo(f"Source file: {result_file.name}")
    typer.echo("="*70)

