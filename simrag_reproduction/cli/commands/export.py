"""Export command - create cross-platform ZIPs of trained models for Colab testing"""

from pathlib import Path
from typing import Optional

import typer

from ..utils import check_venv


def export(
    model_path: Optional[str] = typer.Argument(None, help="Model path to export (optional, interactive if not provided)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output ZIP filename"),
) -> None:
    """
    Export trained model as cross-platform ZIP for Colab testing

    Creates a ZIP file with proper forward-slash paths that work on Linux/macOS.
    Saves the ZIP in the model's directory for easy access.

    Examples:
        simrag export                                    # Interactive menu
        simrag export tuned_models/.../checkpoint-1000   # Direct export
    """
    if not check_venv():
        raise typer.Exit(1)

    try:
        from ...experiments.test_model import list_available_models
        from ...config import get_tuning_config
        from ...utils.export import create_cross_platform_zip, verify_zip_contents
        from ...logging_config import setup_logging

        setup_logging()
        config = get_tuning_config()

        # If model path provided, export directly
        if model_path:
            _export_model(model_path, output)
            raise typer.Exit(0)

        # Otherwise, show interactive menu
        typer.echo("=== Export Model for Colab ===")
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

        # List available models
        typer.echo(f"\nLoading models for {stage}...")
        models = list_available_models(stage, config.model_size)

        if not models:
            typer.echo(f"No models found for {stage}!", err=True)
            typer.echo(
                f"Expected models in: tuned_models/model_{'1b' if config.model_size == 'small' else '8b'}/{stage}/",
                err=True
            )
            raise typer.Exit(1)

        typer.echo(f"\nAvailable models ({len(models)}):")
        for i, model in enumerate(models, 1):
            display_name = model.get('display_name', model['version'])
            notes = model.get('notes', '')
            notes_str = f" - {notes}" if notes else ""
            typer.echo(f"  {i}. {display_name}{notes_str}")
        typer.echo("  0. Exit")
        typer.echo("")

        # Choose model
        try:
            model_choice = typer.prompt("Enter model number", default="0").strip()
            if model_choice == "0":
                typer.echo("Exiting...")
                raise typer.Exit(0)

            model_num = int(model_choice)
            if 1 <= model_num <= len(models):
                selected_model = models[model_num - 1]
                model_path = selected_model["path"]
            else:
                typer.echo(f"Invalid choice: {model_num}", err=True)
                raise typer.Exit(1)
        except ValueError:
            typer.echo("Invalid input. Please enter a number.", err=True)
            raise typer.Exit(1)

        # Export the selected model
        _export_model(model_path, output)

        raise typer.Exit(0)

    except KeyboardInterrupt:
        typer.echo("\nOperation cancelled.")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Export failed: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


def _export_model(model_path: str, output_name: Optional[str] = None) -> None:
    """Export a model to ZIP"""
    from ...utils.export import create_cross_platform_zip, verify_zip_contents

    model_path_obj = Path(model_path).resolve()

    if not model_path_obj.exists():
        typer.echo(f"Model path not found: {model_path}", err=True)
        raise typer.Exit(1)

    if not model_path_obj.is_dir():
        typer.echo(f"Not a directory: {model_path}", err=True)
        raise typer.Exit(1)

    typer.echo(f"\nExporting model: {model_path_obj.name}")
    typer.echo(f"From: {model_path_obj}")

    # Verify required files exist
    adapter_file = model_path_obj / "adapter_model.safetensors"
    adapter_config = model_path_obj / "adapter_config.json"

    if not adapter_file.exists():
        typer.echo(f"⚠️  Warning: adapter_model.safetensors not found", err=True)
    else:
        size_mb = adapter_file.stat().st_size / (1024 ** 2)
        typer.echo(f"  ✓ adapter_model.safetensors ({size_mb:.1f} MB)")

    if not adapter_config.exists():
        typer.echo(f"⚠️  Warning: adapter_config.json not found", err=True)
    else:
        typer.echo(f"  ✓ adapter_config.json")

    # Determine output path (save in model directory)
    if output_name is None:
        output_name = f"{model_path_obj.name}.zip"

    output_path = model_path_obj.parent / output_name

    typer.echo(f"\nCreating ZIP: {output_name}")

    try:
        # Create ZIP
        zip_path = create_cross_platform_zip(str(model_path_obj), str(output_path))

        # Verify ZIP
        typer.echo("Verifying ZIP...")
        verification = verify_zip_contents(
            zip_path,
            required_files=["adapter_model.safetensors", "adapter_config.json"]
        )

        if not verification["valid"]:
            if verification.get("has_backslashes"):
                typer.echo("⚠️  Warning: ZIP contains backslashes (may not work on Linux)", err=True)
            if verification.get("missing_files"):
                typer.echo(f"⚠️  Warning: Missing required files: {verification['missing_files']}", err=True)

        size_mb = Path(zip_path).stat().st_size / (1024 ** 2)
        typer.echo(f"\n✅ Export complete!")
        typer.echo(f"   ZIP: {Path(zip_path).name}")
        typer.echo(f"   Size: {size_mb:.1f} MB")
        typer.echo(f"   Files: {verification['file_count']}")
        typer.echo(f"   Location: {Path(zip_path).parent}")
        typer.echo(f"\n📤 Upload this ZIP to Google Colab for testing")

    except Exception as e:
        typer.echo(f"Failed to create ZIP: {e}", err=True)
        raise typer.Exit(1)
