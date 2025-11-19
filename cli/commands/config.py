"""Config command - display current configuration"""
import typer
import sys
from pathlib import Path

from ..utils import check_venv


def config() -> None:
    """Show current configuration settings"""
    if not check_venv():
        raise typer.Exit(1)

    typer.echo("=== Current Configuration ===")
    typer.echo("")

    try:
        # Import config directly to avoid triggering __init__.py imports (which import AIGateway -> torch)
        import importlib.util
        config_path = Path(__file__).parent.parent.parent / "config.py"
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        get_rag_config = config_module.get_rag_config
        get_tuning_config = config_module.get_tuning_config
        
        rag_config = get_rag_config()
        tuning_config = get_tuning_config()

        typer.echo("Hardware Configuration:")
        typer.echo(f"  Platform: {'Laptop' if rag_config.use_laptop else 'PC'}")
        typer.echo(f"  Model: {rag_config.model_name}")
        typer.echo("")

        typer.echo("RAG Configuration:")
        typer.echo(f"  AI Provider: {'Ollama' if rag_config.use_ollama else 'Purdue API'}")
        typer.echo(f"  Storage: {'Persistent' if rag_config.use_persistent else 'In-memory'}")
        typer.echo(f"  Collection: {rag_config.collection_name}")
        typer.echo(f"  Top-K: {rag_config.top_k}")
        typer.echo("")

        typer.echo("Tuning Configuration:")
        typer.echo(f"  Model: {tuning_config.model_name}")
        typer.echo(f"  Batch Size: {tuning_config.batch_size}")
        typer.echo(f"  Epochs: {tuning_config.num_epochs}")
        typer.echo(f"  Output Dir: {tuning_config.output_dir}")
        typer.echo("")

        typer.echo("Note: You can override these settings by editing config.py")
        typer.echo("or setting environment variables (USE_LAPTOP, USE_OLLAMA, etc.)")

    except Exception as e:
        typer.echo(f"Error: Could not load configuration: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)

    raise typer.Exit(0)

