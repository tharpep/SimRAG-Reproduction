"""Demo command - run demos"""
import typer

from ..utils import check_venv


def demo() -> None:
    """Run demos - currently redirects to experiment commands"""
    if not check_venv():
        raise typer.Exit(1)

    typer.echo("=== Demo Selection ===")
    typer.echo("")
    typer.echo("Demo functionality has been integrated into experiment commands.")
    typer.echo("")
    typer.echo("Available commands:")
    typer.echo("  simrag experiment run      - Run full experiment pipeline")
    typer.echo("  simrag experiment baseline - Run baseline RAG test")
    typer.echo("  simrag experiment simrag   - Run SimRAG pipeline")
    typer.echo("")
    typer.echo("For RAG and tuning demos, use the experiment commands above.")

