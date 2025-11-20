"""SimRAG CLI - Main entry point"""
import typer

from .commands import test, config, experiment

app = typer.Typer(
    name="simrag",
    help="SimRAG Reproduction - RAG and fine-tuning implementation",
    add_completion=True,
)

# Register subcommands
app.command(name="test")(test)
app.command(name="config")(config)
app.command(name="experiment")(experiment)


def main() -> None:
    """Main CLI entry point"""
    app()


if __name__ == "__main__":
    main()

