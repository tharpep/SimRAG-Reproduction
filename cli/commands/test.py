"""Test command - run test suites"""
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from ..utils import check_venv, is_poetry_environment


def _find_project_root() -> Path:
    """Find the project root directory (where pyproject.toml is)"""
    current = Path.cwd()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    # Fallback: assume we're in project root
    return Path.cwd()


def _find_tests_dir() -> Path:
    """Find the tests directory relative to project root"""
    project_root = _find_project_root()
    # Tests are at tests/ relative to project root
    tests_dir = project_root / "tests"
    if tests_dir.exists():
        return tests_dir
    # Fallback: try from current directory
    tests_dir = Path("tests")
    if tests_dir.exists():
        return tests_dir.resolve()
    return None


def test(
    all_tests: bool = typer.Option(False, "--all", "-a", help="Run all tests"),
    category: Optional[str] = typer.Argument(None, help="Test category to run (e.g., tests_api, tests_rag)"),
) -> None:
    """Run tests - specify category directly or use interactive selection"""
    if not check_venv():
        raise typer.Exit(1)

    # Use Poetry's Python if available, otherwise fallback
    python_cmd = "python"
    if not is_poetry_environment():
        from ..utils import get_python_cmd
        python_cmd = get_python_cmd()

    # Find tests directory
    tests_dir = _find_tests_dir()
    if not tests_dir:
        typer.echo("No tests directory found!", err=True)
        typer.echo("Expected: tests/ directory in project root", err=True)
        raise typer.Exit(1)

    # Check if user wants to run all tests
    if all_tests:
        typer.echo("=== Running All Tests ===")
        typer.echo("")
        
        # Find all test folders
        test_folders = []
        for item in tests_dir.iterdir():
            if item.is_dir() and (item.name.startswith("test_") or item.name.startswith("tests_")):
                test_folders.append(item.name)
        
        if not test_folders:
            typer.echo("No test folders found in tests/ directory!", err=True)
            raise typer.Exit(1)
        
        typer.echo(f"Found {len(test_folders)} test folder(s):")
        for folder in test_folders:
            typer.echo(f"  - {folder}")
        typer.echo("")
        
        # Run tests in each folder individually to ensure all are discovered
        all_passed = True
        failed_folders = []
        project_root = _find_project_root()
        
        for folder in test_folders:
            typer.echo(f"Running tests in: {folder}")
            typer.echo("-" * 50)
            try:
                # Use file path relative to project root
                test_path = tests_dir / folder
                # Convert to relative path from project root
                rel_path = test_path.relative_to(project_root)
                result = subprocess.run(
                    [python_cmd, "-m", "pytest", str(rel_path), "-v", "--cache-clear"],
                    check=True,
                    cwd=project_root
                )
                typer.echo(f"[PASS] {folder} passed")
                typer.echo("")
            except subprocess.CalledProcessError as e:
                typer.echo(f"[FAIL] {folder} failed with exit code {e.returncode}", err=True)
                typer.echo("")
                all_passed = False
                failed_folders.append(folder)
        
        if failed_folders:
            typer.echo("Note: If tests failed due to 'ModuleNotFoundError', run 'poetry install' to install dependencies.", err=True)
        
        typer.echo("=" * 50)
        if all_passed:
            typer.echo("All tests passed!")
            raise typer.Exit(0)
        else:
            typer.echo(f"Tests failed in {len(failed_folders)} folder(s): {', '.join(failed_folders)}", err=True)
            raise typer.Exit(1)

    # If category specified, run that directly
    if category:
        test_path = tests_dir / category
        if not test_path.exists():
            typer.echo(f"Test category '{category}' not found!", err=True)
            typer.echo("Available categories:", err=True)
            _list_test_categories()
            raise typer.Exit(1)

        typer.echo(f"Running: {category}")
        typer.echo("=" * 50)
        project_root = _find_project_root()
        try:
            # Use file path relative to project root
            test_path = tests_dir / category
            rel_path = test_path.relative_to(project_root)
            result = subprocess.run([python_cmd, "-m", "pytest", str(rel_path), "-v", "-s"], check=True, cwd=project_root)
            typer.echo(f"\n{category} completed successfully!")
            raise typer.Exit(0)
        except subprocess.CalledProcessError as e:
            typer.echo(f"\nTest failed with exit code {e.returncode}", err=True)
            typer.echo("Note: If tests failed due to 'ModuleNotFoundError', run 'poetry install' to install dependencies.", err=True)
            raise typer.Exit(e.returncode)

    # No category specified - show interactive selection
    _run_tests_interactive(python_cmd, tests_dir)


def _list_test_categories() -> None:
    """List available test categories"""
    tests_dir = _find_tests_dir()
    if not tests_dir or not tests_dir.exists():
        return

    test_folders = []
    for item in tests_dir.iterdir():
        if item.is_dir() and (item.name.startswith("test_") or item.name.startswith("tests_")):
            test_folders.append(item.name)

    for folder in sorted(test_folders):
        typer.echo(f"  - {folder}", err=True)


def _run_tests_interactive(python_cmd: str, tests_dir: Path) -> None:
    """Interactive test selection menu"""
    if not tests_dir or not tests_dir.exists():
        typer.echo("No tests directory found!", err=True)
        raise typer.Exit(1)

    # Find all test folders
    test_folders = []
    for item in tests_dir.iterdir():
        if item.is_dir() and (item.name.startswith("test_") or item.name.startswith("tests_")):
            test_folders.append(item.name)

    if not test_folders:
        typer.echo("No test folders found in tests/ directory!", err=True)
        raise typer.Exit(1)

    # Interactive test selection
    typer.echo("=== Test Selection ===")
    typer.echo("")

    # Display test options
    typer.echo("Available test folders:")
    for i, folder_name in enumerate(test_folders, 1):
        typer.echo(f"  {i}. {folder_name}")
    typer.echo("  0. Exit")
    typer.echo("")

    # Get user selection
    try:
        choice = typer.prompt("Enter test number", default="0").strip()

        if choice == "0":
            typer.echo("Exiting...")
            raise typer.Exit(0)

        choice_num = int(choice)
        if 1 <= choice_num <= len(test_folders):
            selected_folder = test_folders[choice_num - 1]
            typer.echo(f"\nRunning: {selected_folder}")
            typer.echo("=" * 50)

            # Run pytest on the selected folder with verbose output using file path
            project_root = _find_project_root()
            test_path = tests_dir / selected_folder
            rel_path = test_path.relative_to(project_root)
            result = subprocess.run([python_cmd, "-m", "pytest", str(rel_path), "-v", "-s"], check=True, cwd=project_root)

            typer.echo(f"\n{selected_folder} completed successfully!")
            raise typer.Exit(0)
        else:
            typer.echo(f"Invalid choice: {choice_num}", err=True)
            raise typer.Exit(1)

    except ValueError:
        typer.echo("Invalid input. Please enter a number.", err=True)
        raise typer.Exit(1)
    except KeyboardInterrupt:
        typer.echo("\nOperation cancelled.")
        raise typer.Exit(1)
    except subprocess.CalledProcessError as e:
        typer.echo(f"\nTest failed with exit code {e.returncode}", err=True)
        raise typer.Exit(e.returncode)

