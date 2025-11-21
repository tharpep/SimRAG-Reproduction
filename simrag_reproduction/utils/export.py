"""
Export utilities for creating cross-platform model packages
"""

import zipfile
from pathlib import Path
from typing import Optional


def create_cross_platform_zip(source_dir: str, output_name: Optional[str] = None) -> str:
    """
    Create a cross-platform compatible ZIP file

    Fixes Windows backslash issue by forcing forward slashes in ZIP archive paths.
    This ensures the ZIP can be extracted properly on Linux/macOS.

    Args:
        source_dir: Path to directory to ZIP
        output_name: Output ZIP filename (default: source_dir.name + .zip)

    Returns:
        Path to created ZIP file

    Raises:
        ValueError: If source_dir doesn't exist or isn't a directory
    """
    source_path = Path(source_dir).resolve()

    if not source_path.exists():
        raise ValueError(f"Source directory not found: {source_dir}")

    if not source_path.is_dir():
        raise ValueError(f"Not a directory: {source_dir}")

    # Default output name
    if output_name is None:
        output_name = f"{source_path.name}.zip"

    output_path = Path(output_name).resolve()

    # Create ZIP with explicit control over path separators
    file_count = 0
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in source_path.rglob('*'):
            if file_path.is_file():
                # Calculate relative path from parent directory
                # This includes the source directory name in the archive
                rel_path = file_path.relative_to(source_path.parent)

                # Force forward slashes for cross-platform compatibility
                arcname = str(rel_path).replace('\\', '/')

                # Add to ZIP
                zf.write(file_path, arcname=arcname)
                file_count += 1

    return str(output_path)


def verify_zip_contents(zip_path: str, required_files: Optional[list] = None) -> dict:
    """
    Verify ZIP contents and check for issues

    Args:
        zip_path: Path to ZIP file
        required_files: List of required filenames (checked recursively)

    Returns:
        Dictionary with verification results:
        - valid: bool - Whether ZIP is valid
        - file_count: int - Number of files in ZIP
        - has_backslashes: bool - Whether any paths contain backslashes
        - missing_files: list - List of required files not found
        - found_files: dict - Map of required file names to their paths in ZIP
    """
    if required_files is None:
        required_files = []

    results = {
        "valid": True,
        "file_count": 0,
        "has_backslashes": False,
        "missing_files": [],
        "found_files": {}
    }

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = zf.namelist()
            results["file_count"] = len(names)

            # Check for backslashes
            backslash_files = [n for n in names if '\\' in n]
            if backslash_files:
                results["has_backslashes"] = True
                results["valid"] = False

            # Check for required files
            for req_file in required_files:
                found = [n for n in names if n.endswith(f"/{req_file}") or n.endswith(f"{req_file}")]
                if found:
                    results["found_files"][req_file] = found[0]
                else:
                    results["missing_files"].append(req_file)
                    results["valid"] = False

    except Exception as e:
        results["valid"] = False
        results["error"] = str(e)

    return results
