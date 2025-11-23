#!/usr/bin/env python3
"""
Clean personal information from experiment result files.
Replaces absolute paths with relative paths to ensure anonymity.
"""

import json
import os
from pathlib import Path


def clean_path(path_str):
    """
    Clean a path string by converting absolute paths to relative paths.

    Example:
        Input: "C:\\Users\\aatha\\Desktop\\...\\SimRAG-Reproduction\\data\\documents"
        Output: "data/documents"
    """
    if not path_str:
        return path_str

    # Convert to Path object
    try:
        path = Path(path_str)

        # Find the project root (SimRAG-Reproduction)
        parts = list(path.parts)

        # Look for SimRAG-Reproduction in the path
        if 'SimRAG-Reproduction' in parts:
            idx = parts.index('SimRAG-Reproduction')
            # Get everything after SimRAG-Reproduction
            relative_parts = parts[idx + 1:]
            if relative_parts:
                # Return as forward-slash path (platform-independent)
                return '/'.join(relative_parts)

        # If can't find SimRAG-Reproduction, return just the last few meaningful parts
        if len(parts) >= 2:
            # Return last 2-3 parts (e.g., data/documents)
            return '/'.join(parts[-2:])

        return path_str
    except Exception:
        return path_str


def clean_json_object(obj):
    """
    Recursively clean all path strings in a JSON object.
    """
    if isinstance(obj, dict):
        return {k: clean_json_object(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_object(item) for item in obj]
    elif isinstance(obj, str):
        # Check if this looks like a path (contains backslashes, forward slashes, or C:)
        if ('\\' in obj or 'C:' in obj or 'Users' in obj or
            obj.startswith('/') and len(obj) > 10):
            return clean_path(obj)
        return obj
    else:
        return obj


def clean_result_file(filepath):
    """
    Clean a single result JSON file.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Clean all paths in the JSON
        cleaned_data = clean_json_object(data)

        # Write back to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

        return True
    except Exception as e:
        print(f"Error cleaning {filepath}: {e}")
        return False


def main():
    """
    Clean all experiment result files.
    """
    root_dir = Path(__file__).parent

    # Directories to clean
    result_dirs = [
        root_dir / 'simrag_reproduction' / 'experiments' / 'baseline' / 'results',
        root_dir / 'simrag_reproduction' / 'experiments' / 'simrag' / 'results',
    ]

    total_files = 0
    cleaned_files = 0

    print("Cleaning personal information from experiment results...")
    print()

    for results_dir in result_dirs:
        if not results_dir.exists():
            continue

        print(f"Processing: {results_dir.relative_to(root_dir)}")

        for file in results_dir.glob('*.json'):
            total_files += 1
            if clean_result_file(file):
                cleaned_files += 1
                print(f"  [OK] {file.name}")
            else:
                print(f"  [FAIL] {file.name}")

    print()
    print(f"[SUCCESS] Cleaned {cleaned_files}/{total_files} files")
    print()
    print("Verifying: Checking for remaining personal paths...")

    # Verify no personal info remains
    issues_found = 0
    for results_dir in result_dirs:
        if not results_dir.exists():
            continue

        for file in results_dir.glob('*.json'):
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'aatha' in content.lower() or 'C:\\Users' in content:
                    issues_found += 1
                    print(f"  WARNING: {file.name} still contains personal info")

    if issues_found == 0:
        print("  [OK] No personal information found!")
    else:
        print(f"  [WARNING] Found {issues_found} files with personal info")

    return issues_found == 0


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
