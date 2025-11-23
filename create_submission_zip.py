#!/usr/bin/env python3
"""
Create a ZIP file of the codebase for submission.
Excludes:
- Everything in .gitignore (except tuned_models metadata)
- project_docs/ folder
- .github/ folder
- CLAUDE.md file
- Model weights (*.safetensors, *.bin) from tuned_models/
- Training checkpoints (checkpoint-*/) from tuned_models/
Includes:
- Model metadata (adapter_config.json, model_info.json, model_registry.json)
- Tokenizer files (tokenizer*.json, vocab.json, merges.txt, etc.)
- README.md files in tuned_models/
- comparison_results/ JSON files
- data/documents/ source files
"""

import os
import zipfile
from pathlib import Path
import fnmatch
from datetime import datetime

# Directories to always exclude
EXCLUDE_DIRS = {'project_docs', '.github', '__pycache__', '.git'}

# Specific files to exclude
EXCLUDE_FILES = {'CLAUDE.md'}

# Files to explicitly include (even if .gitignore would exclude them)
# These are critical for project setup and reproducibility
EXPLICIT_INCLUDE_FILES = {
    '.env.example',  # Configuration template (README references it)
    'test_model_colab.ipynb',  # Colab notebook (explicitly included in .gitignore)
    'LICENSE',  # License file
    '.dockerignore',  # Docker ignore file (needed for Docker builds)
    'poetry.lock',  # Lock file for reproducible builds
    'requirements.txt',  # Backup dependency file
}

# Model weight file extensions to exclude
MODEL_WEIGHT_EXTENSIONS = {'.safetensors', '.bin', '.pt', '.pth'}

# Files to include from tuned_models/ (metadata only)
TUNED_MODELS_INCLUDE_PATTERNS = [
    '*.json',  # adapter_config.json, model_info.json, model_registry.json, tokenizer files
    '*.txt',   # vocab.json, merges.txt, added_tokens.json
    '*.md',    # README.md
    '*.jinja', # chat_template.jinja
    '*.toml',  # Any config files
]

# Directories to exclude from tuned_models/
TUNED_MODELS_EXCLUDE_DIRS = ['checkpoint-*']

def parse_gitignore(gitignore_path):
    """Parse .gitignore file and return a list of patterns."""
    patterns = []
    if not gitignore_path.exists():
        return patterns
    
    with open(gitignore_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            # Handle negation patterns (lines starting with !)
            if line.startswith('!'):
                patterns.append(('include', line[1:].strip()))
            else:
                patterns.append(('exclude', line))
    
    return patterns

def is_tuned_models_file(file_path, root_dir):
    """Check if a file is in tuned_models/ directory."""
    try:
        rel_path = file_path.relative_to(root_dir)
        return 'tuned_models' in rel_path.parts
    except ValueError:
        return False

def should_include_tuned_models_file(file_path):
    """Check if a file in tuned_models/ should be included (metadata only, no weights)."""
    # Exclude checkpoint directories
    rel_str = str(file_path).replace('\\', '/')
    if 'checkpoint-' in rel_str:
        return False
    
    # Exclude model weight files
    if file_path.suffix.lower() in MODEL_WEIGHT_EXTENSIONS:
        return False
    
    # Include metadata files
    file_name = file_path.name.lower()
    for pattern in TUNED_MODELS_INCLUDE_PATTERNS:
        if fnmatch.fnmatch(file_name, pattern):
            return True
    
    # Include specific important files
    important_files = ['model_registry.json', 'adapter_config.json', 'model_info.json', 
                      'README.md', 'tokenizer.json', 'tokenizer_config.json', 
                      'vocab.json', 'merges.txt', 'added_tokens.json', 
                      'special_tokens_map.json', 'chat_template.jinja']
    if file_name in important_files:
        return True
    
    return False

def should_ignore(path, gitignore_patterns, root_dir):
    """Check if a path should be ignored based on .gitignore patterns."""
    # Special handling for tuned_models/ - override .gitignore for metadata files
    if is_tuned_models_file(path, root_dir):
        # For directories in tuned_models, check if they're checkpoint dirs
        if path.is_dir():
            rel_str = str(path.relative_to(root_dir)).replace('\\', '/')
            if any(fnmatch.fnmatch(rel_str, pattern) for pattern in TUNED_MODELS_EXCLUDE_DIRS):
                return True
            # Don't exclude tuned_models directories themselves, let file-level check handle it
            return False
        # For files in tuned_models, use our custom logic
        return not should_include_tuned_models_file(path)
    
    # Convert to relative path from root
    try:
        rel_path = path.relative_to(root_dir)
    except ValueError:
        return True  # Path is outside root, ignore it
    
    rel_str = str(rel_path).replace('\\', '/')
    rel_parts = rel_str.split('/')
    
    # First check if there's an explicit include pattern (negation) that matches
    # Negation patterns override exclusion patterns (gitignore rule: later patterns override earlier ones)
    # We check includes first because they have higher priority
    for pattern_type, pattern in gitignore_patterns:
        if pattern_type == 'include':
            # Handle ** wildcards (match zero or more directories)
            if '**' in pattern:
                # Convert **/* pattern to match any path structure
                # e.g., "tuned_models/**/model_registry.json" should match "tuned_models/model_1b/stage_1/v1.0/model_registry.json"
                pattern_parts = pattern.split('**')
                if len(pattern_parts) == 2:
                    prefix = pattern_parts[0].rstrip('/')
                    suffix = pattern_parts[1].lstrip('/')
                    # Check if path starts with prefix and ends with suffix
                    if rel_str.startswith(prefix) and rel_str.endswith(suffix):
                        # Check that the middle part doesn't contain excluded patterns
                        return False  # Explicitly included
                    # Also check if any part of the path matches
                    if prefix in rel_str and suffix in rel_str:
                        # Verify the order is correct (prefix before suffix)
                        prefix_idx = rel_str.find(prefix)
                        suffix_idx = rel_str.rfind(suffix)
                        if prefix_idx != -1 and suffix_idx != -1 and prefix_idx < suffix_idx:
                            return False  # Explicitly included
                else:
                    # Multiple **, use simpler matching
                    fnmatch_pattern = pattern.replace('**', '*')
                    if fnmatch.fnmatch(rel_str, fnmatch_pattern):
                        return False
            else:
                # Simple pattern matching without **
                if fnmatch.fnmatch(rel_str, pattern):
                    return False  # Explicitly included, don't ignore
                # Check if any parent directory matches the include pattern
                for i in range(len(rel_parts)):
                    partial_path = '/'.join(rel_parts[:i+1])
                    if fnmatch.fnmatch(partial_path, pattern):
                        return False  # Parent is included, don't ignore
    
    # Then check exclusion patterns
    for i in range(len(rel_parts)):
        partial_path = '/'.join(rel_parts[:i+1])
        for pattern_type, pattern in gitignore_patterns:
            if pattern_type == 'exclude':
                # Check if pattern matches
                if fnmatch.fnmatch(partial_path, pattern) or fnmatch.fnmatch(rel_str, pattern):
                    return True
                # Check directory patterns (ending with /)
                if pattern.endswith('/') and partial_path.startswith(pattern.rstrip('/')):
                    return True
                # Handle ** wildcard patterns in exclusion rules
                if '**' in pattern:
                    pattern_parts = pattern.split('**')
                    if len(pattern_parts) == 2:
                        prefix = pattern_parts[0].rstrip('/')
                        suffix = pattern_parts[1].lstrip('/')
                        if prefix and suffix:
                            if rel_str.startswith(prefix) and rel_str.endswith(suffix):
                                return True
                        elif prefix:
                            if rel_str.startswith(prefix):
                                return True
                        elif suffix:
                            if rel_str.endswith(suffix):
                                return True
                    else:
                        # Multiple **, use simpler matching
                        fnmatch_pattern = pattern.replace('**', '*')
                        if fnmatch.fnmatch(partial_path, fnmatch_pattern) or fnmatch.fnmatch(rel_str, fnmatch_pattern):
                            return True
    
    return False

def create_submission_zip(output_name=None):
    """Create a ZIP file of the codebase for submission."""
    root_dir = Path(__file__).parent.resolve()
    
    # Default output name
    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"SimRAG-Reproduction-Submission-{timestamp}.zip"
    
    output_path = root_dir / output_name
    
    # Parse .gitignore
    gitignore_path = root_dir / '.gitignore'
    gitignore_patterns = parse_gitignore(gitignore_path)
    
    print(f"Creating submission ZIP: {output_name}")
    print(f"Root directory: {root_dir}")
    print(f"Excluding: project_docs/, .github/, and .gitignore patterns")
    print(f"Including: tuned_models/ metadata (configs, tokenizers, registry) - excluding weights")
    print()
    
    files_added = 0
    files_skipped = 0
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through all files
        for root, dirs, files in os.walk(root_dir):
            root_path = Path(root)
            
            # Remove excluded directories from dirs list (modify in place)
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            
            # Skip if root itself is excluded
            rel_root = root_path.relative_to(root_dir)
            if any(excluded in rel_root.parts for excluded in EXCLUDE_DIRS):
                continue
            
            # Special handling for tuned_models: don't skip the directory, but filter files
            is_tuned_models_dir = 'tuned_models' in rel_root.parts
            
            # For non-tuned_models directories, check if should be ignored
            if not is_tuned_models_dir:
                if should_ignore(root_path, gitignore_patterns, root_dir):
                    continue
            else:
                # For tuned_models, filter out checkpoint directories
                dirs[:] = [d for d in dirs if not fnmatch.fnmatch(d, 'checkpoint-*')]
            
            for file in files:
                file_path = root_path / file
                
                # Skip the zip file itself if it exists
                if file_path == output_path:
                    continue
                
                # Skip excluded files
                if file_path.name in EXCLUDE_FILES:
                    files_skipped += 1
                    continue
                
                # Skip if in excluded directories
                if any(excluded in file_path.parts for excluded in EXCLUDE_DIRS):
                    files_skipped += 1
                    continue
                
                # Explicitly include critical files (override .gitignore)
                if file_path.name in EXPLICIT_INCLUDE_FILES:
                    try:
                        arcname = file_path.relative_to(root_dir)
                        zipf.write(file_path, arcname)
                        files_added += 1
                        if files_added % 50 == 0:
                            print(f"  Added {files_added} files...", end='\r')
                        continue
                    except Exception as e:
                        print(f"\nWarning: Could not add {file_path}: {e}")
                        files_skipped += 1
                        continue
                
                # Special handling for tuned_models files
                if is_tuned_models_file(file_path, root_dir):
                    if not should_include_tuned_models_file(file_path):
                        files_skipped += 1
                        continue
                else:
                    # For non-tuned_models files, check .gitignore patterns
                    if should_ignore(file_path, gitignore_patterns, root_dir):
                        files_skipped += 1
                        continue
                
                # Add file to zip
                try:
                    arcname = file_path.relative_to(root_dir)
                    zipf.write(file_path, arcname)
                    files_added += 1
                    if files_added % 50 == 0:
                        print(f"  Added {files_added} files...", end='\r')
                except Exception as e:
                    print(f"\nWarning: Could not add {file_path}: {e}")
                    files_skipped += 1
    
    print(f"\n\n✓ Submission ZIP created: {output_path}")
    print(f"  Files added: {files_added}")
    print(f"  Files skipped: {files_skipped}")
    print(f"  File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"\nNote: Model weights (*.safetensors, *.bin) and checkpoints were excluded.")
    print(f"      Included model metadata (configs, tokenizers, registry) for reproducibility.")
    
    return output_path

if __name__ == '__main__':
    import sys
    
    output_name = sys.argv[1] if len(sys.argv) > 1 else None
    create_submission_zip(output_name)

