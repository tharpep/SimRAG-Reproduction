"""
Create cross-platform compatible ZIP for model export
Fixes Windows backslash issue in ZIP archives for cross-platform compatibility
"""

import zipfile
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def create_cross_platform_zip(source_dir: str, output_path: Optional[str] = None) -> str:
    """
    Create ZIP with forward slashes for cross-platform compatibility
    
    Args:
        source_dir: Directory to zip (e.g., checkpoint directory)
        output_path: Output ZIP file path (optional, auto-generated if None)
        
    Returns:
        Path to the created ZIP file
        
    Raises:
        ValueError: If source directory doesn't exist
        IOError: If ZIP creation fails
    """
    source_path = Path(source_dir)

    if not source_path.exists():
        raise ValueError(f"Source directory does not exist: {source_dir}")

    # Auto-generate output name if not provided
    if output_path is None:
        # Create ZIP in the same directory as the source, with "-fixed" suffix
        output_path = str(source_path.parent / f"{source_path.name}-fixed.zip")
    
    output_path_obj = Path(output_path)

    logger.info(f"Creating ZIP: {output_path_obj.name}")
    logger.info(f"From: {source_path}")

    count = 0
    total_size = 0
    
    try:
        with zipfile.ZipFile(output_path_obj, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in source_path.rglob('*'):
                if file_path.is_file():
                    # Get relative path from source directory
                    rel_path = file_path.relative_to(source_path)
                    # Force forward slashes for cross-platform compatibility
                    arcname = str(rel_path).replace('\\', '/')

                    zf.write(file_path, arcname=arcname)
                    count += 1
                    file_size = file_path.stat().st_size
                    total_size += file_size

                    # Log important files
                    if count <= 3 or file_path.name in ['adapter_model.safetensors', 'adapter_config.json', 'added_tokens.json']:
                        size_mb = file_size / (1024 * 1024)
                        logger.info(f"  + {arcname} ({size_mb:.1f} MB)")

        size_mb = output_path_obj.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Created: {output_path_obj.name} ({size_mb:.1f} MB, {count} files)")
        logger.info(f"  Full path: {output_path_obj.resolve()}")
        
        return str(output_path_obj.resolve())
        
    except Exception as e:
        raise IOError(f"Failed to create ZIP: {e}") from e

