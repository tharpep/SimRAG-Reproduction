"""
Model Utilities
Utility functions for model discovery and export
Extracted from test_model.py for reuse across the codebase
"""

from pathlib import Path
from typing import Optional, List, Dict, Any

from ..config import get_tuning_config
from ..tuning.model_registry import get_model_registry
from ..logging_config import get_logger

logger = get_logger(__name__)


def list_available_models(stage: str, model_size: str = "small") -> List[Dict[str, Any]]:
    """
    List all available models for a given stage, including checkpoints
    
    Args:
        stage: Stage name ("stage_1" or "stage_2")
        model_size: Model size ("small" or "medium")
        
    Returns:
        List of model info dictionaries with version, path, checkpoint info, and metadata
    """
    model_suffix = "1b" if model_size == "small" else "8b"
    stage_dir = Path(f"./tuned_models/model_{model_suffix}/{stage}")
    
    if not stage_dir.exists():
        return []
    
    models = []
    config = get_tuning_config()
    config.model_size = model_size
    registry = get_model_registry(config)
    
    # Get all versions from registry
    all_versions = registry.get_all_versions()
    version_metadata = {v.version: v for v in all_versions}
    
    # Scan all version directories
    for model_dir in stage_dir.iterdir():
        if model_dir.is_dir() and model_dir.name.startswith("v"):
            version = model_dir.name
            version_info = version_metadata.get(version)
            
            # Check for adapter files in version directory and checkpoints
            adapter_files = list(model_dir.glob("**/adapter_model.safetensors")) + \
                           list(model_dir.glob("**/adapter_model.bin"))
            
            if adapter_files:
                # Group by checkpoint directory
                checkpoint_paths = {}
                version_path = None
                
                for adapter_file in adapter_files:
                    parent = adapter_file.parent
                    if "checkpoint" in parent.name:
                        checkpoint_paths[parent.name] = str(parent)
                    elif parent == model_dir:
                        version_path = str(parent)
                
                # Add version-level model if exists
                if version_path:
                    models.append({
                        "version": version,
                        "checkpoint": None,
                        "path": version_path,
                        "display_name": f"{version} (final)",
                        "created_at": version_info.created_at if version_info else "Unknown",
                        "training_time": version_info.training_time_seconds if version_info else None,
                        "final_loss": version_info.final_loss if version_info else None,
                        "notes": version_info.notes if version_info else None,
                        "experiment_run_id": version_info.experiment_run_id if version_info else None
                    })
                
                # Add checkpoint models
                for checkpoint_name, checkpoint_path in sorted(checkpoint_paths.items()):
                    # Extract step number from checkpoint name (e.g., "checkpoint-500" -> 500)
                    step_num = None
                    try:
                        step_num = int(checkpoint_name.split("-")[-1])
                    except (ValueError, IndexError):
                        pass
                    
                    models.append({
                        "version": version,
                        "checkpoint": checkpoint_name,
                        "path": checkpoint_path,
                        "display_name": f"{version} ({checkpoint_name})",
                        "step": step_num,
                        "created_at": version_info.created_at if version_info else "Unknown",
                        "training_time": version_info.training_time_seconds if version_info else None,
                        "final_loss": version_info.final_loss if version_info else None,
                        "notes": version_info.notes if version_info else None,
                        "experiment_run_id": version_info.experiment_run_id if version_info else None
                    })
    
    # Sort by version, then by checkpoint step (checkpoints first, then final)
    def sort_key(m):
        try:
            version_num = float(m["version"][1:])
        except (ValueError, IndexError):
            version_num = 0.0
        
        # Sort: version (desc), then checkpoint step (desc, None last)
        checkpoint_step = m.get("step") if m.get("checkpoint") else float('inf')
        return (-version_num, checkpoint_step)
    
    models.sort(key=sort_key)
    return models


def export_model(
    model_path: str,
    stage: str,
    output_name: Optional[str] = None
) -> str:
    """
    Export a model to a cross-platform ZIP file for Colab
    
    Args:
        model_path: Path to the model directory (checkpoint directory)
        stage: Stage name ("stage_1" or "stage_2")
        output_name: Optional output ZIP filename (auto-generated if None)
        
    Returns:
        Path to the created ZIP file
    """
    logger.info(f"=== Exporting Model: {model_path} ===")
    
    model_path_abs = Path(model_path).resolve()
    
    if not model_path_abs.exists():
        raise ValueError(f"Model path does not exist: {model_path_abs}")
    
    # Verify it's a valid checkpoint/adapter directory
    adapter_files = list(model_path_abs.glob("adapter_model.*"))
    if not adapter_files:
        raise ValueError(f"No adapter files found in {model_path_abs}. Expected adapter_model.safetensors or adapter_model.bin")
    
    # Generate output name if not provided
    if output_name is None:
        # Use checkpoint name with "-fixed" suffix (matching Colab workflow)
        output_name = f"{model_path_abs.name}-fixed.zip"
    
    # Create ZIP in the model's parent directory (same directory as checkpoint)
    output_path = model_path_abs.parent / output_name
    
    # Import and use the zip export utility
    from ..tuning.zip_export import create_cross_platform_zip
    
    zip_path = create_cross_platform_zip(
        source_dir=str(model_path_abs),
        output_path=str(output_path)
    )
    
    logger.info(f"✓ Model exported successfully: {zip_path}")
    return zip_path

