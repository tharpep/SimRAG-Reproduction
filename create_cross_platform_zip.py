"""
DEPRECATED: This script is kept for backwards compatibility.
Use `simrag experiment export` instead, or import from simrag_reproduction.tuning.zip_export

Create cross-platform compatible ZIP for Colab
Fixes Windows backslash issue in ZIP archives
"""

import sys
from pathlib import Path

# Try to use the new module if available
try:
    from simrag_reproduction.tuning.zip_export import create_cross_platform_zip
    
    def create_zip(source_dir: str, output_name: str = None):
        """Create ZIP with forward slashes for cross-platform compatibility"""
        try:
            zip_path = create_cross_platform_zip(source_dir, output_name)
            print(f"\n✓ Success! Upload this ZIP to Colab.")
            return True
        except Exception as e:
            print(f"\n✗ Failed: {e}")
            return False
except ImportError:
    # Fallback to old implementation if module not available
    import zipfile
    
    def create_zip(source_dir: str, output_name: str = None):
        """Create ZIP with forward slashes for cross-platform compatibility"""
        source_path = Path(source_dir)

        if not source_path.exists():
            print(f"ERROR: {source_dir} not found")
            return False

        if output_name is None:
            output_name = f"{source_path.name}.zip"

        output_path = Path(output_name)

        print(f"Creating ZIP: {output_path.name}")
        print(f"From: {source_path}")

        count = 0
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in source_path.rglob('*'):
                if file_path.is_file():
                    # Get relative path and force forward slashes
                    rel_path = file_path.relative_to(source_path.parent)
                    arcname = str(rel_path).replace('\\', '/')

                    zf.write(file_path, arcname=arcname)
                    count += 1

                    if count <= 3 or file_path.name in ['adapter_model.safetensors', 'adapter_config.json']:
                        size = file_path.stat().st_size / (1024 * 1024)
                        print(f"  + {arcname} ({size:.1f} MB)")

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\nCreated: {output_path.name} ({size_mb:.1f} MB, {count} files)")
        print(f"Full path: {output_path.resolve()}")

        return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_cross_platform_zip.py <directory> [output.zip]")
        print("\nExamples:")
        print("  python create_cross_platform_zip.py tuned_models/model_1b/stage_1/v1.8/checkpoint-1000")
        print("  python create_cross_platform_zip.py data/documents documents.zip")
        print("\nNOTE: Consider using 'simrag experiment export' instead for better integration.")
        sys.exit(1)

    source = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None

    if create_zip(source, output):
        print("\n✓ Success! Upload this ZIP to Colab.")
    else:
        print("\n✗ Failed")
        sys.exit(1)
