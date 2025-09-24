#!/usr/bin/env python3
"""
Quick Start Script for SimRAG Reproduction Project

This script provides a simple way to run the experiments and see results.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from loguru import logger


def setup_logging():
    """Setup logging for the quick start script."""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )


def run_command(command: str, description: str) -> bool:
    """
    Run a command and return success status.
    
    Args:
        command: Command to run
        description: Description of what the command does
        
    Returns:
        True if command succeeded, False otherwise
    """
    logger.info(f"Running: {description}")
    logger.info(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.success(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed")
        logger.error(f"Error: {e.stderr}")
        return False


def check_requirements():
    """Check if required tools are installed."""
    logger.info("Checking requirements...")
    
    requirements = [
        ("python", "Python interpreter"),
        ("conda", "Conda package manager"),
        ("git", "Git version control")
    ]
    
    all_good = True
    for cmd, desc in requirements:
        try:
            subprocess.run([cmd, "--version"], capture_output=True, check=True)
            logger.success(f"✓ {desc} is installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error(f"✗ {desc} is not installed")
            all_good = False
    
    return all_good


def setup_environment():
    """Setup the conda environment."""
    logger.info("Setting up conda environment...")
    
    # Check if environment already exists
    try:
        result = subprocess.run(
            ["conda", "env", "list"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        if "simrag-reproduction" in result.stdout:
            logger.info("Environment already exists, activating...")
            return True
    except subprocess.CalledProcessError:
        pass
    
    # Create environment
    if not run_command("conda env create -f environment.yml", "Creating conda environment"):
        return False
    
    return True


def run_baseline_experiment():
    """Run the baseline RAG experiment."""
    logger.info("Running baseline RAG experiment...")
    
    command = "python experiments/run_baseline.py --config config/baseline_config.yaml --output ./results/baseline"
    return run_command(command, "Baseline RAG experiment")


def run_simrag_experiment():
    """Run the SimRAG experiment."""
    logger.info("Running SimRAG experiment...")
    
    command = "python experiments/run_simrag.py --config config/simrag_config.yaml --output ./results/simrag"
    return run_command(command, "SimRAG experiment")


def compare_results():
    """Compare baseline and SimRAG results."""
    logger.info("Comparing results...")
    
    command = "python experiments/compare_results.py --baseline ./results/baseline --simrag ./results/simrag --output ./results/comparison"
    return run_command(command, "Results comparison")


def main():
    """Main quick start function."""
    parser = argparse.ArgumentParser(description="Quick start script for SimRAG reproduction")
    parser.add_argument("--setup-only", action="store_true", help="Only setup environment, don't run experiments")
    parser.add_argument("--baseline-only", action="store_true", help="Only run baseline experiment")
    parser.add_argument("--simrag-only", action="store_true", help="Only run SimRAG experiment")
    parser.add_argument("--compare-only", action="store_true", help="Only compare existing results")
    parser.add_argument("--skip-setup", action="store_true", help="Skip environment setup")
    
    args = parser.parse_args()
    
    setup_logging()
    
    logger.info("🚀 SimRAG Reproduction Project - Quick Start")
    logger.info("=" * 50)
    
    # Check requirements
    if not check_requirements():
        logger.error("❌ Requirements check failed. Please install missing tools.")
        return 1
    
    # Setup environment
    if not args.skip_setup:
        if not setup_environment():
            logger.error("❌ Environment setup failed.")
            return 1
    
    if args.setup_only:
        logger.success("✅ Setup completed successfully!")
        return 0
    
    # Run experiments
    success = True
    
    if args.baseline_only:
        success = run_baseline_experiment()
    elif args.simrag_only:
        success = run_simrag_experiment()
    elif args.compare_only:
        success = compare_results()
    else:
        # Run all experiments
        logger.info("Running full experiment pipeline...")
        
        if not run_baseline_experiment():
            success = False
        
        if not run_simrag_experiment():
            success = False
        
        if success and not compare_results():
            success = False
    
    if success:
        logger.success("🎉 All experiments completed successfully!")
        logger.info("📊 Check the results in the ./results/ directory")
        logger.info("📈 View comparison plots in ./results/comparison/")
        return 0
    else:
        logger.error("❌ Some experiments failed. Check the logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
