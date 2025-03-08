#!/usr/bin/env python3
"""
BOB AI Setup Script

This script sets up BOB AI by installing dependencies, creating necessary directories,
and downloading a default model.
"""

import os
import sys``
import subprocess
import platform
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("BOB_AI.Setup")

def check_python_version():
    """
    Check if the Python version is compatible.
    
    Returns:
        bool: True if compatible, False otherwise
    """
    required_version = (3, 8)
    current_version = sys.version_info
    
    if current_version < required_version:
        logger.error(f"Python {required_version[0]}.{required_version[1]} or higher is required")
        logger.error(f"Current version: {current_version[0]}.{current_version[1]}.{current_version[2]}")
        return False
    
    logger.info(f"Python version check passed: {current_version[0]}.{current_version[1]}.{current_version[2]}")
    return True

def install_dependencies():
    """
    Install required dependencies.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def create_directories():
    """
    Create necessary directories for BOB AI.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Creating directories...")
        
        # Define directories to create
        home_dir = os.path.expanduser("~")
        bob_ai_dir = os.path.join(home_dir, "bob_ai")
        data_dir = os.path.join(bob_ai_dir, "data")
        models_dir = os.path.join(data_dir, "models")
        logs_dir = os.path.join(bob_ai_dir, "logs")
        enhancements_dir = os.path.join(bob_ai_dir, "core", "enhancements")
        
        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(enhancements_dir, exist_ok=True)
        
        logger.info("Directories created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        return False

def download_default_model(model_name="orca-mini-3b"):
    """
    Download a default model for BOB AI.
    
    Args:
        model_name (str): Name of the model to download
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading default model: {model_name}")
        subprocess.check_call([
            sys.executable, "model_downloader.py",
            "--model", model_name,
            "--update-config"
        ])
        logger.info("Default model downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download default model: {e}")
        return False

def setup_bob_ai(skip_deps=False, skip_model=False, model_name="orca-mini-3b"):
    """
    Set up BOB AI.
    
    Args:
        skip_deps (bool): Skip dependency installation
        skip_model (bool): Skip model download
        model_name (str): Name of the model to download
    
    Returns:
        bool: True if setup was successful, False otherwise
    """
    logger.info("Starting BOB AI setup...")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not skip_deps:
        if not install_dependencies():
            return False
    else:
        logger.info("Skipping dependency installation")
    
    # Create directories
    if not create_directories():
        return False
    
    # Download default model
    if not skip_model:
        if not download_default_model(model_name):
            return False
    else:
        logger.info("Skipping model download")
    
    logger.info("BOB AI setup completed successfully!")
    return True

def main():
    """
    Main entry point for the setup script.
    """
    parser = argparse.ArgumentParser(description="BOB AI Setup Script")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-model", action="store_true", help="Skip model download")
    parser.add_argument("--model", default="orca-mini-3b", help="Model to download (default: orca-mini-3b)")
    args = parser.parse_args()
    
    success = setup_bob_ai(
        skip_deps=args.skip_deps,
        skip_model=args.skip_model,
        model_name=args.model
    )
    
    if success:
        print("\n" + "=" * 80)
        print("BOB AI setup completed successfully!")
        print("=" * 80)
        print("\nTo start BOB AI, run:")
        print("python bob_controller.py")
        print("\nFor more options, run:")
        print("python bob_controller.py --help")
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("BOB AI setup failed. Please check the logs for details.")
        print("=" * 80)
        sys.exit(1)

if __name__ == "__main__":
    main() 