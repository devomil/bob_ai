#!/usr/bin/env python3
"""
BOB AI Model Downloader

This script downloads AI models for use with BOB AI.
It supports various model types and sources including GPT4All, Hugging Face, and more.
"""

import os
import sys
import json
import argparse
import requests
import hashlib
from tqdm import tqdm
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("BOB_AI.ModelDownloader")

# Default model directory
DEFAULT_MODEL_DIR = os.path.join(os.path.expanduser("~"), "bob_ai", "data", "models")

# Model information
AVAILABLE_MODELS = {
    "mistral-7b-instruct-v0.1": {
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        "size": 4_100_000_000,  # Approximate size in bytes
        "description": "Mistral 7B Instruct v0.1 (Q4_K_M quantization)",
        "type": "huggingface",
        "family": "mistral",
        "quantization": "Q4_K_M"
    },
    "mistral-7b-instruct-v0.2": {
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "size": 4_370_000_000,  # Approximate size in bytes
        "description": "Mistral 7B Instruct v0.2 (Q4_K_M quantization)",
        "type": "huggingface",
        "family": "mistral",
        "quantization": "Q4_K_M"
    },
    "llama-2-7b-chat": {
        "url": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
        "size": 4_100_000_000,  # Approximate size in bytes
        "description": "Llama 2 7B Chat (Q4_K_M quantization)",
        "type": "huggingface",
        "family": "llama",
        "quantization": "Q4_K_M"
    },
    "orca-mini-3b": {
        "url": "https://huggingface.co/TheBloke/orca_mini_3B-GGUF/resolve/main/orca-mini-3b.Q4_K_M.gguf",
        "size": 2_000_000_000,  # Approximate size in bytes
        "description": "Orca Mini 3B (Q4_K_M quantization)",
        "type": "huggingface",
        "family": "orca",
        "quantization": "Q4_K_M"
    }
}

def list_available_models():
    """
    List all available models with their descriptions.
    """
    print("\nAvailable Models for BOB AI:")
    print("=" * 80)
    print(f"{'Model Name':<25} {'Size':<10} {'Type':<10} {'Description':<35}")
    print("-" * 80)
    
    for model_name, model_info in AVAILABLE_MODELS.items():
        size_gb = model_info["size"] / 1_000_000_000
        print(f"{model_name:<25} {size_gb:.1f} GB   {model_info['type']:<10} {model_info['description']:<35}")
    
    print("=" * 80)
    print("\nTo download a model, use: python model_downloader.py --model MODEL_NAME")
    print("Example: python model_downloader.py --model mistral-7b-instruct-v0.2\n")

def download_file(url, destination, expected_size=None):
    """
    Download a file with progress bar.
    
    Args:
        url (str): URL to download from
        destination (str): Path to save the file
        expected_size (int, optional): Expected file size in bytes
    
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size from headers or use expected_size
        total_size = int(response.headers.get('content-length', 0)) or expected_size or 0
        block_size = 1024 * 1024  # 1 MB chunks
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        with open(destination, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(destination)) as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        return True
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return False

def verify_file(file_path, expected_hash=None):
    """
    Verify file integrity using SHA-256 hash.
    
    Args:
        file_path (str): Path to the file
        expected_hash (str, optional): Expected SHA-256 hash
    
    Returns:
        bool: True if verification passed or no hash provided, False otherwise
    """
    if not expected_hash:
        logger.info("No hash provided for verification, skipping")
        return True
    
    try:
        logger.info(f"Verifying file integrity: {os.path.basename(file_path)}")
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        file_hash = sha256_hash.hexdigest()
        
        if file_hash == expected_hash:
            logger.info("File verification passed")
            return True
        else:
            logger.error(f"File verification failed. Expected: {expected_hash}, Got: {file_hash}")
            return False
    except Exception as e:
        logger.error(f"Error verifying file: {e}")
        return False

def download_model(model_name, output_dir=DEFAULT_MODEL_DIR, force=False):
    """
    Download a model by name.
    
    Args:
        model_name (str): Name of the model to download
        output_dir (str): Directory to save the model
        force (bool): Force download even if file exists
    
    Returns:
        str: Path to the downloaded model or None if download failed
    """
    if model_name not in AVAILABLE_MODELS:
        logger.error(f"Model '{model_name}' not found. Use --list to see available models.")
        return None
    
    model_info = AVAILABLE_MODELS[model_name]
    model_url = model_info["url"]
    model_filename = os.path.basename(model_url)
    model_path = os.path.join(output_dir, model_filename)
    
    # Check if model already exists
    if os.path.exists(model_path) and not force:
        logger.info(f"Model already exists at {model_path}")
        
        # Verify existing file if hash is available
        if "sha256" in model_info and verify_file(model_path, model_info.get("sha256")):
            logger.info("Existing model file is valid")
            return model_path
        else:
            if "sha256" in model_info:
                logger.warning("Existing model file verification failed, redownloading")
            else:
                logger.info("No hash available for verification, using existing file")
                return model_path
    
    # Download the model
    logger.info(f"Downloading model {model_name} from {model_url}")
    success = download_file(model_url, model_path, model_info.get("size"))
    
    if success:
        logger.info(f"Model downloaded successfully to {model_path}")
        
        # Verify downloaded file if hash is available
        if "sha256" in model_info and not verify_file(model_path, model_info.get("sha256")):
            logger.error("Downloaded model failed verification")
            return None
        
        return model_path
    else:
        logger.error("Failed to download model")
        return None

def update_config(model_path):
    """
    Update the BOB AI configuration to use the downloaded model.
    
    Args:
        model_path (str): Path to the downloaded model
    """
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            config["model_path"] = model_path
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            logger.info(f"Updated configuration to use model: {model_path}")
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
    else:
        logger.warning(f"Configuration file not found at {config_path}")

def main():
    """
    Main entry point for the model downloader.
    """
    parser = argparse.ArgumentParser(description="BOB AI Model Downloader")
    parser.add_argument("--model", help="Name of the model to download")
    parser.add_argument("--output", help=f"Output directory (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--force", action="store_true", help="Force download even if file exists")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--update-config", action="store_true", help="Update config.json with the downloaded model path")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = args.output or DEFAULT_MODEL_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    if args.list:
        list_available_models()
        return
    
    if not args.model:
        parser.print_help()
        return
    
    # Download the model
    model_path = download_model(args.model, output_dir, args.force)
    
    if model_path and args.update_config:
        update_config(model_path)

if __name__ == "__main__":
    main()
