#!/usr/bin/env python3
"""
Simple test script to verify that the model works correctly.
"""

import os
import json
from gpt4all import GPT4All

def main():
    # Load config if available to get model path
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    model_path = None
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                if "model_path" in config:
                    model_path = config["model_path"]
                    print(f"Using model path from config: {model_path}")
        except Exception as e:
            print(f"Error loading config: {e}")
    
    if not model_path:
        model_path = r"C:\Users\Ryan\bob_ai\data\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        print(f"Using default model path: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    try:
        print("Loading model...")
        model = GPT4All(model_path)
        print("Model loaded successfully!")
        
        # Test the model with a simple prompt
        prompt = "What is artificial intelligence?"
        print(f"\nPrompt: {prompt}")
        
        print("Generating response...")
        with model.chat_session():
            response = model.generate(
                prompt,
                max_tokens=200,
                temp=0.7,
                top_k=40,
                top_p=0.9
            )
        
        print(f"\nResponse: {response}")
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 