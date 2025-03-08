import os
import sys
import json
from llama_cpp import Llama

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
    
    # Try to load the model with CUDA
    try:
        print("\nLoading model with CUDA...")
        model = Llama(
            model_path=model_path,
            n_gpu_layers=-1,  # Use all layers on GPU
            n_ctx=2048,       # Context size
            verbose=True      # Show detailed logs
        )
        print("Model loaded successfully with CUDA!")
        
        # Test the model with a simple prompt
        prompt = "What is artificial intelligence?"
        print(f"\nPrompt: {prompt}")
        
        print("Generating response...")
        output = model(
            prompt,
            max_tokens=50,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            echo=True
        )
        
        print(f"\nResponse: {output['choices'][0]['text']}")
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"Error loading model with CUDA: {e}")
        
        # Try with CPU as fallback
        try:
            print("\nFalling back to CPU...")
            model = Llama(
                model_path=model_path,
                n_gpu_layers=0,  # Use CPU only
                n_ctx=2048,
                verbose=True
            )
            print("Model loaded successfully with CPU!")
        except Exception as e:
            print(f"Error loading model with CPU: {e}")

if __name__ == "__main__":
    main() 