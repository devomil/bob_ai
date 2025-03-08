import os
import sys
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
    
    # Print GPT4All library path
    import gpt4all
    print(f"GPT4All library path: {os.path.dirname(gpt4all.__file__)}")
    
    # Check for CUDA DLLs
    gpt4all_dir = os.path.dirname(gpt4all.__file__)
    llama_dir = os.path.join(gpt4all_dir, "llama_cpp")
    
    print("\nChecking for CUDA DLLs:")
    cuda_dlls = [
        "llamamodel-mainline-cuda.dll",
        "llamamodel-mainline-cuda-avxonly.dll",
        "llamamodel-mainline-cuda-avx2.dll"
    ]
    
    for dll in cuda_dlls:
        dll_path = os.path.join(llama_dir, dll)
        if os.path.exists(dll_path):
            print(f"✅ Found: {dll}")
        else:
            print(f"❌ Missing: {dll}")
    
    # Try to load the model with CUDA
    try:
        print("\nLoading model with CUDA...")
        model = GPT4All(model_path, device="cuda")
        print("Model loaded successfully with CUDA!")
        
        # Test the model with a simple prompt
        prompt = "What is artificial intelligence?"
        print(f"\nPrompt: {prompt}")
        
        print("Generating response...")
        with model.chat_session():
            response = model.generate(
                prompt,
                max_tokens=50,
                temp=0.7,
                top_k=40,
                top_p=0.9
            )
        
        print(f"\nResponse: {response}")
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"Error loading model with CUDA: {e}")
        
        # Try with CPU as fallback
        try:
            print("\nFalling back to CPU...")
            model = GPT4All(model_path, device="cpu")
            print("Model loaded successfully with CPU!")
        except Exception as e:
            print(f"Error loading model with CPU: {e}")

if __name__ == "__main__":
    main() 