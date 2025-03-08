"""
Test script for the CUDA wrapper.
"""

import os
import sys
import time
from cuda_wrapper import load_model_from_config

def main():
    """
    Test the CUDA wrapper.
    """
    print("Testing CUDA wrapper...")
    
    # Load the model
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    
    # Test with CUDA
    print("\n=== Testing with CUDA ===")
    model_cuda = load_model_from_config(config_path, device="cuda")
    
    if model_cuda and model_cuda.model:
        print("Model loaded successfully with CUDA!")
        
        # Test the model with a simple prompt
        prompt = "What is artificial intelligence?"
        print(f"\nPrompt: {prompt}")
        
        # Measure time
        start_time = time.time()
        
        print("Generating response...")
        with model_cuda.chat_session():
            response = model_cuda.generate(
                prompt,
                max_tokens=100,
                temp=0.7,
                top_k=40,
                top_p=0.9
            )
        
        end_time = time.time()
        cuda_time = end_time - start_time
        
        print(f"\nResponse: {response}")
        print(f"Generation time with CUDA: {cuda_time:.2f} seconds")
    else:
        print("Error: Model not loaded with CUDA.")
    
    # Test with CPU
    print("\n=== Testing with CPU ===")
    model_cpu = load_model_from_config(config_path, device="cpu")
    
    if model_cpu and model_cpu.model:
        print("Model loaded successfully with CPU!")
        
        # Test the model with the same prompt
        prompt = "What is artificial intelligence?"
        print(f"\nPrompt: {prompt}")
        
        # Measure time
        start_time = time.time()
        
        print("Generating response...")
        with model_cpu.chat_session():
            response = model_cpu.generate(
                prompt,
                max_tokens=100,
                temp=0.7,
                top_k=40,
                top_p=0.9
            )
        
        end_time = time.time()
        cpu_time = end_time - start_time
        
        print(f"\nResponse: {response}")
        print(f"Generation time with CPU: {cpu_time:.2f} seconds")
        
        # Compare performance
        if 'cuda_time' in locals():
            speedup = cpu_time / cuda_time
            print(f"\nCUDA speedup: {speedup:.2f}x faster than CPU")
    else:
        print("Error: Model not loaded with CPU.")

if __name__ == "__main__":
    main() 