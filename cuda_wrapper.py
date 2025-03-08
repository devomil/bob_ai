"""
CUDA Wrapper for GPT4All

This module provides a wrapper for GPT4All that uses llama-cpp-python with CUDA support.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from contextlib import contextmanager

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("Warning: llama-cpp-python not available. CUDA acceleration will not work.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("BOB_AI.CUDAWrapper")

class CudaLLM:
    """
    A wrapper for llama-cpp-python with CUDA support.
    """
    
    def __init__(self, model_path: str, device: str = "cuda", n_ctx: int = 2048, verbose: bool = False):
        """
        Initialize the CudaLLM.
        
        Args:
            model_path (str): Path to the model file.
            device (str): Device to use ("cuda" or "cpu").
            n_ctx (int): Context size.
            verbose (bool): Whether to show detailed logs.
        """
        self.model_path = model_path
        self.device = device
        self.n_ctx = n_ctx
        self.verbose = verbose
        self.model = None
        self.current_chat_session = []
        
        if not LLAMA_CPP_AVAILABLE:
            logger.error("llama-cpp-python not available. CUDA acceleration will not work.")
            return
        
        self._load_model()
    
    def _load_model(self):
        """
        Load the model.
        """
        if not LLAMA_CPP_AVAILABLE:
            return
        
        try:
            n_gpu_layers = -1 if self.device == "cuda" else 0
            
            logger.info(f"Loading model from {self.model_path} with {self.device} support...")
            
            # CUDA-specific optimizations
            if self.device == "cuda":
                self.model = Llama(
                    model_path=self.model_path,
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=self.n_ctx,
                    verbose=self.verbose,
                    n_batch=512,  # Increase batch size for better GPU utilization
                    offload_kqv=True,  # Offload key/query/value tensors to GPU
                    f16_kv=True,  # Use half-precision for key/value cache
                    use_mlock=True  # Lock memory to prevent swapping
                )
            else:
                self.model = Llama(
                    model_path=self.model_path,
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=self.n_ctx,
                    verbose=self.verbose
                )
            
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            
            # Try with CPU as fallback if CUDA failed
            if self.device == "cuda":
                logger.info("Falling back to CPU...")
                self.device = "cpu"
                self._load_model()
    
    @contextmanager
    def chat_session(self):
        """
        Context manager for chat sessions.
        """
        self.current_chat_session = []
        try:
            yield
        finally:
            pass
    
    def generate(self, prompt: str, max_tokens: int = 200, temp: float = 0.7, 
                top_k: int = 40, top_p: float = 0.9, repeat_penalty: float = 1.1,
                streaming: bool = False) -> Union[str, Any]:
        """
        Generate text from a prompt.
        
        Args:
            prompt (str): The prompt to generate from.
            max_tokens (int): Maximum number of tokens to generate.
            temp (float): Temperature for sampling.
            top_k (int): Top-k sampling.
            top_p (float): Top-p sampling.
            repeat_penalty (float): Repeat penalty.
            streaming (bool): Whether to stream the output.
            
        Returns:
            str: The generated text.
        """
        if not LLAMA_CPP_AVAILABLE or self.model is None:
            return "Error: Model not loaded."
        
        try:
            # Format the prompt with the chat history
            full_prompt = self._format_prompt(prompt)
            
            # Generate the response
            if streaming:
                return self._generate_streaming(full_prompt, max_tokens, temp, top_k, top_p, repeat_penalty)
            else:
                # Use a larger batch size for better GPU utilization
                output = self.model(
                    full_prompt,
                    max_tokens=max_tokens,
                    temperature=temp,
                    top_k=top_k,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    echo=False
                )
                
                response = output['choices'][0]['text'].strip()
                
                # Add the response to the chat history
                self.current_chat_session.append({
                    "role": "assistant",
                    "content": response
                })
                
                return response
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error: {str(e)}"
    
    def _generate_streaming(self, prompt: str, max_tokens: int, temp: float, 
                           top_k: int, top_p: float, repeat_penalty: float) -> str:
        """
        Generate text from a prompt with streaming.
        
        Args:
            prompt (str): The prompt to generate from.
            max_tokens (int): Maximum number of tokens to generate.
            temp (float): Temperature for sampling.
            top_k (int): Top-k sampling.
            top_p (float): Top-p sampling.
            repeat_penalty (float): Repeat penalty.
            
        Returns:
            str: The generated text.
        """
        response = ""
        
        for output in self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            echo=False,
            stream=True
        ):
            token = output['choices'][0]['text']
            response += token
            yield token
        
        # Add the response to the chat history
        self.current_chat_session.append({
            "role": "assistant",
            "content": response
        })
        
        return response
    
    def _format_prompt(self, prompt: str) -> str:
        """
        Format the prompt with the chat history.
        
        Args:
            prompt (str): The prompt to format.
            
        Returns:
            str: The formatted prompt.
        """
        # Add the user message to the chat history
        self.current_chat_session.append({
            "role": "user",
            "content": prompt
        })
        
        # Format the prompt with the Mistral template
        formatted_prompt = ""
        
        for message in self.current_chat_session:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted_prompt += f"{content}\n"
            elif role == "user":
                formatted_prompt += f"[INST] {content} [/INST]\n"
            elif role == "assistant":
                formatted_prompt += f"{content}\n"
        
        return formatted_prompt
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Generate a chat completion.
        
        Args:
            messages (List[Dict[str, str]]): List of messages.
            **kwargs: Additional arguments.
            
        Returns:
            Dict[str, Any]: The chat completion.
        """
        if not LLAMA_CPP_AVAILABLE or self.model is None:
            return {"choices": [{"message": {"content": "Error: Model not loaded."}}]}
        
        # Reset the chat session
        self.current_chat_session = []
        
        # Add all messages to the chat session
        for message in messages:
            self.current_chat_session.append(message)
        
        # Get the last user message
        last_user_message = None
        for message in reversed(messages):
            if message["role"] == "user":
                last_user_message = message["content"]
                break
        
        if last_user_message is None:
            return {"choices": [{"message": {"content": "Error: No user message found."}}]}
        
        # Generate the response
        response = self.generate(last_user_message, **kwargs)
        
        # Format the response as a chat completion
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response
                    }
                }
            ]
        }


def load_model_from_config(config_path: Optional[str] = None, device: str = "cuda") -> CudaLLM:
    """
    Load a model from the configuration file.
    
    Args:
        config_path (str, optional): Path to the configuration file.
        device (str): Device to use ("cuda" or "cpu").
        
    Returns:
        CudaLLM: The loaded model.
    """
    # Default model path
    model_path = None
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                if "model_path" in config:
                    model_path = config["model_path"]
                    logger.info(f"Using model path from config: {model_path}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    
    if not model_path:
        # Try to find the model in the default location
        default_model_dir = os.path.join(os.path.expanduser("~"), "bob_ai", "data", "models")
        for filename in os.listdir(default_model_dir) if os.path.exists(default_model_dir) else []:
            if filename.endswith(".gguf"):
                model_path = os.path.join(default_model_dir, filename)
                logger.info(f"Using model found at: {model_path}")
                break
    
    if not model_path:
        logger.error("No model path found in config or default location.")
        return None
    
    # Load the model
    return CudaLLM(model_path, device=device)


if __name__ == "__main__":
    # Test the wrapper
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    model = load_model_from_config(config_path)
    
    if model and model.model:
        with model.chat_session():
            response = model.generate("What is artificial intelligence?")
            print(f"Response: {response}")
    else:
        print("Error: Model not loaded.") 