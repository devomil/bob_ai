from llama_cpp import Llama

# Replace with your correct model path
MODEL_PATH = "C:/Users/Ryan/bob_ai/data/models/mistral-7b-instruct-v0.2.Q4_0.gguf"

print("🚀 Loading model...")
llm = Llama(model_path=MODEL_PATH, n_gpu_layers=35)

response = llm("Tell me a joke")
print("🤖 Llama Response:", response)
