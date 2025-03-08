from transformers import AutoModelForCausalLM, AutoTokenizer

# Use the correct path to the local model directory
model_path = r"C:\Users\Ryan\bob_ai\local_models\EleutherAI_gpt-j-6B"

try:
    # Attempt to load tokenizer and model from the local directory
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    print("Model and tokenizer loaded successfully!")
except Exception as e:
    # Print the error if loading fails
    print("Error loading model or tokenizer:", e)
