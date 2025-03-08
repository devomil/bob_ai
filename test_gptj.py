from transformers import AutoModelForCausalLM, AutoTokenizer

# Correct path to the local model directory
model_path = r"C:\Users\Ryan\bob_ai\local_models\EleutherAI_gpt-j-6b"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

# Ensure pad_token_id is set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id  # Use EOS token as the padding token

# Refined prompt
input_text = (
    "Write only the Python function code for a recursive function named `factorial` that calculates the factorial of a number."
)
input_ids = tokenizer.encode(input_text, return_tensors="pt")
attention_mask = (input_ids != tokenizer.pad_token_id).int()  # Fix attention mask logic

# Generate output
output = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    pad_token_id=tokenizer.pad_token_id,
    max_length=150,  # Allow room for complete response
    temperature=0.3,  # Controlled sampling
    do_sample=True,
    top_k=10,
)

# Decode the generated output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Raw Generated Output:")
print(generated_text)

# Define the post-processing function
def extract_valid_python_code(text):
    lines = text.split("\n")
    code_lines = []
    inside_function = False

    for line in lines:
        stripped_line = line.strip()

        # Ignore all lines until we find a valid function definition
        if stripped_line.startswith("def factorial("):
            inside_function = True
            code_lines.append(line)  # Start capturing function lines
            continue

        # Capture valid indented lines inside the function
        if inside_function:
            if stripped_line.startswith("return") or stripped_line.startswith("if") or stripped_line.startswith("else") or stripped_line.startswith("    "):
                code_lines.append(line)
            elif not stripped_line:  # Stop capturing on empty lines
                break

    # Return the captured function block or an error
    if code_lines:
        return "\n".join(code_lines).strip()
    else:
        return "Error: Incomplete or invalid Python function generated."

# Process the raw output
cleaned_output = extract_valid_python_code(generated_text)

print("Processed Output:")
print(cleaned_output)
