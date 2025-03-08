import os
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer


class Enhancer:
    def __init__(self, model_path):
        """
        Initialize the Enhancer with the path to the GPT-J model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

        # Ensure pad_token_id is set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Set the enhancements directory
        self.enhancements_dir = r"C:\Users\Ryan\bob_ai\core\enhancements"
        os.makedirs(self.enhancements_dir, exist_ok=True)

    def generate_code(self, description):
        """
        Generate Python code based on a natural language description.
        """
        prompt = f"Write only the Python function code for {description}."
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        attention_mask = (input_ids != self.tokenizer.pad_token_id).int()

        # Generate output
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=200,
            temperature=0.3,
            do_sample=True,
            top_k=10,
        )
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

    def extract_valid_code(self, text):
        """
        Extract valid Python function code from the generated text.
        """
        lines = text.split("\n")
        code_lines = []
        inside_function = False

        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith("def "):
                inside_function = True
                code_lines.append(line)
                continue
            if inside_function:
                if stripped_line.startswith("    "):  # Capture indented lines
                    code_lines.append(line)
                elif not stripped_line:  # Stop capturing on empty lines
                    break

        return "\n".join(code_lines).strip()

    def test_code(self, code):
        """
        Test the generated Python code to ensure it works correctly.
        """
        temp_file = "temp_test_code.py"
        with open(temp_file, "w") as f:
            f.write(code)

        try:
            result = subprocess.run(
                ["python", temp_file],
                capture_output=True,
                text=True,
                check=True,
            )
            os.remove(temp_file)
            return f"Code executed successfully: {result.stdout}"
        except subprocess.CalledProcessError as e:
            os.remove(temp_file)
            return f"Code execution failed: {e.stderr}"

    def enhance_bob(self, description):
        """
        Generate and validate an enhancement for Bob.
        """
        print(f"Generating code for: {description}")
        raw_code = self.generate_code(description)
        print(f"Raw Generated Code:\n{raw_code}")

        cleaned_code = self.extract_valid_code(raw_code)
        print(f"Extracted Code:\n{cleaned_code}")

        if not cleaned_code:
            return "Failed to generate valid code."

        test_result = self.test_code(cleaned_code)
        print(f"Test Result:\n{test_result}")

        if "successfully" in test_result.lower():
            # Save the enhancement if successful
            enhancement_file = os.path.join(
                self.enhancements_dir, f"{description.replace(' ', '_')}.py"
            )
            with open(enhancement_file, "w") as f:
                f.write(cleaned_code)
            return f"Enhancement saved to {enhancement_file}"
        else:
            return "Enhancement failed validation."

