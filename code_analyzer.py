import re

class CodeAnalyzer:
    def analyze_code(self, code: str):
        """
        Analyze the given code to detect language, functions, and classes.

        Args:
            code (str): The source code to analyze.

        Returns:
            dict: A summary of the analysis.
        """
        # Detect language
        language = self.detect_language(code)

        # Extract classes and functions
        classes = self.extract_classes(code)
        functions = self.extract_functions(code)

        return {
            "language": language,
            "classes": classes,
            "functions": functions
        }

    def detect_language(self, code: str) -> str:
        """A simplified language detector."""
        if "def " in code and "class " in code:
            return "python"
        elif "function " in code or "var " in code:
            return "javascript"
        return "unknown"

    def extract_classes(self, code: str) -> list:
        """Extract class names from the code."""
        return re.findall(r'class\s+(\w+)', code)

    def extract_functions(self, code: str) -> list:
        """Extract function names from the code."""
        return re.findall(r'def\s+(\w+)', code)
