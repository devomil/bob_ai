import os

class CodeInteractionModule:
    """
    A module that allows BOB to interact with its own code.
    This includes reading, modifying, and executing Python files.
    """
    def __init__(self, base_path):
        """
        Initialize the module with the base path for source files.
        Args:
            base_path (str): The base directory for the project.
        """
        self.base_path = base_path

    def read_file(self, file_name):
        """
        Read the contents of a given file.
        Args:
            file_name (str): The name of the file to read.
        Returns:
            str: The contents of the file or an error message if the file does not exist.
        """
        file_path = os.path.join(self.base_path, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                return file.read()
        return f"File {file_name} not found."

    def modify_file(self, file_name, search_text, replace_text):
        """
        Modify a file by searching for specific text and replacing it.
        Args:
            file_name (str): The name of the file to modify.
            search_text (str): The text to search for in the file.
            replace_text (str): The text to replace the search text with.
        Returns:
            str: A success or error message.
        """
        file_path = os.path.join(self.base_path, file_name)
        if os.path.exists(file_path):
            # Read the file
            with open(file_path, 'r') as file:
                content = file.read()

            # Check if the search text exists
            if search_text in content:
                # Replace the text and save the file
                updated_content = content.replace(search_text, replace_text)
                with open(file_path, 'w') as file:
                    file.write(updated_content)
                return f"Successfully updated {file_name}."
            else:
                return f"Text '{search_text}' not found in {file_name}."
        return f"File {file_name} not found."

    def execute_file(self, file_name):
        """
        Execute a Python file.
        Args:
            file_name (str): The name of the file to execute.
        Returns:
            str: A success or error message.
        """
        file_path = os.path.join(self.base_path, file_name)
        if os.path.exists(file_path):
            os.system(f'python {file_path}')
            return f"Executed {file_name}."
        return f"File {file_name} not found."

    def backup_file(self, file_name):
        """
        Create a backup of a file before making changes.
        Args:
            file_name (str): The name of the file to back up.
        Returns:
            str: A success or error message.
        """
        file_path = os.path.join(self.base_path, file_name)
        backup_path = file_path + ".backup"
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                content = file.read()
            with open(backup_path, 'w') as backup:
                backup.write(content)
            return f"Backup created: {backup_path}"
        return f"File {file_name} not found for backup."
