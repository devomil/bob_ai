import pyttsx3
import os
import sys
from core.voice_interface import VoiceCommandProcessor
from core.code_analyzer import CodeAnalyzer
from core.code_generator import CodeGenerator
from core.screen_interactor import ScreenInteractor
from core.code_interaction_module import CodeInteractionModule
from core.bob_ai_dashboard import BOBDashboard
from core.enhancer import Enhancer  # Import the Enhancer module
import tkinter as tk
import threading

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BOBController:
    """
    Main controller for BOB AI, handling commands, tasks, and integrations.
    """

    def __init__(self, dashboard=None):
        # Framework validation
        self.validate_environment()

        self.dashboard = dashboard
        self.analyzer = CodeAnalyzer()
        self.generator = CodeGenerator()
        self.screen = ScreenInteractor()
        self.voice_processor = VoiceCommandProcessor()
        self.code_interactor = CodeInteractionModule(base_path="core")
        self.tts_engine = pyttsx3.init()
        self.enhancer = Enhancer(r"C:\Users\Ryan\bob_ai\local_models\EleutherAI_gpt-j-6b")  # Initialize Enhancer
        self.configure_tts()

    def validate_environment(self):
        """
        Validate that the required frameworks are installed and available.
        """
        missing_frameworks = []

        try:
            import torch
            print(f"PyTorch version: {torch.__version__}")
        except ImportError:
            missing_frameworks.append("torch")

        try:
            import tensorflow as tf
            print(f"TensorFlow version: {tf.__version__}")
        except ImportError:
            missing_frameworks.append("tensorflow")

        try:
            import flax
            print("Flax is available.")
        except ImportError:
            missing_frameworks.append("flax")

        if missing_frameworks:
            print(f"Warning: The following frameworks are missing: {', '.join(missing_frameworks)}")
            print("Please install one of these frameworks to enable model functionality.")
        else:
            print("All required frameworks are available.")

    def configure_tts(self):
        """
        Configure text-to-speech settings.
        """
        voices = self.tts_engine.getProperty('voices')
        self.tts_engine.setProperty('voice', voices[0].id)
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 0.9)

    def speak(self, message: str):
        """
        Speak a message using text-to-speech and display it in the GUI.
        """
        if self.dashboard:
            self.dashboard.display_feedback(message)
        print(message)
        self.tts_engine.say(message)
        self.tts_engine.runAndWait()

    def generate_response(self, prompt):
        """
        Generate a conversational response based on the input prompt.
        """
        input_ids = self.enhancer.tokenizer.encode(prompt, return_tensors="pt")
        attention_mask = (input_ids != self.enhancer.tokenizer.pad_token_id).int()

        output = self.enhancer.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=150,
            temperature=0.7,  # Adjust for creativity
            do_sample=True,
            top_p=0.9,
        )
        return self.enhancer.tokenizer.decode(output[0], skip_special_tokens=True)

    def process_command(self, command: str):
        """
        Process voice/text commands given to BOB AI.
        """
        command = command.lower()

        if command == "exit":
            response = "Exiting BOB AI. Goodbye!"
            self.speak(response)
            exit(0)
        elif command == "hello":
            response = "Hello! I am BOB AI. How can I assist you today?"
            self.speak(response)
        elif "how are you" in command:
            response = "I'm just a program, but I'm functioning perfectly. How about you?"
            self.speak(response)
        elif "learn" in command:
            response = self.learn_new_feature(feature_request=command)
            self.speak(response)
        elif "enhance" in command:
            response = self.enhance_bob(command.replace("enhance ", "").strip())
            self.speak(response)
        else:
            # Use the conversational response generator for unknown commands
            response = self.generate_response(command)
            self.speak(response)

    def enhance_bob(self, description):
        """
        Generate and validate an enhancement for Bob.
        """
        result = self.enhancer.enhance_bob(description)
        return result

    def display_feedback(self, feedback):
        """
        Display feedback in the GUI and print it to the console.
        """
        if self.dashboard:
            self.dashboard.display_feedback(feedback)
        print(feedback)

    def process_voice_command(self):
        while True:
            command = self.voice_processor.listen_for_command()
            if command:
                self.display_feedback(f"Processing voice command: {command}")
                self.process_command(command)


if __name__ == "__main__":
    # Initialize the root Tk instance
    root = tk.Tk()
    dashboard = BOBDashboard(root)

    # Create the BOB controller and link it with the dashboard
    bob = BOBController(dashboard)

    # Run voice commands in a separate thread
    threading.Thread(target=bob.process_voice_command, daemon=True).start()

    # Start the GUI main loop
    root.mainloop()
