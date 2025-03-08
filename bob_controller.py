#!/usr/bin/env python3
"""
BOB AI Controller - Unified Controller for BOB AI

This module serves as the central controller for BOB AI, integrating all components
including voice interface, web API, model management, and enhancement capabilities.
"""

import os
import sys
import json
import argparse
import threading
import logging
import uuid
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"bob_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("BOB_AI")

# Import BOB AI components
try:
    # Try to import CUDA wrapper first
    try:
        from cuda_wrapper import load_model_from_config
        CUDA_WRAPPER_AVAILABLE = True
        logger.info("✅ CUDA wrapper imported successfully")
    except ImportError:
        CUDA_WRAPPER_AVAILABLE = False
        from gpt4all import GPT4All
        logger.warning("⚠️ CUDA wrapper not available, falling back to GPT4All")
    
    # Import memory module
    try:
        from memory import get_memory_manager
        MEMORY_AVAILABLE = True
        logger.info("✅ Memory module imported successfully")
    except ImportError:
        MEMORY_AVAILABLE = False
        logger.warning("⚠️ Memory module not available")
    
    from flask_api import app as flask_app, socketio
    from voice_interface import VoiceCommandProcessor
    from code_analyzer import CodeAnalyzer
    from code_generator import CodeGenerator
    from enhancer import Enhancer
    from task_scheduler import TaskScheduler
    from self_learning import generate_ai_model_from_learning
    logger.info("✅ All components imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import component: {e}")
    print(f"Error: {e}")
    print("Please make sure all dependencies are installed by running: pip install -r requirements.txt")
    sys.exit(1)

class BOBController:
    """
    Unified controller for BOB AI that integrates all components.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the BOB AI controller.
        
        Args:
            config_path (str): Path to the configuration file.
        """
        self.logger = logging.getLogger("BOB_AI.Controller")
        self.logger.info("Initializing BOB AI Controller")
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.config_path = config_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        
        # Initialize components
        self._init_components()
        
        # Status flags
        self.running = False
        self.components_status = {
            "voice": False,
            "web": False,
            "model": False,
            "scheduler": False,
            "memory": False
        }
        
        # Initialize conversation history
        self.conversation_history = []
        
        self.logger.info("BOB AI Controller initialized")
    
    def _load_config(self, config_path):
        """
        Load configuration from a JSON file.
        
        Args:
            config_path (str): Path to the configuration file.
            
        Returns:
            dict: Configuration dictionary.
        """
        default_config = {
            "model_path": os.path.join(os.path.expanduser("~"), "bob_ai", "data", "models", "mistral-7b-instruct-v0.2.Q4_K_M.gguf"),
            "web_port": 5000,
            "web_host": "0.0.0.0",
            "enable_voice": True,
            "enable_web": True,
            "enable_scheduler": True,
            "enable_cuda": True,
            "enable_memory": True,
            "data_dir": os.path.join(os.path.expanduser("~"), "bob_ai", "data"),
            "conversation_file": os.path.join(os.path.expanduser("~"), "bob_ai", "data", "conversations.json"),
            "enhancements_dir": os.path.join(os.path.expanduser("~"), "bob_ai", "core", "enhancements")
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
                self.logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                self.logger.error(f"Failed to load configuration: {e}")
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(default_config["model_path"]), exist_ok=True)
        os.makedirs(default_config["data_dir"], exist_ok=True)
        os.makedirs(default_config["enhancements_dir"], exist_ok=True)
        
        return default_config
    
    def _init_components(self):
        """
        Initialize all BOB AI components.
        """
        self.logger.info("Initializing components")
        
        # Initialize model
        try:
            if CUDA_WRAPPER_AVAILABLE and self.config["enable_cuda"]:
                self.logger.info("Using CUDA wrapper for GPU acceleration")
                self.model = load_model_from_config(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"),
                    device="cuda"
                )
                if self.model and self.model.model:
                    self.components_status["model"] = True
                    self.logger.info(f"✅ Loaded AI model from {self.config['model_path']} with CUDA support")
                else:
                    self.logger.warning(f"⚠️ Failed to load model with CUDA, falling back to CPU")
                    if os.path.exists(self.config["model_path"]):
                        self.model = GPT4All(self.config["model_path"])
                        self.components_status["model"] = True
                        self.logger.info(f"✅ Loaded AI model from {self.config['model_path']} with CPU")
                    else:
                        self.logger.warning(f"⚠️ Model not found at {self.config['model_path']}")
                        self.model = None
            else:
                if os.path.exists(self.config["model_path"]):
                    self.model = GPT4All(self.config["model_path"])
                    self.components_status["model"] = True
                    self.logger.info(f"✅ Loaded AI model from {self.config['model_path']} with CPU")
                else:
                    self.logger.warning(f"⚠️ Model not found at {self.config['model_path']}")
                    self.model = None
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            self.model = None
        
        # Initialize memory manager
        if MEMORY_AVAILABLE and self.config.get("enable_memory", True):
            try:
                self.memory = get_memory_manager(self.config_path)
                self.components_status["memory"] = True
                self.logger.info("✅ Memory manager initialized")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize memory manager: {e}")
                self.memory = None
        else:
            self.memory = None
            if not MEMORY_AVAILABLE:
                self.logger.warning("⚠️ Memory module not available")
            else:
                self.logger.info("Memory disabled in configuration")
        
        # Initialize voice processor
        if self.config["enable_voice"]:
            try:
                self.voice_processor = VoiceCommandProcessor()
                self.logger.info("✅ Voice processor initialized")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize voice processor: {e}")
                self.voice_processor = None
        else:
            self.voice_processor = None
        
        # Initialize task scheduler
        if self.config["enable_scheduler"]:
            try:
                self.scheduler = TaskScheduler()
                self.components_status["scheduler"] = True
                self.logger.info("✅ Task scheduler initialized")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize task scheduler: {e}")
                self.scheduler = None
        else:
            self.scheduler = None
        
        # Initialize enhancer
        try:
            self.enhancer = Enhancer(self.config["enhancements_dir"])
            self.logger.info("✅ Enhancer initialized")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize enhancer: {e}")
            self.enhancer = None
        
        # Initialize code tools
        try:
            self.code_analyzer = CodeAnalyzer()
            self.code_generator = CodeGenerator()
            self.logger.info("✅ Code tools initialized")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize code tools: {e}")
            self.code_analyzer = None
            self.code_generator = None
    
    def start(self):
        """
        Start all BOB AI components.
        """
        if self.running:
            self.logger.warning("BOB AI is already running")
            return
        
        self.logger.info("Starting BOB AI")
        self.running = True
        
        # Start web server in a separate thread if enabled
        if self.config["enable_web"]:
            self.web_thread = threading.Thread(
                target=self._start_web_server,
                daemon=True
            )
            self.web_thread.start()
            self.logger.info(f"✅ Web server started on {self.config['web_host']}:{self.config['web_port']}")
        
        # Start voice processor in a separate thread if enabled
        if self.config["enable_voice"] and self.voice_processor:
            self.voice_thread = threading.Thread(
                target=self._start_voice_processor,
                daemon=True
            )
            self.voice_thread.start()
            self.logger.info("✅ Voice processor started")
        
        # Start task scheduler if enabled
        if self.config["enable_scheduler"] and self.scheduler:
            self.scheduler.start()
            self.logger.info("✅ Task scheduler started")
        
        self.logger.info("BOB AI started successfully")
    
    def _start_web_server(self):
        """
        Start the Flask web server.
        """
        try:
            self.components_status["web"] = True
            socketio.run(
                flask_app,
                host=self.config["web_host"],
                port=self.config["web_port"],
                debug=False,
                use_reloader=False
            )
        except Exception as e:
            self.logger.error(f"❌ Web server error: {e}")
            self.components_status["web"] = False
    
    def _start_voice_processor(self):
        """
        Start the voice command processor.
        """
        try:
            self.components_status["voice"] = True
            while self.running:
                command = self.voice_processor.listen_for_command()
                if command:
                    self.logger.info(f"Voice command received: {command}")
                    self.process_command(command)
        except Exception as e:
            self.logger.error(f"❌ Voice processor error: {e}")
            self.components_status["voice"] = False
    
    def process_command(self, command):
        """
        Process a command from any interface.
        
        Args:
            command (str): The command to process.
        
        Returns:
            str: The response to the command.
        """
        self.logger.info(f"Processing command: {command}")
        
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": command})
        
        # Check for special commands
        if command.lower() == "exit":
            self.stop()
            return "Exiting BOB AI. Goodbye!"
        
        elif command.lower() == "status":
            return self.get_status()
        
        elif command.lower().startswith("learn "):
            topic = command[6:].strip()
            return self.learn(topic)
        
        elif command.lower().startswith("enhance "):
            description = command[8:].strip()
            return self.enhance(description)
        
        elif command.lower().startswith("remember "):
            # Format: remember key: value
            memory_text = command[9:].strip()
            if ":" in memory_text:
                key, value = memory_text.split(":", 1)
                key = key.strip()
                value = value.strip()
                return self.remember(key, value)
            else:
                return "Please provide both a key and value separated by a colon. Example: remember birthday: January 1"
        
        elif command.lower().startswith("recall "):
            # Format: recall key
            key = command[7:].strip()
            return self.recall(key)
        
        elif command.lower() == "knowledge":
            return self.summarize_knowledge()
        
        elif command.lower().startswith("knowledge "):
            # Format: knowledge topic
            topic = command[10:].strip()
            return self.summarize_knowledge(topic)
        
        elif command.lower().startswith("forget "):
            # Format: forget key
            key = command[7:].strip()
            return self.forget(key)
        
        # Use the model for general commands
        elif self.model:
            try:
                if CUDA_WRAPPER_AVAILABLE and hasattr(self.model, 'chat_session'):
                    # Use CUDA wrapper
                    with self.model.chat_session():
                        response = self.model.generate(
                            command,
                            max_tokens=200,
                            temp=0.7,
                            top_k=40,
                            top_p=0.9
                        )
                else:
                    # Use GPT4All
                    with self.model.chat_session():
                        response = self.model.generate(
                            command,
                            max_tokens=200,
                            temp=0.7,
                            top_k=40,
                            top_p=0.9
                        )
                
                # Add assistant response to conversation history
                self.conversation_history.append({"role": "assistant", "content": response})
                
                # Save conversation to memory if available
                if self.memory and self.components_status["memory"]:
                    self.memory.save_conversation(self.conversation_history)
                
                self.logger.info(f"Model response: {response}")
                return response
            except Exception as e:
                self.logger.error(f"❌ Model error: {e}")
                return f"I encountered an error: {str(e)}"
        else:
            return "I'm sorry, but my AI model is not available. Please check the logs for details."
    
    def remember(self, key, value, data_type="text", metadata=None):
        """
        Store information in memory.
        
        Args:
            key (str): Memory key.
            value (str): Memory value.
            data_type (str): Data type.
            metadata (dict, optional): Additional metadata.
            
        Returns:
            str: Status message.
        """
        if not self.memory or not self.components_status["memory"]:
            return "Memory functionality is not available."
        
        try:
            success = self.memory.remember(key, value, data_type, metadata)
            if success:
                return f"I'll remember that {key} is {value}."
            else:
                return f"I couldn't store that information in my memory."
        except Exception as e:
            self.logger.error(f"❌ Memory error: {e}")
            return f"I encountered an error while trying to remember: {str(e)}"
    
    def recall(self, key):
        """
        Retrieve information from memory.
        
        Args:
            key (str): Memory key.
            
        Returns:
            str: Retrieved information or status message.
        """
        if not self.memory or not self.components_status["memory"]:
            return "Memory functionality is not available."
        
        try:
            value = self.memory.recall(key)
            if value is not None:
                return f"{key} is {value}."
            else:
                return f"I don't have any information about {key}."
        except Exception as e:
            self.logger.error(f"❌ Memory error: {e}")
            return f"I encountered an error while trying to recall: {str(e)}"
    
    def forget(self, key):
        """
        Delete information from memory.
        
        Args:
            key (str): Memory key.
            
        Returns:
            str: Status message.
        """
        if not self.memory or not self.components_status["memory"]:
            return "Memory functionality is not available."
        
        try:
            success = self.memory.forget(key)
            if success:
                return f"I've forgotten about {key}."
            else:
                return f"I couldn't forget about {key}. Perhaps I don't have that information."
        except Exception as e:
            self.logger.error(f"❌ Memory error: {e}")
            return f"I encountered an error while trying to forget: {str(e)}"
    
    def summarize_knowledge(self, topic=None):
        """
        Summarize knowledge stored in memory.
        
        Args:
            topic (str, optional): Topic to summarize.
            
        Returns:
            str: Summary of knowledge.
        """
        if not self.memory or not self.components_status["memory"]:
            return "Memory functionality is not available."
        
        try:
            return self.memory.summarize_knowledge(topic)
        except Exception as e:
            self.logger.error(f"❌ Memory error: {e}")
            return f"I encountered an error while summarizing knowledge: {str(e)}"
    
    def learn(self, topic):
        """
        Trigger the self-learning process.
        
        Args:
            topic (str): The topic to learn about.
        
        Returns:
            str: Status message.
        """
        self.logger.info(f"Learning about: {topic}")
        try:
            generate_ai_model_from_learning()
            
            # Store the learned topic in memory if available
            if self.memory and self.components_status["memory"]:
                self.memory.remember(
                    f"learned_topic_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    topic,
                    "text",
                    {"type": "learning", "timestamp": datetime.now().isoformat()}
                )
            
            return f"I've learned about {topic} and updated my knowledge."
        except Exception as e:
            self.logger.error(f"❌ Learning error: {e}")
            return f"I encountered an error while learning: {str(e)}"
    
    def enhance(self, description):
        """
        Enhance BOB AI with new capabilities.
        
        Args:
            description (str): Description of the enhancement.
        
        Returns:
            str: Status message.
        """
        self.logger.info(f"Enhancing with: {description}")
        if self.enhancer:
            try:
                result = self.enhancer.enhance_bob(description)
                
                # Store the enhancement in memory if available
                if self.memory and self.components_status["memory"]:
                    self.memory.remember(
                        f"enhancement_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        description,
                        "text",
                        {"type": "enhancement", "timestamp": datetime.now().isoformat()}
                    )
                
                return result
            except Exception as e:
                self.logger.error(f"❌ Enhancement error: {e}")
                return f"I encountered an error during enhancement: {str(e)}"
        else:
            return "Enhancer component is not available."
    
    def get_status(self):
        """
        Get the status of all BOB AI components.
        
        Returns:
            str: Status message.
        """
        status_messages = [
            "BOB AI Status:",
            f"- Model: {'✅ Available' if self.components_status['model'] else '❌ Not available'}",
            f"- Web Interface: {'✅ Running' if self.components_status['web'] else '❌ Not running'}",
            f"- Voice Interface: {'✅ Running' if self.components_status['voice'] else '❌ Not running'}",
            f"- Task Scheduler: {'✅ Running' if self.components_status['scheduler'] else '❌ Not running'}",
            f"- Memory: {'✅ Available' if self.components_status['memory'] else '❌ Not available'}",
            f"- CUDA Acceleration: {'✅ Enabled' if CUDA_WRAPPER_AVAILABLE and self.config['enable_cuda'] else '❌ Disabled'}"
        ]
        return "\n".join(status_messages)
    
    def stop(self):
        """
        Stop all BOB AI components.
        """
        if not self.running:
            self.logger.warning("BOB AI is not running")
            return
        
        self.logger.info("Stopping BOB AI")
        self.running = False
        
        # Stop task scheduler if running
        if self.scheduler and self.components_status["scheduler"]:
            self.scheduler.stop()
            self.logger.info("Task scheduler stopped")
        
        # Web server and voice processor will stop when the main thread exits
        # because they are daemon threads
        
        self.logger.info("BOB AI stopped")


def main():
    """
    Main entry point for BOB AI.
    """
    parser = argparse.ArgumentParser(description="BOB AI - Your Private AI Assistant")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--no-web", action="store_true", help="Disable web interface")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice interface")
    parser.add_argument("--no-scheduler", action="store_true", help="Disable task scheduler")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA acceleration")
    parser.add_argument("--no-memory", action="store_true", help="Disable memory functionality")
    parser.add_argument("--model", help="Path to the model file")
    args = parser.parse_args()
    
    # Create controller
    controller = BOBController(args.config)
    
    # Override config with command line arguments
    if args.no_web:
        controller.config["enable_web"] = False
    if args.no_voice:
        controller.config["enable_voice"] = False
    if args.no_scheduler:
        controller.config["enable_scheduler"] = False
    if args.no_cuda:
        controller.config["enable_cuda"] = False
    if args.no_memory:
        controller.config["enable_memory"] = False
    if args.model:
        controller.config["model_path"] = args.model
    
    # Start controller
    try:
        controller.start()
        
        # Keep the main thread alive
        while controller.running:
            try:
                command = input("BOB AI> ")
                if command.strip():
                    response = controller.process_command(command)
                    print(response)
            except KeyboardInterrupt:
                controller.stop()
                break
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        controller.stop()


if __name__ == "__main__":
    main() 