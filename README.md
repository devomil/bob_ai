# BOB AI - Your Private AI Assistant

BOB AI is a comprehensive private AI assistant that runs locally on your machine. It combines multiple AI capabilities including:

- Natural language processing with local LLMs (GPT4All)
- Voice interaction capabilities
- Web interface via Flask
- Self-learning and enhancement capabilities
- Code generation and analysis

## Features

- **Voice Interface**: Interact with BOB AI using voice commands
- **Web Dashboard**: Access BOB AI through a web interface
- **Local LLM Integration**: Uses GPT4All to run models locally
- **Self-Learning**: Can learn and improve from interactions
- **Code Generation**: Can generate and test Python code
- **Task Scheduling**: Can manage and execute scheduled tasks
- **Extensible Architecture**: Easily add new capabilities

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- TensorFlow (optional)
- GPT4All
- Flask and Flask-SocketIO
- SoundDevice and SpeechRecognition
- Transformers library

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bob_ai.git
cd bob_ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download a compatible model for GPT4All (e.g., Mistral 7B):
```bash
python model_downloader.py
```

### Usage

#### Command Line Interface

Run BOB AI from the command line:
```bash
python gpt4all_cli.py
```

#### Web Interface

Start the web server:
```bash
python flask_api.py
```
Then open your browser to http://localhost:5000

#### GUI Dashboard

Launch the GUI dashboard:
```bash
python bob_ai.py
```

#### Voice Interface

Start the voice interface:
```bash
python voice_interface.py
```

## Project Structure

- `bob_ai.py`: Main controller for BOB AI
- `voice_interface.py`: Voice command processing
- `flask_api.py`: Web API and interface
- `gpt4all_cli.py`: Command-line interface
- `enhancer.py`: Code generation and enhancement
- `self_learning.py`: Self-improvement capabilities
- `task_scheduler.py`: Task scheduling and management
- `code_analyzer.py`: Code analysis tools
- `code_generator.py`: Code generation tools

## Extending BOB AI

BOB AI is designed to be easily extensible. To add new capabilities:

1. Create a new module in the `enhancements` directory
2. Implement your functionality
3. Import and integrate it with the main BOB AI controller

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- GPT4All for providing local LLM capabilities
- Hugging Face for transformer models
- PyTorch and TensorFlow communities 