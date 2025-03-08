import os
import json
import torch
import pyttsx3
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
from datetime import datetime
import uuid

# Import our CUDA wrapper
try:
    from cuda_wrapper import load_model_from_config
    CUDA_WRAPPER_AVAILABLE = True
except ImportError:
    CUDA_WRAPPER_AVAILABLE = False
    from gpt4all import GPT4All
    print("Warning: CUDA wrapper not available. Falling back to GPT4All.")

# Import memory module
try:
    from memory import get_memory_manager
    MEMORY_AVAILABLE = True
    print("Memory module imported successfully")
except ImportError:
    MEMORY_AVAILABLE = False
    print("Warning: Memory module not available.")

# ✅ Flask Setup
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# ✅ Conversation Log
CONVERSATION_FILE = r"C:\Users\Ryan\bob_ai\data\conversations.json"

def log_event(event):
    """Logs messages with timestamps."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {event}")

def load_conversations():
    """Load previous conversations."""
    if os.path.exists(CONVERSATION_FILE):
        try:
            with open(CONVERSATION_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_conversation(convo):
    """Save conversation history to JSON."""
    try:
        with open(CONVERSATION_FILE, "w", encoding="utf-8") as f:
            json.dump(convo, f, indent=4)
    except Exception as e:
        log_event(f"⚠️ Error saving conversation: {e}")

conversations = load_conversations()

# ✅ Load Model
# Update the model path to use the new downloaded model
MODEL_PATH = r"C:\Users\Ryan\bob_ai\data\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Load config if available to get model path
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
config = None
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
            if "model_path" in config:
                MODEL_PATH = config["model_path"]
                log_event(f"✅ Using model path from config: {MODEL_PATH}")
    except Exception as e:
        log_event(f"⚠️ Error loading config: {e}")

if not os.path.exists(MODEL_PATH):
    log_event(f"❌ AI Model not found at: {MODEL_PATH}")
    raise FileNotFoundError(f"❌ AI Model not found at: {MODEL_PATH}")

# Initialize memory manager if available
memory_manager = None
if MEMORY_AVAILABLE and config and config.get("enable_memory", True):
    try:
        memory_manager = get_memory_manager(CONFIG_PATH)
        log_event("✅ Memory manager initialized")
    except Exception as e:
        log_event(f"❌ Failed to initialize memory manager: {e}")
        memory_manager = None

# Try to load the model with CUDA support
try:
    if CUDA_WRAPPER_AVAILABLE:
        log_event("✅ Using CUDA wrapper for GPU acceleration")
        gpt_model = load_model_from_config(CONFIG_PATH, device="cuda")
        if gpt_model and gpt_model.model:
            log_event(f"✅ Loaded AI Model from {MODEL_PATH} with CUDA support")
        else:
            log_event("⚠️ Failed to load model with CUDA, falling back to GPT4All")
            gpt_model = GPT4All(MODEL_PATH)
            log_event(f"✅ Loaded AI Model from {MODEL_PATH} with CPU")
    else:
        gpt_model = GPT4All(MODEL_PATH)
        log_event(f"✅ Loaded AI Model from {MODEL_PATH} with CPU")
except Exception as e:
    log_event(f"❌ Failed to load AI Model: {e}")
    gpt_model = None

# ✅ Text-to-Speech Setup (Fixes Speech Issue)
engine = pyttsx3.init()
engine.setProperty("rate", 160)
engine.setProperty("volume", 1.0)

def speak(text):
    """Ensures speech does not get stuck in a loop."""
    try:
        if not engine._inLoop:
            engine.say(text)
            engine.runAndWait()
        else:
            engine.endLoop()
            engine.say(text)
            engine.runAndWait()
    except Exception as e:
        log_event(f"⚠️ TTS Error: {e}")

# ✅ New Prompt Template (Keeps Bob AI on Topic)
PROMPT_TEMPLATE = """You are Bob AI, an intelligent assistant. Respond concisely.
User: {user_input}
Bob AI:"""

# Session ID for tracking conversations
SESSION_ID = str(uuid.uuid4())
# Current conversation for this session
current_conversation = []

def process_message(user_message):
    """Generate a structured AI response, ensuring it is relevant."""
    log_event(f"🗣 User: {user_message}")

    if not gpt_model:
        return "AI model is not loaded."

    # Add user message to current conversation
    current_conversation.append({"role": "user", "content": user_message})

    # Check for memory-related commands
    if user_message.lower().startswith("remember ") and memory_manager:
        # Format: remember key: value
        memory_text = user_message[9:].strip()
        if ":" in memory_text:
            key, value = memory_text.split(":", 1)
            key = key.strip()
            value = value.strip()
            success = memory_manager.remember(key, value)
            response = f"I'll remember that {key} is {value}." if success else f"I couldn't store that information in my memory."
            current_conversation.append({"role": "assistant", "content": response})
            log_event(f"🤖 Bob AI: {response}")
            speak(response)
            save_conversation_to_memory()
            return response
        else:
            response = "Please provide both a key and value separated by a colon. Example: remember birthday: January 1"
            current_conversation.append({"role": "assistant", "content": response})
            log_event(f"🤖 Bob AI: {response}")
            speak(response)
            save_conversation_to_memory()
            return response

    elif user_message.lower().startswith("recall ") and memory_manager:
        # Format: recall key
        key = user_message[7:].strip()
        value = memory_manager.recall(key)
        if value is not None:
            response = f"{key} is {value}."
        else:
            response = f"I don't have any information about {key}."
        current_conversation.append({"role": "assistant", "content": response})
        log_event(f"🤖 Bob AI: {response}")
        speak(response)
        save_conversation_to_memory()
        return response

    elif user_message.lower() == "knowledge" and memory_manager:
        response = memory_manager.summarize_knowledge()
        current_conversation.append({"role": "assistant", "content": response})
        log_event(f"🤖 Bob AI: {response}")
        speak(response)
        save_conversation_to_memory()
        return response

    elif user_message.lower().startswith("knowledge ") and memory_manager:
        # Format: knowledge topic
        topic = user_message[10:].strip()
        response = memory_manager.summarize_knowledge(topic)
        current_conversation.append({"role": "assistant", "content": response})
        log_event(f"🤖 Bob AI: {response}")
        speak(response)
        save_conversation_to_memory()
        return response

    elif user_message.lower().startswith("forget ") and memory_manager:
        # Format: forget key
        key = user_message[7:].strip()
        success = memory_manager.forget(key)
        response = f"I've forgotten about {key}." if success else f"I couldn't forget about {key}. Perhaps I don't have that information."
        current_conversation.append({"role": "assistant", "content": response})
        log_event(f"🤖 Bob AI: {response}")
        speak(response)
        save_conversation_to_memory()
        return response

    # For regular messages, use the AI model
    try:
        if CUDA_WRAPPER_AVAILABLE and hasattr(gpt_model, 'generate'):
            # Use our CUDA wrapper
            with gpt_model.chat_session():
                cleaned_response = gpt_model.generate(
                    user_message,
                    max_tokens=60,
                    temp=0.6,
                    top_k=40,
                    top_p=0.9
                )
        else:
            # Use GPT4All
            formatted_prompt = PROMPT_TEMPLATE.format(user_input=user_message)
            response = gpt_model.generate(prompt=formatted_prompt, max_tokens=60, temp=0.6, top_k=40)
            cleaned_response = response.strip()
    except Exception as e:
        log_event(f"⚠️ Model Error: {e}")
        cleaned_response = "I'm experiencing technical difficulties."

    log_event(f"🤖 Bob AI: {cleaned_response}")

    # Add assistant response to current conversation
    current_conversation.append({"role": "assistant", "content": cleaned_response})

    # ✅ Bob AI speaks response (Fixes issue)
    speak(cleaned_response)

    # ✅ Save conversation
    conversations.append({"user": user_message, "bob": cleaned_response})
    save_conversation(conversations)

    # Save conversation to memory if available
    save_conversation_to_memory()

    return cleaned_response

def save_conversation_to_memory():
    """Save the current conversation to memory if available."""
    if memory_manager and current_conversation:
        try:
            memory_manager.save_conversation(current_conversation)
        except Exception as e:
            log_event(f"⚠️ Error saving conversation to memory: {e}")

# ✅ Flask API: Receive Messages
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Message cannot be empty"}), 400
    
    bob_response = process_message(user_message)
    return jsonify({"response": bob_response})

# ✅ WebSocket: Real-Time Chat
@socketio.on("message")
def handle_message(data):
    user_message = data.get("message", "").strip()
    if user_message:
        bob_response = process_message(user_message)
        socketio.emit("response", {"message": bob_response})
    else:
        socketio.emit("error", {"message": "Invalid input"})

# ✅ Memory API Endpoints
@app.route("/api/memory", methods=["GET"])
def get_memories():
    """Get all memories or search for specific memories."""
    if not memory_manager:
        return jsonify({"error": "Memory functionality is not available"}), 503
    
    search_term = request.args.get("search", "")
    limit = int(request.args.get("limit", 10))
    
    if search_term:
        memories = memory_manager.search(search_term, limit)
    else:
        # Get all memories (limited for performance)
        session = memory_manager.db_manager.get_session()
        if session:
            try:
                from database import Memory
                memories_query = session.query(Memory).limit(limit).all()
                memories = [memory.to_dict() for memory in memories_query]
            except Exception as e:
                log_event(f"⚠️ Error getting memories: {e}")
                memories = []
            finally:
                session.close()
        else:
            memories = []
    
    return jsonify({"memories": memories})

@app.route("/api/memory/<key>", methods=["GET"])
def get_memory(key):
    """Get a specific memory by key."""
    if not memory_manager:
        return jsonify({"error": "Memory functionality is not available"}), 503
    
    memory = memory_manager.recall(key)
    if memory is None:
        return jsonify({"error": f"Memory not found: {key}"}), 404
    
    return jsonify({"key": key, "value": memory})

@app.route("/api/memory", methods=["POST"])
def store_memory():
    """Store a memory."""
    if not memory_manager:
        return jsonify({"error": "Memory functionality is not available"}), 503
    
    data = request.json
    key = data.get("key")
    value = data.get("value")
    data_type = data.get("data_type", "text")
    metadata = data.get("metadata", {})
    
    if not key or value is None:
        return jsonify({"error": "Key and value are required"}), 400
    
    success = memory_manager.remember(key, value, data_type, metadata)
    if not success:
        return jsonify({"error": "Failed to store memory"}), 500
    
    return jsonify({"success": True, "key": key, "value": value})

@app.route("/api/memory/<key>", methods=["DELETE"])
def delete_memory(key):
    """Delete a memory by key."""
    if not memory_manager:
        return jsonify({"error": "Memory functionality is not available"}), 503
    
    success = memory_manager.forget(key)
    if not success:
        return jsonify({"error": f"Failed to delete memory: {key}"}), 404
    
    return jsonify({"success": True, "key": key})

@app.route("/api/conversations", methods=["GET"])
def get_conversations():
    """Get conversation history."""
    if not memory_manager:
        return jsonify({"error": "Memory functionality is not available"}), 503
    
    limit = int(request.args.get("limit", 10))
    conversations = memory_manager.get_conversation_history(limit)
    
    return jsonify({"conversations": conversations})

# ✅ Render Web Dashboard
@app.route("/")
def index():
    return render_template("dashboard.html")

# ✅ Run Flask App
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
