"""
Transfer Learning Module for BOB AI

This module implements transfer learning capabilities for BOB AI, allowing it to leverage
pre-trained models and adapt them to new tasks with minimal training data.
"""

import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from collections import defaultdict
import random
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("BOB_AI.TransferLearning")

class ModelRegistry:
    """Registry for pre-trained models that can be used for transfer learning."""
    
    def __init__(self, models_dir: str = "pretrained_models"):
        """
        Initialize the model registry.
        
        Args:
            models_dir: Directory to store pre-trained models
        """
        self.models_dir = models_dir
        self.models = {}
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load the model registry from disk."""
        os.makedirs(self.models_dir, exist_ok=True)
        registry_file = os.path.join(self.models_dir, "registry.json")
        
        if os.path.exists(registry_file):
            try:
                with open(registry_file, 'r') as f:
                    self.models = json.load(f)
                logger.info(f"Loaded model registry with {len(self.models)} models")
            except Exception as e:
                logger.error(f"Error loading model registry: {e}")
                self.models = {}
    
    def _save_registry(self) -> bool:
        """Save the model registry to disk."""
        os.makedirs(self.models_dir, exist_ok=True)
        registry_file = os.path.join(self.models_dir, "registry.json")
        
        try:
            with open(registry_file, 'w') as f:
                json.dump(self.models, f, indent=4)
            logger.info("Saved model registry")
            return True
        except Exception as e:
            logger.error(f"Error saving model registry: {e}")
            return False
    
    def register_model(self, model_id: str, model_info: Dict[str, Any]) -> bool:
        """
        Register a pre-trained model.
        
        Args:
            model_id: Unique identifier for the model
            model_info: Information about the model
            
        Returns:
            True if successful, False otherwise
        """
        if model_id in self.models:
            logger.warning(f"Model {model_id} already exists in registry")
            return False
        
        self.models[model_id] = {
            **model_info,
            "registered_at": datetime.now().isoformat()
        }
        return self._save_registry()
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a registered model.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            Model information if found, None otherwise
        """
        return self.models.get(model_id)
    
    def list_models(self, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List registered models.
        
        Args:
            domain: Filter models by domain
            
        Returns:
            List of model information
        """
        if domain:
            return [
                {"id": model_id, **model_info}
                for model_id, model_info in self.models.items()
                if model_info.get("domain") == domain
            ]
        return [{"id": model_id, **model_info} for model_id, model_info in self.models.items()]
    
    def unregister_model(self, model_id: str) -> bool:
        """
        Unregister a model.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            True if successful, False otherwise
        """
        if model_id not in self.models:
            logger.warning(f"Model {model_id} not found in registry")
            return False
        
        del self.models[model_id]
        return self._save_registry()

class TransferLearningModel:
    """Base class for transfer learning models."""
    
    def __init__(self, base_model: nn.Module, model_id: str, domain: str):
        """
        Initialize a transfer learning model.
        
        Args:
            base_model: Pre-trained base model
            model_id: Unique identifier for the model
            domain: Domain of the model (e.g., "nlp", "vision")
        """
        self.base_model = base_model
        self.model_id = model_id
        self.domain = domain
        self.fine_tuned_models = {}
    
    def freeze_base_model(self) -> None:
        """Freeze the parameters of the base model."""
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def unfreeze_base_model(self) -> None:
        """Unfreeze the parameters of the base model."""
        for param in self.base_model.parameters():
            param.requires_grad = True
    
    def create_task_specific_head(self, task_id: str, output_size: int) -> nn.Module:
        """
        Create a task-specific head for the model.
        
        Args:
            task_id: Unique identifier for the task
            output_size: Size of the output layer
            
        Returns:
            Task-specific head
        """
        # This is a placeholder - actual implementation depends on the model architecture
        return nn.Linear(self.base_model.output_size, output_size)
    
    def fine_tune(self, task_id: str, train_data: Any, val_data: Any, epochs: int = 5, lr: float = 0.001) -> Dict[str, Any]:
        """
        Fine-tune the model for a specific task.
        
        Args:
            task_id: Unique identifier for the task
            train_data: Training data
            val_data: Validation data
            epochs: Number of training epochs
            lr: Learning rate
            
        Returns:
            Training metrics
        """
        # This is a placeholder - actual implementation depends on the model architecture
        return {"task_id": task_id, "status": "fine-tuned"}
    
    def predict(self, task_id: str, inputs: Any) -> Any:
        """
        Make predictions using a fine-tuned model.
        
        Args:
            task_id: Unique identifier for the task
            inputs: Input data
            
        Returns:
            Predictions
        """
        # This is a placeholder - actual implementation depends on the model architecture
        return None
    
    def save(self, models_dir: str) -> bool:
        """
        Save the model to disk.
        
        Args:
            models_dir: Directory to save the model
            
        Returns:
            True if successful, False otherwise
        """
        # This is a placeholder - actual implementation depends on the model architecture
        return True
    
    @classmethod
    def load(cls, model_path: str) -> 'TransferLearningModel':
        """
        Load a model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model
        """
        # This is a placeholder - actual implementation depends on the model architecture
        return None

class NLPTransferModel(TransferLearningModel):
    """Transfer learning model for natural language processing tasks."""
    
    def __init__(self, base_model: nn.Module, model_id: str):
        """
        Initialize an NLP transfer learning model.
        
        Args:
            base_model: Pre-trained base model
            model_id: Unique identifier for the model
        """
        super().__init__(base_model, model_id, "nlp")
    
    def fine_tune(self, task_id: str, train_data: Any, val_data: Any, epochs: int = 5, lr: float = 0.001) -> Dict[str, Any]:
        """
        Fine-tune the model for a specific NLP task.
        
        Args:
            task_id: Unique identifier for the task
            train_data: Training data
            val_data: Validation data
            epochs: Number of training epochs
            lr: Learning rate
            
        Returns:
            Training metrics
        """
        # Freeze the base model
        self.freeze_base_model()
        
        # Create a task-specific head
        head = self.create_task_specific_head(task_id, output_size=train_data.num_classes)
        
        # Create a combined model
        model = nn.Sequential(self.base_model, head)
        
        # Define optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        metrics = {"loss": [], "accuracy": []}
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            
            for inputs, targets in train_data:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == targets).sum().item()
            
            train_loss /= len(train_data)
            train_accuracy = train_correct / len(train_data.dataset)
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            
            with torch.no_grad():
                for inputs, targets in val_data:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == targets).sum().item()
            
            val_loss /= len(val_data)
            val_accuracy = val_correct / len(val_data.dataset)
            
            # Save metrics
            metrics["loss"].append({"train": train_loss, "val": val_loss})
            metrics["accuracy"].append({"train": train_accuracy, "val": val_accuracy})
            
            logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Save the fine-tuned model
        self.fine_tuned_models[task_id] = model
        
        return {
            "task_id": task_id,
            "status": "fine-tuned",
            "metrics": metrics,
            "epochs": epochs
        }
    
    def predict(self, task_id: str, inputs: Any) -> Any:
        """
        Make predictions using a fine-tuned NLP model.
        
        Args:
            task_id: Unique identifier for the task
            inputs: Input text
            
        Returns:
            Predictions
        """
        if task_id not in self.fine_tuned_models:
            logger.warning(f"No fine-tuned model found for task {task_id}")
            return None
        
        model = self.fine_tuned_models[task_id]
        model.eval()
        
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.tolist()

class VisionTransferModel(TransferLearningModel):
    """Transfer learning model for computer vision tasks."""
    
    def __init__(self, base_model: nn.Module, model_id: str):
        """
        Initialize a vision transfer learning model.
        
        Args:
            base_model: Pre-trained base model
            model_id: Unique identifier for the model
        """
        super().__init__(base_model, model_id, "vision")
    
    def fine_tune(self, task_id: str, train_data: Any, val_data: Any, epochs: int = 5, lr: float = 0.001) -> Dict[str, Any]:
        """
        Fine-tune the model for a specific vision task.
        
        Args:
            task_id: Unique identifier for the task
            train_data: Training data
            val_data: Validation data
            epochs: Number of training epochs
            lr: Learning rate
            
        Returns:
            Training metrics
        """
        # Similar to NLPTransferModel.fine_tune, but adapted for vision tasks
        # This is a placeholder - actual implementation depends on the model architecture
        return {"task_id": task_id, "status": "fine-tuned"}
    
    def predict(self, task_id: str, inputs: Any) -> Any:
        """
        Make predictions using a fine-tuned vision model.
        
        Args:
            task_id: Unique identifier for the task
            inputs: Input images
            
        Returns:
            Predictions
        """
        # Similar to NLPTransferModel.predict, but adapted for vision tasks
        # This is a placeholder - actual implementation depends on the model architecture
        return None

class TransferLearningManager:
    """
    Manager for transfer learning capabilities.
    
    This class manages transfer learning models and tasks, providing an interface
    for BOB AI to leverage pre-trained models and adapt them to new tasks.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the transfer learning manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        
        # Load configuration
        self.config = {}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if "transfer_learning" in config:
                        self.config = config["transfer_learning"]
                        logger.info("Transfer learning configuration loaded")
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        # Initialize model registry
        models_dir = self.config.get("models_dir", "pretrained_models")
        self.registry = ModelRegistry(models_dir)
        
        # Initialize model cache
        self.models = {}
    
    def register_pretrained_model(self, model_id: str, model_path: str, model_type: str, domain: str, description: str = "") -> bool:
        """
        Register a pre-trained model for transfer learning.
        
        Args:
            model_id: Unique identifier for the model
            model_path: Path to the model file
            model_type: Type of the model (e.g., "bert", "resnet")
            domain: Domain of the model (e.g., "nlp", "vision")
            description: Description of the model
            
        Returns:
            True if successful, False otherwise
        """
        model_info = {
            "path": model_path,
            "type": model_type,
            "domain": domain,
            "description": description
        }
        return self.registry.register_model(model_id, model_info)
    
    def get_model(self, model_id: str) -> Optional[TransferLearningModel]:
        """
        Get a transfer learning model by ID.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            Transfer learning model if found, None otherwise
        """
        # Check if model is already loaded
        if model_id in self.models:
            return self.models[model_id]
        
        # Get model info from registry
        model_info = self.registry.get_model_info(model_id)
        if not model_info:
            logger.warning(f"Model {model_id} not found in registry")
            return None
        
        # Load model
        try:
            model_path = model_info["path"]
            model_type = model_info["type"]
            domain = model_info["domain"]
            
            # Load base model based on type and domain
            if domain == "nlp":
                if model_type == "bert":
                    # Load BERT model
                    from transformers import BertModel
                    base_model = BertModel.from_pretrained(model_path)
                    model = NLPTransferModel(base_model, model_id)
                elif model_type == "gpt":
                    # Load GPT model
                    from transformers import GPT2Model
                    base_model = GPT2Model.from_pretrained(model_path)
                    model = NLPTransferModel(base_model, model_id)
                else:
                    logger.warning(f"Unsupported NLP model type: {model_type}")
                    return None
            elif domain == "vision":
                if model_type == "resnet":
                    # Load ResNet model
                    from torchvision.models import resnet50
                    base_model = resnet50(pretrained=True)
                    model = VisionTransferModel(base_model, model_id)
                else:
                    logger.warning(f"Unsupported vision model type: {model_type}")
                    return None
            else:
                logger.warning(f"Unsupported domain: {domain}")
                return None
            
            # Cache the model
            self.models[model_id] = model
            
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            return None
    
    def fine_tune_model(self, model_id: str, task_id: str, train_data: Any, val_data: Any, epochs: int = 5, lr: float = 0.001) -> Dict[str, Any]:
        """
        Fine-tune a model for a specific task.
        
        Args:
            model_id: Unique identifier for the model
            task_id: Unique identifier for the task
            train_data: Training data
            val_data: Validation data
            epochs: Number of training epochs
            lr: Learning rate
            
        Returns:
            Training metrics
        """
        model = self.get_model(model_id)
        if not model:
            return {"error": f"Model {model_id} not found"}
        
        return model.fine_tune(task_id, train_data, val_data, epochs, lr)
    
    def predict(self, model_id: str, task_id: str, inputs: Any) -> Any:
        """
        Make predictions using a fine-tuned model.
        
        Args:
            model_id: Unique identifier for the model
            task_id: Unique identifier for the task
            inputs: Input data
            
        Returns:
            Predictions
        """
        model = self.get_model(model_id)
        if not model:
            return {"error": f"Model {model_id} not found"}
        
        return model.predict(task_id, inputs)
    
    def list_models(self, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available pre-trained models.
        
        Args:
            domain: Filter models by domain
            
        Returns:
            List of model information
        """
        return self.registry.list_models(domain)

def get_transfer_learning_manager(config_path: Optional[str] = None) -> TransferLearningManager:
    """
    Get a transfer learning manager instance.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Transfer learning manager instance
    """
    return TransferLearningManager(config_path) 