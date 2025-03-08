"""
Meta-Learning Module for BOB AI

This module implements meta-learning capabilities for BOB AI, allowing it to learn how to learn
and adapt quickly to new tasks with minimal examples (few-shot learning).
"""

import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import defaultdict
import random
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("BOB_AI.MetaLearning")

class TaskExample:
    """Class representing a task example for meta-learning."""
    
    def __init__(self, inputs: Any, outputs: Any, task_id: str = None):
        """
        Initialize a task example.
        
        Args:
            inputs: Input data for the task
            outputs: Expected output for the task
            task_id: Identifier for the task
        """
        self.inputs = inputs
        self.outputs = outputs
        self.task_id = task_id or str(random.randint(0, 1000000))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "inputs": self.inputs,
            "outputs": self.outputs,
            "task_id": self.task_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskExample':
        """Create from dictionary."""
        return cls(
            inputs=data["inputs"],
            outputs=data["outputs"],
            task_id=data.get("task_id")
        )

class MetaTask:
    """Class representing a meta-learning task."""
    
    def __init__(self, name: str, description: str = "", examples: List[TaskExample] = None):
        """
        Initialize a meta-learning task.
        
        Args:
            name: Name of the task
            description: Description of the task
            examples: List of task examples
        """
        self.name = name
        self.description = description
        self.examples = examples or []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def add_example(self, example: TaskExample) -> None:
        """Add an example to the task."""
        self.examples.append(example)
        self.updated_at = datetime.now()
    
    def get_examples(self, n: int = None) -> List[TaskExample]:
        """Get n examples from the task."""
        if n is None or n >= len(self.examples):
            return self.examples
        return random.sample(self.examples, n)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "examples": [example.to_dict() for example in self.examples],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetaTask':
        """Create from dictionary."""
        task = cls(
            name=data["name"],
            description=data.get("description", "")
        )
        task.examples = [TaskExample.from_dict(example) for example in data.get("examples", [])]
        if "created_at" in data:
            task.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            task.updated_at = datetime.fromisoformat(data["updated_at"])
        return task

class MAML(nn.Module):
    """
    Model-Agnostic Meta-Learning (MAML) implementation.
    
    This class implements the MAML algorithm for meta-learning, allowing the model
    to quickly adapt to new tasks with just a few examples.
    """
    
    def __init__(self, model: nn.Module, inner_lr: float = 0.01, meta_lr: float = 0.001):
        """
        Initialize the MAML model.
        
        Args:
            model: Base model to meta-train
            inner_lr: Learning rate for task-specific adaptation
            meta_lr: Learning rate for meta-update
        """
        super(MAML, self).__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=self.meta_lr)
    
    def adapt(self, task_examples: List[TaskExample], steps: int = 1) -> nn.Module:
        """
        Adapt the model to a specific task.
        
        Args:
            task_examples: Examples for the task
            steps: Number of adaptation steps
            
        Returns:
            Adapted model for the task
        """
        # Create a clone of the model for task-specific adaptation
        adapted_model = self._clone_model()
        
        # Perform adaptation steps
        for _ in range(steps):
            loss = self._compute_loss(adapted_model, task_examples)
            grads = torch.autograd.grad(loss, adapted_model.parameters(), create_graph=True)
            
            # Update adapted model parameters
            for param, grad in zip(adapted_model.parameters(), grads):
                param.data = param.data - self.inner_lr * grad
        
        return adapted_model
    
    def meta_update(self, tasks: List[List[TaskExample]], steps: int = 1) -> float:
        """
        Perform meta-update using multiple tasks.
        
        Args:
            tasks: List of task examples for multiple tasks
            steps: Number of adaptation steps
            
        Returns:
            Meta-loss value
        """
        meta_loss = 0.0
        
        for task_examples in tasks:
            # Split examples into support (for adaptation) and query (for meta-update)
            support_examples = task_examples[:len(task_examples)//2]
            query_examples = task_examples[len(task_examples)//2:]
            
            # Adapt model to the task
            adapted_model = self.adapt(support_examples, steps)
            
            # Compute loss on query examples
            task_loss = self._compute_loss(adapted_model, query_examples)
            meta_loss += task_loss
        
        # Average meta-loss over tasks
        meta_loss = meta_loss / len(tasks)
        
        # Perform meta-update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def _clone_model(self) -> nn.Module:
        """Create a clone of the model."""
        clone = type(self.model)(*self.model.args, **self.model.kwargs)
        clone.load_state_dict(self.model.state_dict())
        return clone
    
    def _compute_loss(self, model: nn.Module, examples: List[TaskExample]) -> torch.Tensor:
        """Compute loss for a batch of examples."""
        # This is a placeholder - actual implementation depends on the task
        loss = torch.tensor(0.0, requires_grad=True)
        return loss

class MetaLearningManager:
    """
    Manager for meta-learning capabilities.
    
    This class manages meta-learning tasks and models, providing an interface
    for BOB AI to learn from few examples and adapt to new tasks quickly.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the meta-learning manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.tasks = {}
        self.models = {}
        
        # Load configuration
        self.config = {}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if "meta_learning" in config:
                        self.config = config["meta_learning"]
                        logger.info("Meta-learning configuration loaded")
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        # Initialize meta-learning components
        self._init_components()
    
    def _init_components(self) -> None:
        """Initialize meta-learning components."""
        # Load saved tasks
        self._load_tasks()
    
    def _load_tasks(self) -> None:
        """Load saved meta-learning tasks."""
        tasks_dir = self.config.get("tasks_dir", "meta_tasks")
        if not os.path.exists(tasks_dir):
            os.makedirs(tasks_dir, exist_ok=True)
            return
        
        for filename in os.listdir(tasks_dir):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(tasks_dir, filename), 'r') as f:
                        task_data = json.load(f)
                        task = MetaTask.from_dict(task_data)
                        self.tasks[task.name] = task
                        logger.info(f"Loaded task: {task.name}")
                except Exception as e:
                    logger.error(f"Error loading task {filename}: {e}")
    
    def _save_task(self, task: MetaTask) -> bool:
        """Save a meta-learning task."""
        tasks_dir = self.config.get("tasks_dir", "meta_tasks")
        os.makedirs(tasks_dir, exist_ok=True)
        
        try:
            filename = os.path.join(tasks_dir, f"{task.name}.json")
            with open(filename, 'w') as f:
                json.dump(task.to_dict(), f, indent=4)
            logger.info(f"Saved task: {task.name}")
            return True
        except Exception as e:
            logger.error(f"Error saving task {task.name}: {e}")
            return False
    
    def create_task(self, name: str, description: str = "") -> MetaTask:
        """
        Create a new meta-learning task.
        
        Args:
            name: Name of the task
            description: Description of the task
            
        Returns:
            Created task
        """
        if name in self.tasks:
            logger.warning(f"Task {name} already exists, returning existing task")
            return self.tasks[name]
        
        task = MetaTask(name, description)
        self.tasks[name] = task
        self._save_task(task)
        return task
    
    def add_example(self, task_name: str, inputs: Any, outputs: Any) -> bool:
        """
        Add an example to a meta-learning task.
        
        Args:
            task_name: Name of the task
            inputs: Input data for the example
            outputs: Expected output for the example
            
        Returns:
            True if successful, False otherwise
        """
        if task_name not in self.tasks:
            logger.warning(f"Task {task_name} does not exist")
            return False
        
        task = self.tasks[task_name]
        example = TaskExample(inputs, outputs)
        task.add_example(example)
        return self._save_task(task)
    
    def get_task(self, name: str) -> Optional[MetaTask]:
        """
        Get a meta-learning task by name.
        
        Args:
            name: Name of the task
            
        Returns:
            Task if found, None otherwise
        """
        return self.tasks.get(name)
    
    def get_tasks(self) -> List[MetaTask]:
        """
        Get all meta-learning tasks.
        
        Returns:
            List of tasks
        """
        return list(self.tasks.values())
    
    def delete_task(self, name: str) -> bool:
        """
        Delete a meta-learning task.
        
        Args:
            name: Name of the task
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self.tasks:
            logger.warning(f"Task {name} does not exist")
            return False
        
        tasks_dir = self.config.get("tasks_dir", "meta_tasks")
        filename = os.path.join(tasks_dir, f"{name}.json")
        
        try:
            if os.path.exists(filename):
                os.remove(filename)
            del self.tasks[name]
            logger.info(f"Deleted task: {name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting task {name}: {e}")
            return False
    
    def few_shot_learning(self, task_name: str, query_input: Any, n_shots: int = 3) -> Any:
        """
        Perform few-shot learning on a task.
        
        Args:
            task_name: Name of the task
            query_input: Input to generate a prediction for
            n_shots: Number of examples to use
            
        Returns:
            Predicted output
        """
        if task_name not in self.tasks:
            logger.warning(f"Task {task_name} does not exist")
            return None
        
        task = self.tasks[task_name]
        examples = task.get_examples(n_shots)
        
        if not examples:
            logger.warning(f"No examples found for task {task_name}")
            return None
        
        # This is a simple implementation - in a real system, this would use a model
        # For now, we'll just return the output of the most similar example
        most_similar_example = min(examples, key=lambda ex: self._similarity(ex.inputs, query_input))
        return most_similar_example.outputs
    
    def _similarity(self, a: Any, b: Any) -> float:
        """Compute similarity between two inputs."""
        # This is a placeholder - actual implementation depends on the input type
        if isinstance(a, str) and isinstance(b, str):
            # Simple string similarity
            return 1.0 - len(set(a.lower().split()) & set(b.lower().split())) / max(len(set(a.lower().split())), len(set(b.lower().split())), 1)
        return 1.0  # Default: no similarity

def get_meta_learning_manager(config_path: Optional[str] = None) -> MetaLearningManager:
    """
    Get a meta-learning manager instance.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Meta-learning manager instance
    """
    return MetaLearningManager(config_path) 