"""
Memory Module for BOB AI

This module provides memory capabilities for BOB AI, allowing it to store and retrieve
information from a PostgreSQL database.
"""

import os
import json
import logging
import uuid
import datetime
from typing import List, Dict, Any, Optional, Union

from database import get_db_manager, DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("BOB_AI.Memory")

class MemoryManager:
    """
    Manager for BOB AI's memory capabilities.
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None, config_path: Optional[str] = None):
        """
        Initialize the memory manager.
        
        Args:
            db_manager (DatabaseManager, optional): Database manager instance.
            config_path (str, optional): Path to the configuration file.
        """
        self.config_path = config_path
        
        # Load configuration if provided
        self.config = {}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                    logger.info("Configuration loaded")
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        # Initialize database manager
        self.db_manager = db_manager or get_db_manager(config_path=config_path)
        
        # Initialize session ID for conversations
        self.session_id = str(uuid.uuid4())
        logger.info(f"Memory manager initialized with session ID: {self.session_id}")
    
    def remember(self, key: str, value: Any, data_type: str = "text", metadata: Dict[str, Any] = None) -> bool:
        """
        Store information in memory.
        
        Args:
            key (str): Memory key.
            value (Any): Memory value.
            data_type (str): Data type.
            metadata (Dict[str, Any], optional): Additional metadata.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.db_manager:
            logger.error("Database manager not initialized")
            return False
        
        # Convert value to string if it's not already
        if not isinstance(value, str):
            if data_type == "json" and isinstance(value, (dict, list)):
                value = json.dumps(value)
            else:
                value = str(value)
        
        # Store memory in database
        memory_id = self.db_manager.store_memory(key, value, data_type, metadata)
        return memory_id is not None
    
    def recall(self, key: str) -> Optional[Any]:
        """
        Retrieve information from memory.
        
        Args:
            key (str): Memory key.
            
        Returns:
            Any: Memory value.
        """
        if not self.db_manager:
            logger.error("Database manager not initialized")
            return None
        
        # Retrieve memory from database
        memory = self.db_manager.retrieve_memory(key)
        if not memory:
            return None
        
        # Convert value based on data type
        value = memory["value"]
        data_type = memory["data_type"]
        
        if data_type == "json":
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON value for key: {key}")
                return value
        elif data_type == "number":
            try:
                return float(value)
            except ValueError:
                logger.warning(f"Failed to parse number value for key: {key}")
                return value
        elif data_type == "boolean":
            return value.lower() in ("true", "yes", "1")
        else:
            return value
    
    def search(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for information in memory.
        
        Args:
            search_term (str): Search term.
            limit (int): Maximum number of results to return.
            
        Returns:
            List[Dict[str, Any]]: List of matching memories.
        """
        if not self.db_manager:
            logger.error("Database manager not initialized")
            return []
        
        # Search memories in database
        return self.db_manager.search_memories(search_term, limit)
    
    def forget(self, key: str) -> bool:
        """
        Delete information from memory.
        
        Args:
            key (str): Memory key.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.db_manager:
            logger.error("Database manager not initialized")
            return False
        
        # Delete memory from database
        return self.db_manager.delete_memory(key)
    
    def save_conversation(self, messages: List[Dict[str, str]]) -> Optional[int]:
        """
        Save a conversation to memory.
        
        Args:
            messages (List[Dict[str, str]]): List of messages.
            
        Returns:
            int: Conversation ID.
        """
        if not self.db_manager:
            logger.error("Database manager not initialized")
            return None
        
        # Save conversation to database
        return self.db_manager.save_conversation(self.session_id, messages)
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get conversation history from memory.
        
        Args:
            limit (int): Maximum number of conversations to return.
            
        Returns:
            List[Dict[str, Any]]: List of conversations.
        """
        if not self.db_manager:
            logger.error("Database manager not initialized")
            return []
        
        # Get conversations from database
        return self.db_manager.get_conversations(self.session_id, limit)
    
    def summarize_knowledge(self, topic: Optional[str] = None) -> str:
        """
        Summarize knowledge stored in memory.
        
        Args:
            topic (str, optional): Topic to summarize.
            
        Returns:
            str: Summary of knowledge.
        """
        if not self.db_manager:
            logger.error("Database manager not initialized")
            return "No knowledge available."
        
        # Get all memories or filter by topic
        memories = []
        if topic:
            memories = self.search(topic)
        else:
            # Get all memories (limited to 100 for performance)
            session = self.db_manager.get_session()
            if session:
                try:
                    from database import Memory
                    memories_query = session.query(Memory).limit(100).all()
                    memories = [memory.to_dict() for memory in memories_query]
                except Exception as e:
                    logger.error(f"Error getting memories: {e}")
                finally:
                    session.close()
        
        if not memories:
            return f"No knowledge available{' about ' + topic if topic else ''}."
        
        # Build summary
        summary = []
        if topic:
            summary.append(f"Knowledge about {topic}:")
        else:
            summary.append("Knowledge summary:")
        
        for memory in memories:
            key = memory["key"]
            value = memory["value"]
            if len(value) > 100:
                value = value[:97] + "..."
            summary.append(f"- {key}: {value}")
        
        return "\n".join(summary)


def get_memory_manager(config_path: Optional[str] = None) -> MemoryManager:
    """
    Get a memory manager instance.
    
    Args:
        config_path (str, optional): Path to the configuration file.
        
    Returns:
        MemoryManager: Memory manager instance.
    """
    return MemoryManager(config_path=config_path) 