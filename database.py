"""
Database Module for BOB AI

This module handles database connections and operations for BOB AI.
It provides a unified interface for storing and retrieving data from PostgreSQL.
"""

import os
import json
import logging
import datetime
from typing import List, Dict, Any, Optional, Union

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.sql import func
from sqlalchemy.exc import SQLAlchemyError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("BOB_AI.Database")

# Create SQLAlchemy Base
Base = declarative_base()

# Define database models
class Conversation(Base):
    """Model for storing conversation history."""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(64), index=True)
    timestamp = Column(DateTime, default=func.now())
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "messages": [message.to_dict() for message in self.messages]
        }

class Message(Base):
    """Model for storing individual messages."""
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String(20))  # "user", "assistant", "system"
    content = Column(Text)
    timestamp = Column(DateTime, default=func.now())
    conversation = relationship("Conversation", back_populates="messages")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }

class Memory(Base):
    """Model for storing long-term memory and information."""
    __tablename__ = "memories"
    
    id = Column(Integer, primary_key=True)
    key = Column(String(255), index=True, unique=True)
    value = Column(Text)
    data_type = Column(String(50))  # "text", "json", "number", etc.
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "key": self.key,
            "value": self.value,
            "data_type": self.data_type,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

class DatabaseManager:
    """
    Manager for database operations.
    """
    
    def __init__(self, connection_string: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize the database manager.
        
        Args:
            connection_string (str, optional): PostgreSQL connection string.
            config_path (str, optional): Path to the configuration file.
        """
        self.connection_string = connection_string
        self.engine = None
        self.Session = None
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if "database" in config and "connection_string" in config["database"]:
                        self.connection_string = config["database"]["connection_string"]
                        logger.info("Database connection string loaded from config")
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        # Initialize database connection
        self._init_db()
    
    def _init_db(self):
        """
        Initialize the database connection.
        """
        if not self.connection_string:
            logger.error("No database connection string provided")
            return
        
        try:
            self.engine = create_engine(self.connection_string)
            self.Session = sessionmaker(bind=self.engine)
            
            # Create tables if they don't exist
            Base.metadata.create_all(self.engine)
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            self.engine = None
            self.Session = None
    
    def get_session(self) -> Optional[Session]:
        """
        Get a database session.
        
        Returns:
            Session: SQLAlchemy session.
        """
        if not self.Session:
            logger.error("Database not initialized")
            return None
        
        return self.Session()
    
    def save_conversation(self, session_id: str, messages: List[Dict[str, str]]) -> Optional[int]:
        """
        Save a conversation to the database.
        
        Args:
            session_id (str): Session ID.
            messages (List[Dict[str, str]]): List of messages.
            
        Returns:
            int: Conversation ID.
        """
        if not self.Session:
            logger.error("Database not initialized")
            return None
        
        session = self.Session()
        try:
            # Create a new conversation
            conversation = Conversation(session_id=session_id)
            session.add(conversation)
            session.flush()  # Flush to get the conversation ID
            
            # Add messages to the conversation
            for msg in messages:
                message = Message(
                    conversation_id=conversation.id,
                    role=msg.get("role", "user"),
                    content=msg.get("content", "")
                )
                session.add(message)
            
            session.commit()
            logger.info(f"Conversation saved with ID: {conversation.id}")
            return conversation.id
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error saving conversation: {e}")
            return None
        finally:
            session.close()
    
    def get_conversation(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a conversation from the database.
        
        Args:
            conversation_id (int): Conversation ID.
            
        Returns:
            Dict[str, Any]: Conversation data.
        """
        if not self.Session:
            logger.error("Database not initialized")
            return None
        
        session = self.Session()
        try:
            conversation = session.query(Conversation).filter_by(id=conversation_id).first()
            if not conversation:
                logger.warning(f"Conversation not found: {conversation_id}")
                return None
            
            return conversation.to_dict()
        except SQLAlchemyError as e:
            logger.error(f"Error getting conversation: {e}")
            return None
        finally:
            session.close()
    
    def get_conversations(self, session_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get conversations from the database.
        
        Args:
            session_id (str, optional): Session ID to filter by.
            limit (int): Maximum number of conversations to return.
            
        Returns:
            List[Dict[str, Any]]: List of conversations.
        """
        if not self.Session:
            logger.error("Database not initialized")
            return []
        
        session = self.Session()
        try:
            query = session.query(Conversation)
            if session_id:
                query = query.filter_by(session_id=session_id)
            
            conversations = query.order_by(Conversation.timestamp.desc()).limit(limit).all()
            return [conversation.to_dict() for conversation in conversations]
        except SQLAlchemyError as e:
            logger.error(f"Error getting conversations: {e}")
            return []
        finally:
            session.close()
    
    def store_memory(self, key: str, value: str, data_type: str = "text", metadata: Dict[str, Any] = None) -> Optional[int]:
        """
        Store a memory in the database.
        
        Args:
            key (str): Memory key.
            value (str): Memory value.
            data_type (str): Data type.
            metadata (Dict[str, Any], optional): Additional metadata.
            
        Returns:
            int: Memory ID.
        """
        if not self.Session:
            logger.error("Database not initialized")
            return None
        
        session = self.Session()
        try:
            # Check if memory already exists
            memory = session.query(Memory).filter_by(key=key).first()
            if memory:
                # Update existing memory
                memory.value = value
                memory.data_type = data_type
                if metadata:
                    memory.metadata = metadata
                memory.updated_at = func.now()
            else:
                # Create new memory
                memory = Memory(
                    key=key,
                    value=value,
                    data_type=data_type,
                    metadata=metadata or {}
                )
                session.add(memory)
            
            session.commit()
            logger.info(f"Memory stored with key: {key}")
            return memory.id
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error storing memory: {e}")
            return None
        finally:
            session.close()
    
    def retrieve_memory(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a memory from the database.
        
        Args:
            key (str): Memory key.
            
        Returns:
            Dict[str, Any]: Memory data.
        """
        if not self.Session:
            logger.error("Database not initialized")
            return None
        
        session = self.Session()
        try:
            memory = session.query(Memory).filter_by(key=key).first()
            if not memory:
                logger.warning(f"Memory not found: {key}")
                return None
            
            return memory.to_dict()
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving memory: {e}")
            return None
        finally:
            session.close()
    
    def search_memories(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search memories in the database.
        
        Args:
            search_term (str): Search term.
            limit (int): Maximum number of memories to return.
            
        Returns:
            List[Dict[str, Any]]: List of memories.
        """
        if not self.Session:
            logger.error("Database not initialized")
            return []
        
        session = self.Session()
        try:
            # Search in both key and value
            memories = session.query(Memory).filter(
                (Memory.key.ilike(f"%{search_term}%")) | 
                (Memory.value.ilike(f"%{search_term}%"))
            ).limit(limit).all()
            
            return [memory.to_dict() for memory in memories]
        except SQLAlchemyError as e:
            logger.error(f"Error searching memories: {e}")
            return []
        finally:
            session.close()
    
    def delete_memory(self, key: str) -> bool:
        """
        Delete a memory from the database.
        
        Args:
            key (str): Memory key.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.Session:
            logger.error("Database not initialized")
            return False
        
        session = self.Session()
        try:
            memory = session.query(Memory).filter_by(key=key).first()
            if not memory:
                logger.warning(f"Memory not found: {key}")
                return False
            
            session.delete(memory)
            session.commit()
            logger.info(f"Memory deleted: {key}")
            return True
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error deleting memory: {e}")
            return False
        finally:
            session.close()


def get_db_manager(connection_string: Optional[str] = None, config_path: Optional[str] = None) -> DatabaseManager:
    """
    Get a database manager instance.
    
    Args:
        connection_string (str, optional): PostgreSQL connection string.
        config_path (str, optional): Path to the configuration file.
        
    Returns:
        DatabaseManager: Database manager instance.
    """
    return DatabaseManager(connection_string, config_path) 