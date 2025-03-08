#!/usr/bin/env python3
"""
Setup Database Script for BOB AI

This script sets up the PostgreSQL database for BOB AI.
It creates the necessary database and tables if they don't exist.
"""

import os
import sys
import json
import argparse
import logging
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("BOB_AI.SetupDatabase")

def load_config(config_path=None):
    """
    Load configuration from a JSON file.
    
    Args:
        config_path (str, optional): Path to the configuration file.
        
    Returns:
        dict: Configuration dictionary.
    """
    default_config = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "user": "postgres",
            "password": "postgres",
            "dbname": "bob_ai"
        }
    }
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                if "database" in config:
                    default_config["database"].update(config["database"])
                    logger.info(f"Loaded database configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    
    return default_config

def create_database(config):
    """
    Create the PostgreSQL database if it doesn't exist.
    
    Args:
        config (dict): Database configuration.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    db_config = config["database"]
    host = db_config.get("host", "localhost")
    port = db_config.get("port", 5432)
    user = db_config.get("user", "postgres")
    password = db_config.get("password", "postgres")
    dbname = db_config.get("dbname", "bob_ai")
    
    try:
        # Connect to PostgreSQL server
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (dbname,))
        exists = cursor.fetchone()
        
        if not exists:
            # Create database
            cursor.execute(f"CREATE DATABASE {dbname}")
            logger.info(f"Database '{dbname}' created successfully")
        else:
            logger.info(f"Database '{dbname}' already exists")
        
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        return False

def create_tables(config):
    """
    Create the necessary tables in the PostgreSQL database.
    
    Args:
        config (dict): Database configuration.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    db_config = config["database"]
    host = db_config.get("host", "localhost")
    port = db_config.get("port", 5432)
    user = db_config.get("user", "postgres")
    password = db_config.get("password", "postgres")
    dbname = db_config.get("dbname", "bob_ai")
    
    try:
        # Connect to the database
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=dbname
        )
        cursor = conn.cursor()
        
        # Create conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(64),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT unique_session_id UNIQUE (session_id)
            )
        """)
        
        # Create messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
                role VARCHAR(20),
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create memories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id SERIAL PRIMARY KEY,
                key VARCHAR(255) UNIQUE,
                value TEXT,
                data_type VARCHAR(50),
                metadata JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key)")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("Tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        return False

def update_config_with_connection_string(config_path, config):
    """
    Update the configuration file with the database connection string.
    
    Args:
        config_path (str): Path to the configuration file.
        config (dict): Configuration dictionary.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    if not config_path:
        logger.warning("No configuration file path provided, skipping update")
        return False
    
    try:
        # Load existing config
        with open(config_path, 'r') as f:
            existing_config = json.load(f)
        
        # Get database configuration
        db_config = config["database"]
        host = db_config.get("host", "localhost")
        port = db_config.get("port", 5432)
        user = db_config.get("user", "postgres")
        password = db_config.get("password", "postgres")
        dbname = db_config.get("dbname", "bob_ai")
        
        # Create connection string
        connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        
        # Update config
        if "database" not in existing_config:
            existing_config["database"] = {}
        
        existing_config["database"]["connection_string"] = connection_string
        existing_config["database"]["enabled"] = True
        
        # Save updated config
        with open(config_path, 'w') as f:
            json.dump(existing_config, f, indent=4)
        
        logger.info(f"Configuration updated with connection string: {connection_string}")
        return True
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        return False

def setup_database(config_path=None, host=None, port=None, user=None, password=None, dbname=None):
    """
    Set up the PostgreSQL database for BOB AI.
    
    Args:
        config_path (str, optional): Path to the configuration file.
        host (str, optional): Database host.
        port (int, optional): Database port.
        user (str, optional): Database user.
        password (str, optional): Database password.
        dbname (str, optional): Database name.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    # Load configuration
    config = load_config(config_path)
    
    # Override configuration with command-line arguments
    if host:
        config["database"]["host"] = host
    if port:
        config["database"]["port"] = port
    if user:
        config["database"]["user"] = user
    if password:
        config["database"]["password"] = password
    if dbname:
        config["database"]["dbname"] = dbname
    
    # Create database
    if not create_database(config):
        return False
    
    # Create tables
    if not create_tables(config):
        return False
    
    # Update configuration with connection string
    if config_path:
        update_config_with_connection_string(config_path, config)
    
    return True

def main():
    """
    Main entry point for the setup database script.
    """
    parser = argparse.ArgumentParser(description="BOB AI Database Setup")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--host", help="Database host")
    parser.add_argument("--port", type=int, help="Database port")
    parser.add_argument("--user", help="Database user")
    parser.add_argument("--password", help="Database password")
    parser.add_argument("--dbname", help="Database name")
    args = parser.parse_args()
    
    # Set default config path if not provided
    if not args.config:
        args.config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    
    # Set up database
    success = setup_database(
        config_path=args.config,
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        dbname=args.dbname
    )
    
    if success:
        print("\n" + "=" * 80)
        print("BOB AI database setup completed successfully!")
        print("=" * 80)
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("BOB AI database setup failed. Please check the logs for details.")
        print("=" * 80)
        sys.exit(1)

if __name__ == "__main__":
    main() 