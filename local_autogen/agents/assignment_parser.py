import os
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
import re
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool

# Configure logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("assignment_parser")

class AssignmentParser:
    """Agent for parsing and understanding assignment text."""
    
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.2, workspace_dir: str = None):
        """Initialize the AssignmentParser.
        
        Args:
            model: The model to use for parsing
            temperature: Temperature setting for model generation
            workspace_dir: Directory for reading files (defaults to current directory)
        """
        self.model = model
        self.temperature = temperature
        self.workspace_dir = workspace_dir or os.getcwd()
        
        self.model_client = OpenAIChatCompletionClient(
            model=model,
            temperature=temperature
        )
        
        # Create read_file tool
        self.read_file_tool = FunctionTool(
            func=self._read_file,
            name="read_file",
            description="Read the contents of a file at the specified path"
        )
        
        # Initialize the agent with the tool
        self.agent = AssistantAgent(
            name="AssignmentParserAgent",
            model_client=self.model_client,
            description="Expert at analyzing and breaking down assignment documents",
            tools=[self.read_file_tool],
            system_message="""You are an expert at analyzing educational assignments and breaking them down into structured components.
            Your task is to carefully read assignment documents and extract key information such as:
            
            1. Title - The main title of the assignment
            2. Description - The overall description or context of the assignment
            3. Requirements - The specific items that must be completed
            4. Deliverables - The tangible outputs that must be submitted
            5. Due Date - When the assignment must be completed by
            6. Extensions - Optional additional tasks that can be completed for extra credit
            
            Be thorough and precise in your extraction. If a category isn't explicitly mentioned, make a reasonable inference or leave it empty.
            When identifying requirements and deliverables, keep them as separate, numbered lists.
            
            You will be given the path to a file, you have access to a read_file tool that you can use to read assignment files.
            """
        )
    
    def get_agent(self) -> AssistantAgent:
        """Return the assignment parser agent.
        
        Returns:
            The AssistantAgent instance ready to use in a group chat
        """
        return self.agent
    

    def _read_file(self, path: str) -> str:
        """Read content from a file.
        
        Args:
            path: Path to the file to read
            
        Returns:
            The content of the file or an error message
        """
        try:
            # Check if the path is absolute or relative
            if not os.path.isabs(path):
                # Try the path as is first
                if os.path.exists(path):
                    file_path = path
                else:
                    # Try with the workspace directory
                    file_path = os.path.join(self.workspace_dir, path)
                    
                    # If that doesn't work, try with current working directory
                    if not os.path.exists(file_path):
                        file_path = os.path.join(os.getcwd(), path)
            else:
                file_path = path
                
            logger.info(f"Attempting to read file from: {file_path}")
            
            with open(file_path, "r") as f:
                content = f.read()
            
            return content
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")
            return f"Error reading file: {e}"