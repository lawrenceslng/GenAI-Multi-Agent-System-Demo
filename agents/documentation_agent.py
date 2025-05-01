#!/usr/bin/env python3
"""
Documentation Agent for Multi-Agent Homework System
Responsible for generating documentation based on assignment descriptions
"""

import json
from pathlib import Path
from typing import Dict, List
import os

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import Context

# Initialize LLM
llm = OpenAI(model="gpt-4.1-nano-2025-04-14", api_key=os.getenv("OPENAI_API_KEY"))

class DocumentationAgent:
    def __init__(self):
        self.templates_dir = Path("templates") if Path("templates").exists() else None
        
    def parse_requirements(self, task_description: str) -> Dict[str, List[str]]:
        """Extract documentation requirements from task description."""
        # In a future version, this would use LlamaIndex to parse requirements
        return {
            "sections": [
                "Overview",
                "Requirements",
                "Implementation Details",
                "Usage Instructions",
                "Testing"
            ],
            "key_points": [task_description.split('\n')[0]]  # First line as main point
        }
        
    def generate_section(self, section: str, requirements: Dict[str, List[str]]) -> str:
        """Generate content for a specific documentation section."""
        # In a future version, this would use LlamaIndex to generate actual content
        return f"""
## {section}

This section would contain detailed information about {section.lower()}.
Key points from the assignment will be incorporated here.

"""

    def generate_documentation(self, requirements: Dict[str, List[str]]) -> str:
        """Generate full documentation based on requirements."""
        docs = "# Assignment Documentation\n\n"
        
        # Add key points from requirements
        docs += "## Key Points\n\n"
        for point in requirements["key_points"]:
            docs += f"- {point}\n"
        docs += "\n"
        
        # Generate each section
        for section in requirements["sections"]:
            docs += self.generate_section(section, requirements)
            
        return docs
        
    def format_output(self, documentation: str) -> str:
        """Format the documentation in Markdown."""
        # Add metadata and formatting
        metadata = {
            "format": "markdown",
            "word_count": len(documentation.split()),
            "sections": documentation.count("##")
        }
        
        return json.dumps({
            "metadata": metadata,
            "content": documentation
        }, indent=2)

def handle_task(task_description: str) -> str:
    """
    Main entry point for the Documentation Agent.
    Args:
        task_description: Description of the documentation task
    Returns:
        Generated documentation as a JSON string containing metadata and content
    """
    agent = DocumentationAgent()
    
    # Parse requirements
    requirements = agent.parse_requirements(task_description)
    
    # Generate documentation
    documentation = agent.generate_documentation(requirements)
    
    # Format and return
    return agent.format_output(documentation)

# Create the Documentation Agent with LlamaIndex
documentation_agent = FunctionAgent(
    name="DocumentationAgent",
    description="Generates Documentation for a Collection of Python files",
    system_prompt="""You are an expert Python programmer that generates and tests code for assignments.
    
    Your primary tools:
    - AnalyzeRequirements: Extract coding requirements from assignment descriptions
    - GenerateCode: Create Python code that meets the requirements
    - TestCode: Run and test the code in a secure Docker sandbox
    
    When generating code:
    1. Always start by analyzing and breaking down the requirements
    2. Generate well-structured, documented Python code
    3. Include proper error handling and type hints
    4. Add appropriate unit tests
    5. Test the code in the Docker sandbox
    
    Based on the results, you should:
    1. If tests pass, provide the code and test results
    2. If tests fail, analyze the errors and revise the code
    3. Ask for clarification if requirements are unclear
    
    Always ensure code is secure and follows best practices.""",
    tools=[
        FunctionTool.from_defaults(fn=handle_task, name="HandleCodingTask")
    ],
    llm=llm,
    verbose=True
)

if __name__ == "__main__":
    # For testing the agent directly
    test_task = "Create documentation for a Python web application"
    print(handle_task(test_task))