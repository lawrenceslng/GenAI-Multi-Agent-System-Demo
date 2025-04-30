#!/usr/bin/env python3
"""
Documentation Agent for Multi-Agent Homework System
Responsible for generating documentation based on assignment descriptions
"""

import json
from pathlib import Path
from typing import Dict, List

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

if __name__ == "__main__":
    # For testing the agent directly
    test_task = "Create documentation for a Python web application"
    print(handle_task(test_task))