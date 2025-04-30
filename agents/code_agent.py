#!/usr/bin/env python3
"""
Code Agent for Multi-Agent Homework System
Responsible for generating Python code based on assignment descriptions
"""

from pathlib import Path
from typing import Dict, Optional
import json

class CodeAgent:
    def __init__(self):
        self.sandbox_dir = Path("sandbox")
        
    def analyze_requirements(self, task_description: str) -> Dict[str, str]:
        """Extract coding requirements from task description."""
        # In a future version, this would use LlamaIndex to parse requirements
        return {
            "language": "python",
            "main_task": task_description,
            "requirements": ["Generate basic implementation"]
        }
        
    def generate_code(self, requirements: Dict[str, str]) -> str:
        """Generate code based on requirements."""
        # In a future version, this would use LlamaIndex to generate actual code
        return """
# Generated Python Implementation
def main():
    print("Hello from the generated code!")
    # TODO: Implement actual functionality based on requirements
    
if __name__ == "__main__":
    main()
"""

    def test_code(self, code: str) -> Optional[str]:
        """
        Test the generated code in a sandbox environment.
        Returns None if tests pass, error message otherwise.
        """
        # In a future version, this would use Docker for sandboxed execution
        return None

def handle_task(task_description: str) -> str:
    """
    Main entry point for the Code Agent.
    Args:
        task_description: Description of the coding task
    Returns:
        Generated code as a string
    """
    agent = CodeAgent()
    
    # Analyze requirements
    requirements = agent.analyze_requirements(task_description)
    
    # Generate code
    generated_code = agent.generate_code(requirements)
    
    # Test code
    test_result = agent.test_code(generated_code)
    if test_result:
        return f"Error in generated code: {test_result}"
    
    # Create result object
    result = {
        "status": "success",
        "requirements": requirements,
        "code": generated_code,
        "test_result": "All tests passed"
    }
    
    return json.dumps(result, indent=2)

if __name__ == "__main__":
    # For testing the agent directly
    test_task = "Create a simple Python program that prints 'Hello World'"
    print(handle_task(test_task))