#!/usr/bin/env python3
"""
Presentation Agent for Multi-Agent Homework System
Responsible for generating presentation outlines based on assignment descriptions
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import os

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import Context

# Initialize LLM
llm = OpenAI(model="gpt-4.1-nano-2025-04-14", api_key=os.getenv("OPENAI_API_KEY"))

class PresentationAgent:
    def __init__(self):
        self.templates_dir = Path("templates") if Path("templates").exists() else None
        
    def analyze_content(self, task_description: str) -> Dict[str, List[str]]:
        """Extract key points and structure from task description."""
        # In a future version, this would use LlamaIndex to analyze content
        return {
            "title": task_description.split('\n')[0],
            "key_points": [
                "Project Overview",
                "Technical Implementation",
                "Key Features",
                "Demo",
                "Future Enhancements"
            ]
        }
        
    def generate_slide(self, topic: str, content_analysis: Dict[str, List[str]]) -> Dict[str, str]:
        """Generate content for a single slide."""
        # In a future version, this would use LlamaIndex to generate actual content
        return {
            "title": topic,
            "content": f"• Key points about {topic.lower()}\n• Supporting details\n• Examples or demonstrations",
            "notes": f"Speaker notes for {topic.lower()} slide"
        }

    def generate_presentation(self, content_analysis: Dict[str, List[str]]) -> List[Dict[str, str]]:
        """Generate full presentation outline."""
        presentation = []
        
        # Title slide
        presentation.append({
            "title": content_analysis["title"],
            "content": "Project Presentation",
            "notes": "Introduction and overview"
        })
        
        # Content slides
        for point in content_analysis["key_points"]:
            presentation.append(self.generate_slide(point, content_analysis))
            
        # Conclusion slide
        presentation.append({
            "title": "Thank You",
            "content": "• Summary of key points\n• Questions and discussion",
            "notes": "Wrap up and invite questions"
        })
        
        return presentation
        
    def format_output(self, presentation: List[Dict[str, str]]) -> str:
        """Format the presentation outline as JSON with metadata."""
        metadata = {
            "slide_count": len(presentation),
            "estimated_duration": f"{len(presentation) * 2} minutes",
            "format": "outline"
        }
        
        return json.dumps({
            "metadata": metadata,
            "slides": presentation
        }, indent=2)

def handle_task(task_description: str) -> str:
    """
    Main entry point for the Presentation Agent.
    Args:
        task_description: Description of the presentation task
    Returns:
        Generated presentation outline as a JSON string
    """
    agent = PresentationAgent()
    
    # Analyze content
    content_analysis = agent.analyze_content(task_description)
    
    # Generate presentation
    presentation = agent.generate_presentation(content_analysis)
    
    # Format and return
    return agent.format_output(presentation)

# Create the Presentation Agent with LlamaIndex
presentation_agent = FunctionAgent(
    name="PresentationAgent",
    description="Generates Google Slides for a code repository",
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
        FunctionTool.from_defaults(fn=handle_task, name="HandlePresentationTask")
    ],
    llm=llm,
    verbose=True
)


if __name__ == "__main__":
    # For testing the agent directly
    test_task = "Present a new machine learning model implementation"
    print(handle_task(test_task))