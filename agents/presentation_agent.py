#!/usr/bin/env python3
"""
Presentation Agent for Multi-Agent Homework System
Responsible for generating presentation outlines based on assignment descriptions
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

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

if __name__ == "__main__":
    # For testing the agent directly
    test_task = "Present a new machine learning model implementation"
    print(handle_task(test_task))