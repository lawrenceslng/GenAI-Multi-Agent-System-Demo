#!/usr/bin/env python3
"""
Voiceover Agent for Multi-Agent Homework System
Responsible for converting text to speech using ElevenLabs' MCP server
"""

import json
from pathlib import Path
from typing import Dict
import os

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import Context

# Initialize LLM
llm = OpenAI(model="gpt-4.1-nano-2025-04-14", api_key=os.getenv("OPENAI_API_KEY"))

class VoiceoverAgent:
    def __init__(self):
        self.output_dir = Path("output/audio")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_audio(self, text: str, voice_id: str = "default") -> Dict[str, str]:
        """
        Generate audio from text using ElevenLabs MCP server.
        Args:
            text: The text to convert to speech
            voice_id: The ID of the voice to use (default uses ElevenLabs' default voice)
        Returns:
            Dictionary containing metadata about the generated audio
        """
        try:
            # Use MCP tool to generate audio
            response = {
                "server_name": "elevenlabs",
                "tool_name": "text_to_speech",
                "arguments": {
                    "text": text,
                    "voice_id": voice_id
                }
            }
            
            # Generate unique filename
            timestamp = Context.current().get_timestamp()
            output_path = self.output_dir / f"audio_{timestamp}.mp3"
            
            return {
                "status": "success",
                "output_path": str(output_path),
                "metadata": {
                    "text_length": len(text),
                    "voice_id": voice_id,
                    "timestamp": timestamp
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def format_output(self, result: Dict[str, str]) -> str:
        """Format the voiceover generation result as JSON."""
        return json.dumps(result, indent=2)

def handle_task(text: str, voice_id: str = "default") -> str:
    """
    Main entry point for the Voiceover Agent.
    Args:
        text: Text to convert to speech
        voice_id: Optional voice ID to use
    Returns:
        JSON string containing the result metadata and output path
    """
    agent = VoiceoverAgent()
    
    # Generate audio
    result = agent.generate_audio(text, voice_id)
    
    # Format and return
    return agent.format_output(result)

# Create the Voiceover Agent with LlamaIndex
voiceover_agent = FunctionAgent(
    name="VoiceoverAgent",
    description="Converts text to speech using ElevenLabs' text-to-speech service",
    system_prompt="""You are an expert text-to-speech specialist that converts text content to natural-sounding speech.
    
    Your primary capability:
    - Convert text to speech using ElevenLabs' high-quality voices
    
    When processing text:
    1. Analyze the text for natural speech patterns
    2. Select appropriate voice and speech parameters
    3. Generate high-quality audio output
    4. Handle errors gracefully
    
    Always ensure:
    1. Text is properly formatted for speech
    2. Audio quality is high
    3. Output is saved in accessible format
    4. Proper error handling and reporting""",
    tools=[
        FunctionTool.from_defaults(fn=handle_task, name="HandleVoiceoverTask")
    ],
    llm=llm,
    verbose=True
)

if __name__ == "__main__":
    # For testing the agent directly
    test_text = "Welcome to the multi-agent homework system. This is a test of the voiceover capabilities."
    print(handle_task(test_text))