import asyncio
import logging
import sys
import os

from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult, ToolCall
from llama_index.core.workflow import Context
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

import dotenv

dotenv.load_dotenv()

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Load LLM
llm = OpenAI(model="gpt-4.1-nano-2025-04-14", api_key=os.getenv("OPENAI_API_KEY"))

# Initialize ElevenLabs MCP client
mcp_client = BasicMCPClient(
    command_or_url="uv",
    args=[
        "--directory",
        os.getenv("ELEVENLABS_PATH"),
        "run",
        "elevenlabs-mcp-server"
    ],
    env={
        "ELEVENLABS_API_KEY": os.getenv("ELEVENLABS_API_KEY"),
        "ELEVENLABS_VOICE_ID": os.getenv("ELEVENLABS_VOICE_ID"),
        "ELEVENLABS_MODEL_ID": os.getenv("ELEVENLABS_MODEL_ID"),
        "ELEVENLABS_STABILITY": os.getenv("ELEVENLABS_STABILITY"),
        "ELEVENLABS_SIMILARITY_BOOST": os.getenv("ELEVENLABS_SIMILARITY_BOOST"),
        "ELEVENLABS_STYLE": os.getenv("ELEVENLABS_STYLE"),
        "ELEVENLABS_OUTPUT_DIR": os.getenv("ELEVENLABS_OUTPUT_DIR")
    }
)

mcp_tool = McpToolSpec(client=mcp_client)

SYSTEM_PROMPT = """
You are an AI assistant specialized in text-to-speech conversion using ElevenLabs.

You can use ElevenLabs MCP tools to generate high-quality audio from text.
Available tools:
- generate_audio_simple: Generate audio from plain text using default voice settings
- get_audio_file: Get the audio file by its ID
- list_voices: List all available voices
- get_voiceover_history: Get voiceover job history

When processing text:
1. Generate audio using generate_audio_simple with the text
2. The response will contain the audio data directly
3. Save the audio data to a file

Always ensure:
1. Text is properly formatted for speech
2. Audio quality is high
3. Output is saved in accessible format
4. Proper error handling and reporting
"""

async def get_agent(tools: McpToolSpec):
    tools = await tools.to_tool_list_async()
    agent = FunctionAgent(
        name="VoiceoverAgent",
        description="An agent that converts text to speech using ElevenLabs.",
        tools=tools,
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
        verbose=True,
    )
    return agent

async def generate_and_save_audio(text: str, agent: FunctionAgent, agent_context: Context, output_dir: str = "output/audio") -> dict:
    """
    Generate audio from text and save it locally.
    
    Args:
        text: Text to convert to speech
        agent: The function agent instance
        agent_context: The agent context
        output_dir: Directory to save the audio file
    
    Returns:
        Dictionary with job info and local file path
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the prompt for generating audio
        prompt = f"Generate audio for the text: '{text}'"
        
        # Run the agent to generate audio
        handler = agent.run(prompt, ctx=agent_context)
        audio_saved = False
        
        async for event in handler.stream_events():
            if type(event) == ToolCall:
                logger.info(f"Tool Call: {event.tool_name} with kwargs {event.tool_kwargs}")
            elif type(event) == ToolCallResult:
                logger.info(f"Tool Result: {event.tool_name}")
                if event.tool_name == "generate_audio_simple":
                    import shutil
                    
                    if event.tool_output and event.tool_output.content:
                        import re

                        audio_uri = None
                        audio_uri = event.tool_output.content
                        # Extract audio URI from the string using regex
                        match = re.search(r"audio://[\w\-\.]+\.mp3", event.tool_output.content)
                        if match:
                            audio_uri = match.group(0)  # e.g., 'audio://full_audio_20250502104348.mp3'
                            filename = audio_uri.replace("audio://", "")
                            
                            elevenlabs_path = os.getenv("ELEVENLABS_PATH")
                            elevenlabs_output_dir = os.getenv("ELEVENLABS_OUTPUT_DIR", "outputs")
                            source_path = os.path.join(elevenlabs_path, elevenlabs_output_dir, filename)

                            if os.path.exists(source_path):
                                timestamp = int(asyncio.get_event_loop().time())
                                output_path = os.path.join(output_dir, f"audio_{timestamp}.mp3")
                                shutil.copyfile(source_path, output_path)
                                audio_saved = True
                                return {
                                    "success": True,
                                    "file_path": output_path
                                }
                            else:
                                logger.error(f"Resolved audio file does not exist at: {source_path}")
                                logger.info(f"Checking if ELEVENLABS_PATH is set correctly: {elevenlabs_path}")
                                logger.info(f"Checking if ELEVENLABS_OUTPUT_DIR is set correctly: {elevenlabs_output_dir}")
                                logger.info(f"Full source path that doesn't exist: {source_path}")
                        else:
                            logger.error("Could not find audio URI in tool output content")
                            logger.info(f"Content received: {event.tool_output.content}")

        
        if not audio_saved:
            logger.error("No audio was saved during the process")
            return {"error": "Failed to generate or save audio"}
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in generate_and_save_audio: {error_msg}")
        
        # Check for specific error types
        if "quota_exceeded" in error_msg.lower():
            return {
                "error": "quota_exceeded",
                "message": error_msg
            }
        elif "too_many_concurrent_requests" in error_msg.lower():
            return {
                "error": "too_many_concurrent_requests",
                "message": error_msg
            }
        
        return {"error": error_msg}

async def main():
    # Get the agent
    agent = await get_agent(mcp_tool)
    agent_context = Context(agent)

    # Test message
    text = "Welcome to the multi-agent system demo"
    result = await generate_and_save_audio(text, agent, agent_context)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Audio generated successfully!")
        print(f"File saved to: {result.get('file_path')}")

if __name__ == "__main__":
    asyncio.run(main())