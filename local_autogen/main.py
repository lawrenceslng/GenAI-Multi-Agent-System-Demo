import os
import argparse
import json
import re
import logging
import time
from typing import Dict, List, Any, Optional, Sequence
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import TextMentionTermination
from autogen_core.tools import FunctionTool

from dotenv import load_dotenv

# MCP Client Libraries
import requests
import github
from github import Github
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import google.auth.exceptions

# Add MCP related imports
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
import asyncio

# Import our new AssignmentParser
from local_autogen.agents.assignment_parser import AssignmentParser

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("main")

DEFAULT_OUTPUT_DIR = "./output"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MODEL = "gpt-4o-mini"

class GitHubMCPClient():
    """Client for interacting with GitHub API using MCP."""
    
    def __init__(self):
        super().__init__()
        self.workbench = None
        self.tools = []
        self.repo = None
        
    async def initialize(self) -> bool:
        try:
            # Initialize GitHub workbench
            logger.info("Initializing GitHub workbench...")
            github_params = StdioServerParams(
                command=os.getenv("GITHUB_MCP_URL", "github-mcp"),
                args=["stdio"],
                env={
                    "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PAT")
                }
            )
            self.workbench = McpWorkbench(server_params=github_params)
            await self.workbench.start()
            
            # Get tools
            self.tools = await self.workbench.list_tools()
            logger.info(f"Created {len(self.tools)} GitHub tools")
            
            return True
        except Exception as e:
            logger.error(f"GitHub workbench initialization failed: {e}")
            return False

class GoogleSlidesMCPClient():
    """Client for interacting with Google Slides API."""
    
    def __init__(self):
        super().__init__()
        self.workbench = None
        self.tools = []
        
    async def initialize(self) -> bool:
        try:
            logger.info("Initializing Google Slides workbench...")
            slides_params = StdioServerParams(
                command="node",
                args=[
                    os.getenv("SLIDES_MCP_PATH"),
                    "--google-client-id", os.getenv("GOOGLE_CLIENT_ID"),
                    "--google-client-secret", os.getenv("GOOGLE_CLIENT_SECRET"),
                    "--google-refresh-token", os.getenv("GOOGLE_REFRESH_TOKEN")
                ]
            )
            self.workbench = McpWorkbench(server_params=slides_params)
            await self.workbench.start()
            
            # Get tools
            self.tools = await self.workbench.list_tools()
            logger.info(f"Created {len(self.tools)} Google Slides tools")
            
            return True
        except Exception as e:
            logger.error(f"Google Slides workbench initialization failed: {e}")
            return False

class ElevenLabsMCPClient():
    """Client for interacting with ElevenLabs API using MCP."""
    
    def __init__(self):
        super().__init__()
        self.workbench = None
        self.tools = []
    
    async def initialize(self) -> bool:
        try:
            # Initialize ElevenLabs workbench
            logger.info("Initializing ElevenLabs workbench...")
            elevenlabs_params = StdioServerParams(
                command="uv",
                args=[
                    "--directory",
                    os.getenv("ELEVENLABS_PATH", "/home/erudman21/school/genai/GenAI-Multi-Agent-System-Demo/elevenlabs-mcp"),
                    "run",
                    "elevenlabs-mcp-server"
                ],
                env={
                    "ELEVENLABS_API_KEY": os.getenv("ELEVENLABS_API_KEY"),
                    "ELEVENLABS_VOICE_ID": os.getenv("ELEVENLABS_VOICE_ID"),
                    "ELEVENLABS_MODEL_ID": os.getenv("ELEVENLABS_MODEL_ID", "eleven_monolingual_v2"),
                    "ELEVENLABS_STABILITY": os.getenv("ELEVENLABS_STABILITY", "0.5"),
                    "ELEVENLABS_SIMILARITY_BOOST": os.getenv("ELEVENLABS_SIMILARITY_BOOST", "0.75"),
                    "ELEVENLABS_STYLE": os.getenv("ELEVENLABS_STYLE", "0.0"),
                    "ELEVENLABS_OUTPUT_DIR": os.getenv("ELEVENLABS_OUTPUT_DIR", "./output/audio")
                },
                read_timeout_seconds=60
            )
            self.workbench = McpWorkbench(server_params=elevenlabs_params)
            await self.workbench.start()
            
            # Get tools
            self.tools = await self.workbench.list_tools()
            logger.info(f"Created {len(self.tools)} ElevenLabs tools")
            
            return True
        except Exception as e:
            logger.error(f"ElevenLabs workbench initialization failed: {e}")
            return False
    
class MultiAgentSystem:
    """Main class for the multi-agent system."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MCPs
        self.github_mcp = GitHubMCPClient()
        self.slides_mcp = GoogleSlidesMCPClient()
        self.elevenlabs_mcp = ElevenLabsMCPClient()
        
        # Initialize the assignment parser
        self.assignment_parser = AssignmentParser(
            model=DEFAULT_MODEL,
            temperature=DEFAULT_TEMPERATURE
        )
        
        self.termination_msgs = ["TERMINATE", "TASK_COMPLETE", "DONE"]
        
        self.assignment_data = {}
        self.task_status = {
            "coding": False,
            "documentation": False,
            "presentation": False,
            "voice_over": False
        }
    
    async def initialize_mcps(self) -> bool:
        """Initialize and authenticate all MCPs."""
        github_auth = await self.github_mcp.initialize()
        slides_auth = await self.slides_mcp.initialize()
        elevenlabs_auth = await self.elevenlabs_mcp.initialize()
        
        # Create output directory for audio files if it doesn't exist
        audio_dir = self.output_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Log authentication status
        logger.info(f"GitHub MCP: {'Authenticated' if github_auth else 'Not authenticated or disabled'}")
        logger.info(f"Google Slides MCP: {'Authenticated' if slides_auth else 'Not authenticated or disabled'}")
        logger.info(f"ElevenLabs MCP: {'Authenticated' if elevenlabs_auth else 'Not authenticated or disabled'}")
        
        return github_auth or slides_auth or elevenlabs_auth
    
    def _create_file_tools(self):
        """Create file read/write tools."""
        # Create file write tool
        def file_write(path: str, content: str, mode: str = "w"):
            try:
                file_path = self.output_dir / path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(file_path, mode) as f:
                    f.write(content)
                
                logger.error(f"Successfully wrote {len(content)} bytes to {path}")
                return f"Successfully wrote to {path}"
            except Exception as e:
                logger.error(f"Error writing to file: {e}")
                return f"Error writing to file: {e}"
        
        # Create file read tool
        def file_read(path: str):
            try:
                file_path = self.output_dir / path
                if not file_path.exists():
                    return f"Error: File {path} does not exist"
                
                with open(file_path, "r") as f:
                    content = f.read()
                
                return content
            except Exception as e:
                logger.error(f"Error reading file: {e}")
                return f"Error reading file: {e}"
        
        return [
            FunctionTool(func=file_write, name="file_write", description="Write content to a file"),
            FunctionTool(func=file_read, name="file_read", description="Read content from a file")
        ]

    async def setup_agents(self) -> None:
        """Set up the agents for the system."""
        try:
            # Initialize model client
            model_client = OpenAIChatCompletionClient(
                model=DEFAULT_MODEL,
                temperature=DEFAULT_TEMPERATURE
            )
            
            large_model_client = OpenAIChatCompletionClient(
                model="gpt-4o-mini",
                temperature=DEFAULT_TEMPERATURE
            )
            
            small_model_client = OpenAIChatCompletionClient(
                model="gpt-3.5-turbo",
                temperature=DEFAULT_TEMPERATURE
            )
            
            gemini_model_client = OpenAIChatCompletionClient(
                model="gemini-2.0-flash",
                temperature=DEFAULT_TEMPERATURE
            )
            
            # Create output directories
            code_dir = self.output_dir / "code"
            code_dir.mkdir(parents=True, exist_ok=True)
            
            # Create file tools
            file_tools = self._create_file_tools()
            
            # Define agent for coding tasks with GitHub workbench
            coder_agent = AssistantAgent(
                name="CoderAgent",
                model_client=model_client,
                workbench=self.github_mcp.workbench,
                description="Expert software developer responsible for implementing code",
                system_message="""You are an expert software developer responsible for implementing the coding part of assignments.
                Your primary focus is on producing clean, efficient, and well-documented code that meets all requirements.
                You should prioritize creating a working prototype before adding advanced features or optimizations.
                
                IMPORTANT: Your main responsibility is to write actual CODE, not just describe or document it.
                You MUST create real, working code files (like .py, .js, .html, etc.) not just markdown or text descriptions.
                
                When writing code:
                1. Create proper file extensions matching the language (.py for Python, .js for JavaScript, etc.)
                2. Implement complete, runnable code that fulfills the requirements
                3. Include comprehensive comments to explain your approach and any non-trivial logic
                
                For web applications, focus on both frontend and backend components, ensuring they work together seamlessly.
                Use best practices for the languages and frameworks specified in the assignment.
                Break down complex tasks into manageable components and implement them systematically.
                
                You have access to GitHub tools that allow you to interact with repositories.
                Your github username is robot-coder and your repos will always be in this format: robot-coder/{repo-name}
                Use these tools to manage code and documentation.

                ** CRITICAL **
                Always create a new GitHub Repository for the assignment and push the code you write to the repository you created.
                If a repository with that name already exists, choose a different name.
                Do not perform extraneous actions like creating multiple branches, creating issues, or creating comments in the repository.
                When updating files in GitHub repositories, always follow these steps:
                1. First use get_file_contents to check if the file exists and get its SHA
                2. If the file exists, use create_or_update_file with the SHA parameter
                3. If the file doesn't exist, use create_or_update_file without a SHA
                
                When you complete your part of the assignment, clearly state "CODING COMPLETE" in your message.
                """
            )
            
            # Define agent for documentation
            documenter_agent = AssistantAgent(
                name="DocumenterAgent",
                model_client=model_client,
                workbench=self.github_mcp.workbench,
                description="Expert technical writer responsible for creating clear documentation",
                system_message="""You are an expert technical writer responsible for creating clear, comprehensive documentation.
                Your documentation should cover installation instructions, usage examples, API references, and architectural overview.
                Use markdown format for all documentation, with proper headings, lists, and code blocks.
                Create a main README.md file that provides an overview of the project and links to other documentation.
                Include diagrams or illustrations where they can clarify complex concepts or workflows.
                Document both the public API and internal architecture to aid future development.
                Focus on making the documentation accessible to users with varying levels of technical expertise.
                For each component or feature, explain both how to use it and why certain design decisions were made.
                
                When creating your documentation, reach out to the coder agent for any information you need.
                
                You have access to GitHub tools to manage documentation files.
                
                ** CRITICAL **
                Do not perform extraneous actions like creating multiple branches or issues in the repository.
                When updating files in GitHub repositories, always follow these steps:
                1. First use get_file_contents to check if the file exists and get its SHA
                2. If the file exists, use create_or_update_file with the SHA parameter
                3. If the file doesn't exist, use create_or_update_file without a SHA
                
                When you complete your part of the assignment, clearly state "DOCUMENTATION COMPLETE" in your message.
                """
            )
            
            # Define agent for presentation creation with Google Slides
            presenter_agent = AssistantAgent(
                name="PresenterAgent",
                model_client=model_client,
                workbench=self.slides_mcp.workbench,
                description="Expert in creating engaging technical presentations",
                system_message="""You are an expert in creating engaging technical presentations.
                Your presentations should be concise, visually appealing, and focused on the key aspects of the project.
                Begin with an introduction slide that clearly states the purpose and goals of the project.
                Include slides that demonstrate the functionality and features of the implemented solution.
                Highlight technical challenges and how they were overcome in the implementation.
                Use bullet points rather than long paragraphs to maintain audience engagement.
                Include code snippets only when they illustrate important concepts or techniques.
                End with a conclusion that summarizes the achievements and potential future enhancements.
                Each slide should have a clear purpose and contribute to the overall narrative.
                Focus on demonstrating the value and capabilities of the solution rather than implementation details.
                
                When creating the presentation, create a dialogue with the coder agent to discuss interesting parts of the code or interesting challenges and solutions the agent faced.
                Use the responses from this dialogue in the presentation.
                
                You have access to Google Slides tools to create and manage slides.
                
                When creating presentations, you MUST follow this EXACT sequence:
                1. First use create_presentation to make a new presentation
                2. Then use create_slide to add slides with the right layouts:
                - Use TITLE_SLIDE layout for the first slide
                - Use TITLE_AND_BODY layout for content slides
                3. Then get the presentation structure using get_presentation to find the correct object IDs
                4. MOST IMPORTANTLY, you MUST add content with batch_update_presentation
                
                YOU MUST ADD CONTENT TO YOUR SLIDES. Creating empty slides is not enough!
                
                Here's an example of how to update a presentation with content using batch_update_presentation:
                ```
                # After getting presentation structure with get_presentation to find object IDs
                presentation_id = "your_presentation_id"
                response = get_presentation({"presentation_id": presentation_id})
                
                # Find slide IDs and element IDs
                slides = response.get("slides", [])
                title_slide = slides[0]
                title_slide_id = title_slide.get("objectId")
                
                # Find specific elements on the slide
                elements = title_slide.get("pageElements", [])
                title_element = None
                subtitle_element = None
                
                for element in elements:
                    if "shape" in element and "placeholder" in element["shape"]:
                        placeholder_type = element["shape"]["placeholder"].get("type")
                        if placeholder_type == "TITLE":
                            title_element = element
                        elif placeholder_type == "SUBTITLE":
                            subtitle_element = element
                
                # Get element IDs
                title_element_id = title_element["objectId"] if title_element else None
                subtitle_element_id = subtitle_element["objectId"] if subtitle_element else None
                
                # Prepare the update requests
                requests = []
                
                # Add title text
                if title_element_id:
                    requests.append({
                        "insertText": {
                            "objectId": title_element_id,
                            "text": "Project Title",
                            "insertionIndex": 0
                        }
                    })
                
                # Add subtitle text
                if subtitle_element_id:
                    requests.append({
                        "insertText": {
                            "objectId": subtitle_element_id,
                            "text": "Presentation by: PresenterAgent",
                            "insertionIndex": 0
                        }
                    })
                
                # Execute the batch update
                batch_update_presentation({
                    "presentation_id": presentation_id,
                    "requests": requests
                })
                ```
                
                NEVER try to reference object IDs like "title" or "slide1" directly - these don't exist by default.
                Always use the actual object IDs from the get_presentation response.
                
                **CRITICAL**
                - Always create a title slide with the assignment title and your name as the first slide.
                - Use themes and layout that are visually appealing and appropriate for the assignment.
                - Make sure the presentation is not only created but there is actual content in it.
                - NEVER USE ANYTHING EXCEPT GOOGLE SLIDES TO CREATE THE PRESENTATION.
                - YOU MUST ADD ACTUAL CONTENT using batch_update_presentation BEFORE marking as complete.
                - NEVER SAY PRESENTATION COMPLETE UNTIL THE PRESENTATION HAS ACTUAL CONTENT IN IT.
                
                When you complete your part of the assignment, clearly state "PRESENTATION COMPLETE" in your message.
                """
            )
            
            presentation_verifier_agent = AssistantAgent(
                name="PresentationVerifierAgent",
                model_client=model_client,
                workbench=self.slides_mcp.workbench,
                description="Expert in verifying technical presentations",
                system_message="""Your only job is to use the Google slides tools available to you to verify that the presentation
                created by the PresenterAgent has actual content in it and is not just the default initial slide.
                
                When you complete your verification, clearly state PRESENTATION VERIFICATION COMPLETE"
                """
            )
            
            # Define agent for voice-over creation
            voice_over_agent = AssistantAgent(
                name="VoiceOverAgent",
                model_client=gemini_model_client,
                workbench=self.elevenlabs_mcp.workbench,
                description="Expert in creating voice-over scripts for presentations",
                system_message="""You are an expert in creating clear, engaging voice-over scripts for technical presentations.
                Your voice-overs should complement the slides, providing additional context and explanation.
                Use natural, conversational language that is easy to follow when spoken aloud.
                Keep sentences relatively short and avoid complex sentence structures that may be difficult to follow aurally.
                Emphasize key points and ensure transitions between slides are smooth and logical.
                Avoid using technical jargon without explanation, considering that the audience may have varying expertise.
                Maintain a consistent tone and pace throughout the presentation.
                Script the voice-over to match the timing of the slides, allowing adequate time for complex concepts.
                Include brief pauses in the script where the audience may need time to process information.
                Use an engaging, enthusiastic tone to maintain audience interest without being overly dramatic.
                Generate the audio file in mp3 format.
                
                You have access to ElevenLabs tools for generating voice-overs.
                
                When using generate_audio_simple, ALWAYS explicitly include the voice_id parameter:
                generate_audio_simple({"text": "Your script", "voice_id": "iEw1wkYocsNy7I7pteSN"})
                
                If audio generation fails with an error about voice_id, try the following:
                1. Pass the voice_id explicitly as shown above
                2. If still failing, use the FileWriterAgent to save your script to voice_over_script.txt
                3. Then mark your task as VOICE-OVER COMPLETE so the workflow can continue
                
                When you complete your part of the assignment, clearly state "VOICE-OVER COMPLETE" in your message.
                """
            )
            
            # Define planning agent with access to all workbenches
            planning_agent = AssistantAgent(
                name="PlanningAgent",
                model_client=large_model_client,
                description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
                system_message="""You are a planning agent.
                Your job is to break down complex tasks into smaller, manageable subtasks.
                Your team members are:
                    AssignmentParserAgent: Parses the assignment description
                    CoderAgent: Writes code for the assignment
                    DocumenterAgent: Writes documentation for the assignment
                    PresenterAgent: Creates presentation slides for the assignment
                    VoiceOverAgent: Does a voice over for the presentation
                    
                You only plan and delegate tasks - you do not execute them yourself.
                ONLY USE THE FILE WRITER AGENT TO WRITE CODE AND AUDIO FILES. NEVER USE IT WITH THE PRESENTATION AGENT. NEVER HAVE IT WRITE PPTX FILES.
                ONLY USE THE DOCUMENTER AGENT TO WRITE DOCUMENTATION FOR CODE. NEVER USE IT WITH THE PRESENTER AGENT.
                MAKE ABSOLUTELY SURE AND CONFIRM MULTIPLE TIMES WITH THE PRESENTATION AGENT THAT THE PRESENTATION HAS ACTUAL CONTENT IN IT AND IS NOT ONLY CREATED.
                ALWAYS CREATE THE VOICE OVER USING ELEVENLABS. NEVER CREATE THE VOICE OVER IN ANY OTHER WAY.
                
                Always create a new directory in output/ for each task run as the first thing you do.

                When assigning tasks, use this format:
                1. <agent> : <task>

                Only terminate the conversation when ALL of these have been completed:
                1. CoderAgent has stated "CODING COMPLETE"
                2. DocumenterAgent has stated "DOCUMENTATION COMPLETE" 
                3. PresenterAgent has stated "PRESENTATION COMPLETE"
                4. VoiceOverAgent has stated "VOICE-OVER COMPLETE"
                5. PresentationVerifierAgent has stated "PRESENTATION VERIFICATION COMPLETE"

                DO NOT send the TERMINATE command until all four agents have explicitly stated they've completed their tasks.
                """
            )
            
            # file_writer_agent = AssistantAgent(
            #     name="FileWriterAgent",
            #     model_client=model_client,
            #     tools=file_tools,
            #     description="An agent to write files to the output directory",
            #     system_message="""You are an agent to write files to the output directory.
            #     You have access to the file_write tool to write files to the output directory.
            #     ONLY WRITE CODE AND AUDIO FILES. DO NOT ASSIST THE PRESENTATION AGENT AT ALL.
            #     """
            # )
            
            # Define selector prompt
            selector_prompt = """Please analyze the conversation history and the current request, and select the most appropriate agent to respond.
            Be thoughtful in your analysis and consider which agent has the expertise needed to address the current situation:
            
            {roles}

            Current conversation context:
            {history}

            Read the above conversation, then select an agent from {participants} to perform the next task.
            Make sure the planner agent has assigned tasks before other agents start working.
            Use the FileWriterAgent to write ALL files created by the other agents to the output directory.
            Create a new directory in the output directory for each task run.
            
            When selecting the CoderAgent, ensure actual code implementation is being done, not just discussion or documentation.
            The CoderAgent should produce code files with proper file extensions (.py, .js, etc.) not just markdown.
            
            Only select one agent.
            
            Always select the agent whose expertise best matches the current need in the conversation. If you're unsure, select the PlannerAgent to provide guidance.
            """
            
            assignment_parser_agent = self.assignment_parser.get_agent()
            
            # def selector_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
            #     if messages[-1].source != planning_agent.name:
            #         return planning_agent.name
            #     return None
            
            # Create a SelectorGroupChat with all agents
            self.group_chat = SelectorGroupChat(
                [
                    planning_agent,
                    coder_agent, 
                    documenter_agent,
                    presenter_agent,
                    voice_over_agent,
                    assignment_parser_agent,
                    presentation_verifier_agent
                    # file_writer_agent
                ],
                model_client=model_client,
                termination_condition=TextMentionTermination("TERMINATE"),
                selector_prompt=selector_prompt,
                allow_repeated_speaker=True,
                # selector_func=selector_func
            )
            
            logger.info(f"SelectorGroupChat created with {len(self.group_chat._participants)} agents")
        except ImportError as e:
            logger.error(f"ImportError setting up agents: {e}")
    
    def _read_file(self, path: str) -> str:
        """Read content from a file."""
        try:
            with open(self.output_dir / path, "r") as f:
                content = f.read()
            
            return content
        except Exception as e:
            return f"Error reading file: {e}"
    
    async def run(self, assignment_path: str) -> None:
        """Run the multi-agent system on the assignment."""
        # First parse the assignment
        # Create the assignment parser
        self.assignment_parser = AssignmentParser(
            model=DEFAULT_MODEL,
            temperature=DEFAULT_TEMPERATURE,
            workspace_dir=str(self.output_dir)
        )
        
        # Initialize MCPs
        if not await self.initialize_mcps():
            logger.warning("Some MCPs failed to initialize")
        
        # Set up agents
        await self.setup_agents()
        
        # Add a timestamp to avoid termination message detection in initial message
        timestamp = f"Timestamp: {os.path.getmtime(assignment_path)}"
        
        # Start the conversation
        initial_message = f"""
        I need you to complete the following assignment at this path:
        {assignment_path}
        
        Please coordinate the team to complete this assignment.
        
        IMPORTANT GUIDELINES:
        1. The AssignmentParserAgent should first read the assignment file
        2. The PlannerAgent should break down the assignment into tasks
        3. The CoderAgent MUST create actual code files (not markdown) with proper extensions (.py, .js, etc.)
        4. The DocumenterAgent should create documentation
        5. The PresenterAgent should create a presentation
        
        Make sure to create actual, working code files in the output directory.
        """
        
        # Run the conversation with SelectorGroupChat and Console
        logger.info("Starting agent conversation...")
        try:
            # Use Console with run_stream for interactive visualization
            await Console(self.group_chat.run_stream(task=initial_message))
            logger.info("Agent conversation completed")
        except Exception as e:
            # Catch any other exception during the chat execution
            logger.error(f"An error occurred during the agent conversation: {e}", exc_info=True)
        
async def main() -> None:
    """Main function to run the multi-agent system."""
    parser = argparse.ArgumentParser(description="Multi-Agent System for Assignment Completion")
    parser.add_argument("--assignment", required=True, help="Path to the assignment file")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to store output files")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    else:
        # Ensure ERROR level is enforced even if debug is not enabled
        logging.getLogger().setLevel(logging.ERROR)
        logger.setLevel(logging.ERROR)
    
    # Create and run the multi-agent system
    system = MultiAgentSystem(args.output_dir)
    await system.run(args.assignment)


if __name__ == "__main__":
    asyncio.run(main())