import os
import argparse
import json
import re
import logging
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

from autogen import AssistantAgent, UserProxyAgent
from autogen.agentchat.groupchat import GroupChat, GroupChatManager

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

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("main")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

DEFAULT_OUTPUT_DIR = "./output"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MODEL = "gpt-4o"
repo_name = "project_1-autogen"

# MCP API endpoints and scopes
GITHUB_API_URL = "https://api.github.com"
GITHUB_SCOPES = ["repo"]

GOOGLE_SLIDES_SCOPES = [
    'https://www.googleapis.com/auth/presentations',
    'https://www.googleapis.com/auth/drive'
]

ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1"

class MCPClient:
    """Base class for MCP (Machine Communication Protocol) clients."""
    
    def __init__(self):
        self.is_authenticated = False
        
    def authenticate(self) -> bool:
        """Authenticate with the MCP service."""
        raise NotImplementedError("Subclasses must implement authenticate()")


class GitHubMCPClient(MCPClient):
    """Client for interacting with GitHub API."""
    
    def __init__(self):
        super().__init__()
        self.github = None
        self.repo = None
        
    def authenticate(self) -> bool:
        try:
            self.github = Github(os.getenv("GITHUB_PAT"))
            user = self.github.get_user()
            logger.info(f"Authenticated with GitHub as {user.login}")
            self.is_authenticated = True
            return True
        except github.GithubException as e:
            logger.error(f"GitHub authentication failed: {e}")
            return False
    
    def get_or_create_repo(self) -> Optional[github.Repository.Repository]:
        """Get or create a repository for the project."""
        if not self.is_authenticated:
            logger.error("Not authenticated with GitHub")
            return None
        
        try:
            user = self.github.get_user()
            try:
                # Try to get existing repo
                self.repo = user.get_repo(repo_name)
                logger.info(f"Found existing repository: {repo_name}")
            except github.GithubException:
                # Create new repo if it doesn't exist
                self.repo = user.create_repo(
                    repo_name,
                    description=f"Auto-generated repository for {repo_name}"
                )
                logger.info(f"Created new repository: {repo_name}")
            
            return self.repo
        except github.GithubException as e:
            logger.error(f"Error accessing/creating repository: {e}")
            return None
    
    def create_file(self, path: str, content: str, commit_message: str) -> bool:
        """Create a file in the repository."""
        if not self.repo:
            logger.error("No repository selected")
            return False
        
        try:
            # Check if file exists
            try:
                contents = self.repo.get_contents(path)
                # Update file if it exists
                self.repo.update_file(
                    path=path,
                    message=commit_message,
                    content=content,
                    sha=contents.sha
                )
                logger.info(f"Updated file {path}")
            except github.GithubException:
                # Create file if it doesn't exist
                self.repo.create_file(
                    path=path,
                    message=commit_message,
                    content=content
                )
                logger.info(f"Created file {path}")
            return True
        except github.GithubException as e:
            logger.error(f"Error creating file {path}: {e}")
            return False


class GoogleSlidesMCPClient(MCPClient):
    """Client for interacting with Google Slides API."""
    
    def __init__(self):
        super().__init__()
        self.credentials = None
        self.slides_service = None
        self.drive_service = None
        self.presentation_id = None
    
    def authenticate(self) -> bool:
        try:
            creds = None
            token_file = Path("/home/erudman21/school/genai/GenAI-Multi-Agent-System-Demo/token.json")
            
            # Load credentials from token file if it exists
            if token_file.exists():
                creds = Credentials.from_authorized_user_info(
                    json.loads(token_file.read_text()),
                    GOOGLE_SLIDES_SCOPES
                )
            
            # If credentials don't exist or are invalid, get new ones
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        "/home/erudman21/school/genai/GenAI-Multi-Agent-System-Demo/credentials.json", 
                        GOOGLE_SLIDES_SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                
                # Save credentials for future use
                token_file.write_text(json.dumps({
                    'token': creds.token,
                    'refresh_token': creds.refresh_token,
                    'token_uri': creds.token_uri,
                    'client_id': creds.client_id,
                    'client_secret': creds.client_secret,
                    'scopes': creds.scopes,
                }))
            
            # Build services
            self.credentials = creds
            self.slides_service = build('slides', 'v1', credentials=creds)
            self.drive_service = build('drive', 'v3', credentials=creds)
            
            self.is_authenticated = True
            logger.info("Authenticated with Google Slides")
            return True
        except google.auth.exceptions.GoogleAuthError as e:
            logger.error(f"Google Slides authentication failed: {e}")
            return False
    
    def create_presentation(self, title: str) -> Optional[str]:
        """Create a new presentation."""
        if not self.is_authenticated:
            logger.error("Not authenticated with Google Slides")
            return None
        
        try:
            # Create a new presentation
            presentation = self.slides_service.presentations().create(
                body={'title': title}
            ).execute()
            self.presentation_id = presentation.get('presentationId')
            logger.info(f"Created presentation with ID: {self.presentation_id}")
            
            # Set permissions to anyone with the link can view
            self.drive_service.permissions().create(
                fileId=self.presentation_id,
                body={
                    'type': 'anyone',
                    'role': 'reader'
                },
                fields='id'
            ).execute()
            
            return self.presentation_id
        except Exception as e:
            logger.error(f"Error creating presentation: {e}")
            return None
    
    def add_slide(self, title: str, content: str) -> bool:
        """Add a slide to the presentation."""
        if not self.is_authenticated or not self.presentation_id:
            logger.error("Not authenticated or no presentation selected")
            return False
        
        try:
            # Generate unique object IDs (at least 5 characters long)
            title_id = f"title_{int(time.time())}"
            body_id = f"body_{int(time.time())}"
            
            # Create a new slide
            requests = [
                {
                    'createSlide': {
                        'slideLayoutReference': {
                            'predefinedLayout': 'TITLE_AND_BODY'
                        },
                        'placeholderIdMappings': [
                            {
                                'layoutPlaceholder': {
                                    'type': 'TITLE',
                                    'index': 0
                                },
                                'objectId': title_id
                            },
                            {
                                'layoutPlaceholder': {
                                    'type': 'BODY',
                                    'index': 0
                                },
                                'objectId': body_id
                            }
                        ]
                    }
                }
            ]
            
            response = self.slides_service.presentations().batchUpdate(
                presentationId=self.presentation_id,
                body={'requests': requests}
            ).execute()
            
            # Get the slide ID
            slide_id = response.get('replies')[0].get('createSlide').get('objectId')
            
            # Add title and content to the slide
            requests = [
                {
                    'insertText': {
                        'objectId': title_id,
                        'text': title
                    }
                },
                {
                    'insertText': {
                        'objectId': body_id,
                        'text': content
                    }
                }
            ]
            
            self.slides_service.presentations().batchUpdate(
                presentationId=self.presentation_id,
                body={'requests': requests}
            ).execute()
            
            logger.info(f"Added slide with title: {title}")
            return True
        except Exception as e:
            logger.error(f"Error adding slide: {e}")
            return False
    
    def get_presentation_url(self) -> Optional[str]:
        """Get the URL of the presentation."""
        if not self.presentation_id:
            logger.error("No presentation selected")
            return None
        
        return f"https://docs.google.com/presentation/d/{self.presentation_id}/edit"


class ElevenLabsMCPClient(MCPClient):
    """Client for interacting with ElevenLabs API."""
    
    def __init__(self):
        super().__init__()
        self.api_key = None
    
    def authenticate(self) -> bool:
        """Authenticate with ElevenLabs."""    
        try:
            headers = {
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                f"{ELEVENLABS_API_URL}/voices",
                headers=headers
            )
            
            if response.status_code == 200:
                self.api_key = ELEVENLABS_API_KEY
                self.is_authenticated = True
                logger.info("Authenticated with ElevenLabs")
                return True
            else:
                logger.error(f"ElevenLabs authentication failed: {response.text}")
                return False
        except requests.RequestException as e:
            logger.error(f"ElevenLabs authentication failed: {e}")
            return False
    
    def generate_voice_over(self, text: str, output_path: str) -> bool:
        """Generate a voice-over for the given text."""
        if not self.is_authenticated:
            logger.error("Not authenticated with ElevenLabs")
            return False
        
        try:
            headers = {
                "xi-api-key": self.api_key,
                "Content-Type": "application/json"
            }
            
            data = {
                "text": text,
                "model_id": os.getenv("ELEVENLABS_MODEL_ID", "eleven_monolingual_v2"),
                "voice_settings": {
                    "stability": os.getenv("ELEVENLABS_STABILITY", 0.5),
                    "similarity_boost": os.getenv("ELEVENLABS_SIMILARITY_BOOST", 0.75)
                }
            }
            
            response = requests.post(
                f"{ELEVENLABS_API_URL}/text-to-speech/{os.getenv('ELEVENLABS_VOICE_ID')}",
                headers=headers,
                json=data,
                stream=True
            )
            
            if response.status_code == 200:
                # Write audio content to file
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                
                logger.info(f"Generated voice-over at {output_path}")
                return True
            else:
                logger.error(f"Voice-over generation failed: {response.text}")
                return False
        except requests.RequestException as e:
            logger.error(f"Voice-over generation failed: {e}")
            return False

class AssignmentParser:
    """Parser for assignment text files."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.raw_text = None
        self.parsed_data = {
            "title": "",
            "description": "",
            "requirements": [],
            "deliverables": [],
            "due_date": "",
            "extensions": []
        }
    
    def parse(self) -> Dict[str, Any]:
        """Parse the assignment file."""
        try:
            with open(self.file_path, 'r') as f:
                self.raw_text = f.read()
            
            # Extract title (first line)
            lines = self.raw_text.split('\n')
            self.parsed_data["title"] = lines[0].strip()
            
            # Extract due date using regex
            due_date_match = re.search(r'due\s+([A-Za-z]+\s+[A-Za-z]+\s+\d+\s+\d+:\d+[apm]{2})', self.raw_text, re.IGNORECASE)
            if due_date_match:
                self.parsed_data["due_date"] = due_date_match.group(1)
            
            # Extract requirements
            req_start = self.raw_text.find("Requirements")
            if req_start != -1:
                req_text = self.raw_text[req_start:]
                ext_start = req_text.find("Extensions")
                
                if ext_start != -1:
                    req_text = req_text[:ext_start]
                
                # Extract bullet points or numbered list items
                requirements = re.findall(r'[\•\-\*]\s+(.+?)(?=[\•\-\*]|\n\n|$)', req_text, re.DOTALL)
                if not requirements:
                    requirements = re.findall(r'\d+\.\s+(.+?)(?=\d+\.\s+|\n\n|$)', req_text, re.DOTALL)
                
                self.parsed_data["requirements"] = [req.strip() for req in requirements if req.strip()]
            
            # Extract extensions
            ext_start = self.raw_text.find("Extensions")
            if ext_start != -1:
                ext_text = self.raw_text[ext_start:]
                
                # Extract bullet points or numbered list items
                extensions = re.findall(r'[\•\-\*]\s+(.+?)(?=[\•\-\*]|\n\n|$)', ext_text, re.DOTALL)
                if not extensions:
                    extensions = re.findall(r'\d+\.\s+(.+?)(?=\d+\.\s+|\n\n|$)', ext_text, re.DOTALL)
                
                self.parsed_data["extensions"] = [ext.strip() for ext in extensions if ext.strip()]
            
            # Extract description (everything between title and requirements)
            desc_start = len(self.parsed_data["title"]) + 1
            desc_end = req_start if req_start != -1 else len(self.raw_text)
            self.parsed_data["description"] = self.raw_text[desc_start:desc_end].strip()
            
            # Extract deliverables from description
            deliv_start = self.parsed_data["description"].find("Deliverables")
            if deliv_start != -1:
                deliv_text = self.parsed_data["description"][deliv_start:]
                next_section = re.search(r'\n\n[A-Z][a-z]+', deliv_text)
                
                if next_section:
                    deliv_text = deliv_text[:next_section.start()]
                
                # Remove "Deliverables" header
                deliv_text = re.sub(r'^Deliverables.*?\n', '', deliv_text).strip()
                
                # Split by newlines and filter empty lines
                deliverables = [line.strip() for line in deliv_text.split('\n') if line.strip()]
                self.parsed_data["deliverables"] = deliverables
            
            return self.parsed_data
        except Exception as e:
            logger.error(f"Error parsing assignment file: {e}")
            return self.parsed_data


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
        
        # Initialize agents
        self.agents = {}
        self.termination_msgs = ["TERMINATE", "TASK_COMPLETE", "DONE"]
        
        self.assignment_data = {}
        self.task_status = {
            "coding": False,
            "documentation": False,
            "presentation": False,
            "voice_over": False
        }
    
    def initialize_mcps(self) -> bool:
        """Initialize and authenticate all MCPs."""
        github_auth = self.github_mcp.authenticate()
        slides_auth = self.slides_mcp.authenticate()
        elevenlabs_auth = self.elevenlabs_mcp.authenticate()
        
        # Log authentication status
        logger.info(f"GitHub MCP: {'Authenticated' if github_auth else 'Not authenticated or disabled'}")
        logger.info(f"Google Slides MCP: {'Authenticated' if slides_auth else 'Not authenticated or disabled'}")
        logger.info(f"ElevenLabs MCP: {'Authenticated' if elevenlabs_auth else 'Not authenticated or disabled'}")
        
        return github_auth or slides_auth or elevenlabs_auth
    
    # Replace the _create_agent_function_map method and setup_agents method with these versions
    def _create_function_schemas(self) -> List[Dict[str, Any]]:
        """Create a list of function schemas without the actual functions."""
        function_schemas = []
        
        # GitHub functions
        if self.github_mcp.is_authenticated:
            function_schemas.append({
                "name": "github_create_file",
                "description": "Create a file in the GitHub repository",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file in the repository"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content of the file"
                        },
                        "message": {
                            "type": "string",
                            "description": "Commit message"
                        }
                    },
                    "required": ["path", "content"]
                }
            })
        
        # Google Slides functions
        if self.slides_mcp.is_authenticated:
            function_schemas.append({
                "name": "slides_add_slide",
                "description": "Add a slide to the presentation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Title of the slide"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content of the slide"
                        }
                    },
                    "required": ["title", "content"]
                }
            })
            
            function_schemas.append({
                "name": "slides_get_url",
                "description": "Get the URL of the presentation",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            })
        
        # ElevenLabs functions
        if self.elevenlabs_mcp.is_authenticated:
            function_schemas.append({
                "name": "elevenlabs_generate_voice_over",
                "description": "Generate a voice-over for the given text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to convert to speech"
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Path to save the audio file"
                        }
                    },
                    "required": ["text", "output_path"]
                }
            })
        
        # File functions
        function_schemas.append({
            "name": "file_write",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    },
                    "mode": {
                        "type": "string",
                        "description": "File mode (w for write, a for append)",
                        "enum": ["w", "a"]
                    }
                },
                "required": ["path", "content"]
            }
        })
        
        function_schemas.append({
            "name": "file_read",
            "description": "Read content from a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file"
                    }
                },
                "required": ["path"]
            }
        })
        
        return function_schemas
    
    def safer_termination_check(self, msg):
        """A safer termination check that properly handles different message formats."""
        if msg is None:
            return False
            
        # If msg is a dictionary, try to get its content
        if isinstance(msg, dict):
            content = msg.get("content", "")
            # If content is None or not a string, return False
            if content is None or not isinstance(content, str):
                return False
            # Check if any termination message appears in the content
            return any(term in content for term in ["TERMINATE", "TASK_COMPLETE", "DONE"])
        
        # If msg is a string, check directly
        if isinstance(msg, str):
            return any(term in msg for term in ["TERMINATE", "TASK_COMPLETE", "DONE"])
        
        # For other types, return False
        return False

    def setup_agents(self) -> None:
        """Set up the agents for the system."""
        # Create function schemas (without the actual function implementations)
        function_schemas = self._create_function_schemas()
        # Set up LLM configuration for agents with function schemas
        llm_config = {
            "config_list": [{
                "model": "gpt-4o",
                "api_key": os.environ.get("OPENAI_API_KEY"),
            }],
            "functions": function_schemas
        }
        
        # Set up LLM configuration for GroupChatManager without functions
        manager_llm_config = {
            "config_list": [{
                "model": "gpt-4o",
                "api_key": os.environ.get("OPENAI_API_KEY"),
            }]
        }
        
        # Create code directory if it doesn't exist
        code_dir = self.output_dir / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        
        # Define agent for coding tasks
        self.agents["coder"] = AssistantAgent(
            name="CoderAgent",
            system_message="""You are an expert software developer responsible for implementing the coding part of assignments.
            Your primary focus is on producing clean, efficient, and well-documented code that meets all requirements.
            You should prioritize creating a working prototype before adding advanced features or optimizations.
            When writing code, include comprehensive comments to explain your approach and any non-trivial logic.
            For web applications, focus on both frontend and backend components, ensuring they work together seamlessly.
            Use best practices for the languages and frameworks specified in the assignment.
            Break down complex tasks into manageable components and implement them systematically.
            
            When you complete your part of the assignment, clearly state "CODING COMPLETE" in your message.
            
            You can use functions to write code to files and manage the GitHub repository.
            """,
            llm_config=llm_config
        )
        
        # Define agent for documentation
        self.agents["documenter"] = AssistantAgent(
            name="DocumenterAgent",
            system_message="""You are an expert technical writer responsible for creating clear, comprehensive documentation.
            Your documentation should cover installation instructions, usage examples, API references, and architectural overview.
            Use markdown format for all documentation, with proper headings, lists, and code blocks.
            Create a main README.md file that provides an overview of the project and links to other documentation.
            Include diagrams or illustrations where they can clarify complex concepts or workflows.
            Document both the public API and internal architecture to aid future development.
            Focus on making the documentation accessible to users with varying levels of technical expertise.
            For each component or feature, explain both how to use it and why certain design decisions were made.
            
            When you complete your part of the assignment, clearly state "DOCUMENTATION COMPLETE" in your message.
            
            You can use functions to write documentation files.
            """,
            llm_config=llm_config
        )
        
        # Define agent for presentation creation
        self.agents["presenter"] = AssistantAgent(
            name="PresenterAgent",
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
            
            When you complete your part of the assignment, clearly state "PRESENTATION COMPLETE" in your message.
            
            You can use functions to create slides in the presentation.
            """,
            llm_config=llm_config
        )
        
        # Define agent for voice-over creation
        self.agents["voice_over"] = AssistantAgent(
            name="VoiceOverAgent",
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
            
            When you complete your part of the assignment, clearly state "VOICE-OVER COMPLETE" in your message.
            
            You can use functions to generate voice-overs for the presentation.
            """,
            llm_config=llm_config
        )
        
        # Define orchestrator agent
        self.agents["orchestrator"] = AssistantAgent(
            name="OrchestratorAgent",
            system_message="""You are the project manager responsible for coordinating the efforts of the team.
            Your role is to break down the assignment into tasks and assign them to the appropriate agents.
            Monitor the progress of each task and ensure that all parts of the project are completed on time.
            Facilitate communication between agents when dependencies exist between their tasks.
            Provide guidance on priorities and resolve any conflicts or ambiguities in the requirements.
            Ensure that all deliverables meet the requirements specified in the assignment.
            At the end of the project, compile all outputs into a coherent, integrated solution.
            Make sure to validate that each component works correctly and integrates well with others.
            Keep track of the overall project timeline and adjust task assignments if necessary.
            Your final output should be a comprehensive report on the completed project.
            
            You can use functions to coordinate the project and track progress.
            
            IMPORTANT: When the entire project is finished and all components are complete,
            include the exact phrase "TASK_COMPLETE" in your message to signal completion.
            """,
            llm_config=llm_config
        )
        
        # User proxy agent that can execute code and use tools
        self.agents["user_proxy"] = UserProxyAgent(
            name="UserProxy",
            is_termination_msg=self.safer_termination_check,
            human_input_mode="TERMINATE",
            code_execution_config={
                "work_dir": str(code_dir),
                "use_docker": False
            }
        )
        
        # Register function implementations with each agent
        self._register_functions_with_agents()
        
        # Create a group chat for the agents
        self.group_chat = GroupChat(
            agents=[
                self.agents["orchestrator"],
                self.agents["coder"],
                self.agents["documenter"],
                self.agents["presenter"],
                self.agents["voice_over"],
                self.agents["user_proxy"]
            ],
            messages=[],
            max_round=50
        )
        
        # Create a group chat manager with a configuration without functions
        self.manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=manager_llm_config
        )
        
        logger.info(f"Group chat created with {len(self.group_chat.agents)} agents")

    # Replace the _register_functions_with_agents method to ensure all functions return strings

    def _register_functions_with_agents(self) -> None:
        """Register function implementations with each agent."""
        # GitHub functions
        if self.github_mcp.is_authenticated:
            def github_create_file(path, content, message="Add file via agent"):
                result = self.github_mcp.create_file(path, content, message)
                # Convert boolean result to string message
                if result:
                    return f"Successfully created file: {path}"
                else:
                    return f"Failed to create file: {path}"
            
            for agent_name, agent in self.agents.items():
                if hasattr(agent, "register_function"):
                    agent.register_function(
                        function_map={"github_create_file": github_create_file}
                    )
        
        # Google Slides functions
        if self.slides_mcp.is_authenticated:
            def slides_add_slide(title, content):
                result = self.slides_mcp.add_slide(title, content)
                # Convert boolean result to string message
                if result:
                    return f"Successfully added slide with title: {title}"
                else:
                    return f"Failed to add slide with title: {title}"
            
            def slides_get_url():
                url = self.slides_mcp.get_presentation_url()
                if url:
                    return f"Presentation URL: {url}"
                else:
                    return "No presentation URL available."
            
            for agent_name, agent in self.agents.items():
                if hasattr(agent, "register_function"):
                    agent.register_function(
                        function_map={
                            "slides_add_slide": slides_add_slide,
                            "slides_get_url": slides_get_url
                        }
                    )
        
        # ElevenLabs functions
        if self.elevenlabs_mcp.is_authenticated:
            def elevenlabs_generate_voice_over(text, output_path):
                result = self.elevenlabs_mcp.generate_voice_over(text, output_path)
                # Convert boolean result to string message
                if result:
                    return f"Successfully generated voice-over at: {output_path}"
                else:
                    return f"Failed to generate voice-over at: {output_path}"
            
            for agent_name, agent in self.agents.items():
                if hasattr(agent, "register_function"):
                    agent.register_function(
                        function_map={"elevenlabs_generate_voice_over": elevenlabs_generate_voice_over}
                    )
        
        # File functions
        def file_write(path, content, mode="w"):
            result = self._write_file(path, content, mode)
            # The _write_file method already returns a string, so we don't need to convert
            return result
        
        def file_read(path):
            result = self._read_file(path)
            # The _read_file method already returns a string, so we don't need to convert
            return result
        
        for agent_name, agent in self.agents.items():
            if hasattr(agent, "register_function"):
                agent.register_function(
                    function_map={
                        "file_write": file_write,
                        "file_read": file_read
                    }
                )
    
    def _write_file(self, path: str, content: str, mode: str = "w") -> str:
        """Write content to a file."""
        try:
            # Make sure the directory exists
            file_path = self.output_dir / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, mode) as f:
                f.write(content)
            
            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error writing to file: {e}"
    
    def _read_file(self, path: str) -> str:
        """Read content from a file."""
        try:
            with open(self.output_dir / path, "r") as f:
                content = f.read()
            
            return content
        except Exception as e:
            return f"Error reading file: {e}"
    
    def parse_assignment(self, assignment_path: str) -> Dict[str, Any]:
        """Parse the assignment file."""
        parser = AssignmentParser(assignment_path)
        self.assignment_data = parser.parse()
        return self.assignment_data
    
    def run(self, assignment_path: str) -> None:
        """Run the multi-agent system on the assignment."""
        # Parse assignment
        assignment_data = self.parse_assignment(assignment_path)
        if not assignment_data["title"]:
            logger.error("Failed to parse assignment or title is empty")
            return
        
        # Initialize MCPs
        if not self.initialize_mcps():
            logger.warning("Some MCPs failed to initialize")
        
        # Set up agents
        self.setup_agents()
        
        # Set up GitHub repository
        if self.github_mcp.is_authenticated:
            repo = self.github_mcp.get_or_create_repo()
            if not repo:
                logger.error("Failed to set up GitHub repository")
        
        # Create initial presentation
        if self.slides_mcp.is_authenticated:
            presentation_id = self.slides_mcp.create_presentation(f"Presentation: {assignment_data['title']}")
            if not presentation_id:
                logger.error("Failed to create initial presentation")
        
        # Add a timestamp to avoid termination message detection in initial message
        timestamp = f"Timestamp: {os.path.getmtime(assignment_path)}"
        
        # Start the conversation
        initial_message = f"""
        I need you to complete the following assignment:
        
        TITLE: {assignment_data['title']}
        
        DESCRIPTION:
        {assignment_data['description']}
        
        REQUIREMENTS:
        {os.linesep.join('- ' + req for req in assignment_data['requirements'])}
        
        DELIVERABLES:
        {os.linesep.join('- ' + deliv for deliv in assignment_data['deliverables'])}
        
        DUE DATE: {assignment_data['due_date']}
        
        EXTENSIONS (OPTIONAL):
        {os.linesep.join('- ' + ext for ext in assignment_data['extensions'])}
        
        {timestamp}
        
        Please coordinate the team to complete this assignment. The CoderAgent should implement the code, 
        the DocumenterAgent should create documentation, the PresenterAgent should create a presentation, 
        and the VoiceOverAgent should create voice-over scripts for the presentation.
        
        Start by breaking down the assignment into specific tasks for each team member.
        The conversation should only conclude when all tasks are fully implemented and 
        you explicitly mention "TASK_COMPLETE" to indicate everything is done.
        """
        
        # Debug: test if the termination condition would trigger
        test_msg = {"content": initial_message}
        logger.debug(f"Would terminate on initial message: {self.agents['user_proxy']._is_termination_msg(test_msg)}")
        
        # Run the conversation with retry logic
        logger.info("Starting agent conversation...")
        try:
            # Use the user proxy to initiate the chat with the orchestrator
            self.agents["user_proxy"].initiate_chat(
                self.manager,
                message=initial_message
            )
            logger.info("Agent conversation completed")
        except Exception as e:
            # Catch any other exception during the chat execution
            logger.error(f"An error occurred during the agent conversation: {e}", exc_info=True)
            # You might still want the overall retry loop here if needed for setup errors,
            # but it's less useful for runtime rate limits if the library handles them.
            # The original loop from your code could be kept or removed depending on
            # whether you need to retry the *entire* process upon failure.
            
        self._process_results()
    
    def _process_results(self) -> None:
        """Process the results of the conversation and generate final outputs."""
        logger.info("Processing results...")
        
        # Examine chat history to determine task status
        if hasattr(self.group_chat, "messages") and self.group_chat.messages:
            for message in self.group_chat.messages:
                if not isinstance(message, dict) or "content" not in message:
                    continue
                    
                content = message.get("content", "").lower()
                
                # Track task completion by keywords in messages
                if "coding complete" in content or "code implementation complete" in content:
                    self.task_status["coding"] = True
                
                if "documentation complete" in content or "documentation finished" in content:
                    self.task_status["documentation"] = True
                
                if "presentation complete" in content or "slides complete" in content:
                    self.task_status["presentation"] = True
                
                if "voice-over complete" in content or "audio files generated" in content:
                    self.task_status["voice_over"] = True
        
        # Alternatively, check for completed files as indicators of task status
        code_files = list((self.output_dir / "code").glob("**/*"))
        if code_files:
            self.task_status["coding"] = True
        
        doc_files = list(self.output_dir.glob("**/*.md"))
        if doc_files:
            self.task_status["documentation"] = True
        
        if self.slides_mcp.is_authenticated and self.slides_mcp.presentation_id:
            self.task_status["presentation"] = True
        
        audio_files = list(self.output_dir.glob("**/*.mp3"))
        if audio_files:
            self.task_status["voice_over"] = True
        
        # Log completion status
        logger.info(f"Task status: {self.task_status}")
        
        # Gather all files created in the output directory
        output_files = list(self.output_dir.glob("**/*"))
        logger.info(f"Generated {len(output_files)} files in output directory")
        
        # Get presentation URL if available
        if self.slides_mcp.is_authenticated and self.slides_mcp.presentation_id:
            presentation_url = self.slides_mcp.get_presentation_url()
            logger.info(f"Presentation available at: {presentation_url}")
        
        # Create final summary report
        summary = f"""
        # Assignment Completion Report
        
        ## Assignment: {self.assignment_data['title']}
        
        ## Status
        - Coding: {'Completed' if self.task_status['coding'] else 'Incomplete'}
        - Documentation: {'Completed' if self.task_status['documentation'] else 'Incomplete'}
        - Presentation: {'Completed' if self.task_status['presentation'] else 'Incomplete'}
        - Voice-over: {'Completed' if self.task_status['voice_over'] else 'Incomplete'}
        
        ## Generated Files
        {os.linesep.join('- ' + str(f.relative_to(self.output_dir)) for f in output_files)}
        
        ## Resources
        """
        
        if self.github_mcp.is_authenticated and self.github_mcp.repo:
            summary += f"- GitHub Repository: {self.github_mcp.repo.html_url}\n"
        
        if self.slides_mcp.is_authenticated and self.slides_mcp.presentation_id:
            summary += f"- Presentation: {self.slides_mcp.get_presentation_url()}\n"
        
        # Write summary to file
        with open(self.output_dir / "completion_report.md", "w") as f:
            f.write(summary)
        
        logger.info("Results processing complete.")


def main() -> None:
    """Main function to run the multi-agent system."""
    parser = argparse.ArgumentParser(description="Multi-Agent System for Assignment Completion")
    parser.add_argument("--assignment", required=True, help="Path to the assignment file")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to store output files")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Create and run the multi-agent system
    system = MultiAgentSystem(args.output_dir)
    system.run(args.assignment)


if __name__ == "__main__":
    main()