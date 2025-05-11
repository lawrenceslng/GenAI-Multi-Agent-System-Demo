import os
import asyncio
import logging
import time
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
import autogen

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("presentation_agent")

load_dotenv()

@dataclass
class SlideContent:
    """Data class for slide content"""
    title: str
    content: List[str]
    speaker_notes: str = ""

class PresentationWorkflow:
    """
    Manages the sequential workflow for creating presentations.
    Handles the entire process as an atomic operation with proper error handling.
    """
    
    def __init__(self, slides_workbench, logger=None):
        """Initialize the presentation workflow with required components"""
        self.slides_workbench = slides_workbench
        self.logger = logger or logging.getLogger("presentation_workflow")
        self.presentation_id = None
        self.slides_created = []
        self.errors = []
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        after=lambda retry_state: logging.info(f"Retry {retry_state.attempt_number} after error")
    )
    async def create_presentation(self, title: str) -> str:
        """
        Create a new presentation with retry logic
        
        Args:
            title: Presentation title
            
        Returns:
            presentation_id: ID of the created presentation
        """
        self.logger.info(f"Creating presentation titled: {title}")
        try:
            # Get the create_presentation tool
            tools = await self.slides_workbench.list_tools()
            create_presentation_tool = None
            for tool in tools:
                if tool.name == "create_presentation":
                    create_presentation_tool = tool
                    break
                    
            if not create_presentation_tool:
                raise ValueError("create_presentation tool not found")
                
            # Call the tool
            result = await create_presentation_tool.acall(title=title)
            
            # Extract presentation ID from result
            if hasattr(result, 'content'):
                content = result.content
            else:
                content = str(result)
                
            # Handle different result formats
            if isinstance(content, str):
                if content.startswith("{"):
                    # It's a JSON string
                    data = json.loads(content)
                    presentation_id = data.get("presentationId")
                else:
                    # Try to extract ID with regex
                    match = re.search(r'"presentationId"\s*:\s*"([^"]+)"', content)
                    presentation_id = match.group(1) if match else None
            elif isinstance(content, dict):
                presentation_id = content.get("presentationId")
            else:
                raise ValueError(f"Unexpected response format: {content}")
                
            if not presentation_id:
                raise ValueError("Failed to extract presentation ID from response")
                
            self.presentation_id = presentation_id
            self.logger.info(f"Successfully created presentation with ID: {presentation_id}")
            return presentation_id
            
        except Exception as e:
            self.logger.error(f"Failed to create presentation: {str(e)}")
            self.errors.append(f"Create presentation error: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    async def get_presentation_details(self, presentation_id: str) -> Dict[str, Any]:
        """
        Get details about the presentation
        
        Args:
            presentation_id: ID of the presentation
            
        Returns:
            Dict containing presentation details
        """
        self.logger.info(f"Getting details for presentation: {presentation_id}")
        try:
            # Get the get_presentation tool
            tools = await self.slides_workbench.list_tools()
            get_presentation_tool = None
            for tool in tools:
                if tool.name == "get_presentation":
                    get_presentation_tool = tool
                    break
                    
            if not get_presentation_tool:
                raise ValueError("get_presentation tool not found")
                
            # Call the tool
            result = await get_presentation_tool.acall(presentationId=presentation_id)
            
            # Handle different result formats
            if hasattr(result, 'content'):
                content = result.content
            else:
                content = str(result)
                
            if isinstance(content, str):
                if content.startswith("{"):
                    content = json.loads(content)
            
            self.logger.info(f"Successfully retrieved presentation details")
            return content
        
        except Exception as e:
            self.logger.error(f"Failed to get presentation details: {str(e)}")
            self.errors.append(f"Get presentation details error: {str(e)}")
            raise
    
    async def create_slide(self, slide_content: SlideContent) -> str:
        """
        Create a single slide with retries
        
        Args:
            slide_content: Content for the slide
            
        Returns:
            slide_id: ID of the created slide
        """
        if not self.presentation_id:
            raise ValueError("No presentation ID. Create presentation first.")
            
        self.logger.info(f"Creating slide: {slide_content.title}")
        
        # Generate a unique ID for this slide
        slide_id = f"slide_{int(time.time())}_{len(self.slides_created)}"
        
        # Get current slides count to determine insertion index
        try:
            presentation_details = await self.get_presentation_details(self.presentation_id)
            slides_count = len(presentation_details.get("slides", []))
        except Exception:
            # Default to end if we can't determine
            slides_count = 0
            
        # Prepare the batch update request
        try:
            # Step 1: Create the slide structure
            create_slide_result = await self._create_slide_structure(slide_id, slides_count)
            
            # Step 2: Add content to the slide
            content_result = await self._add_slide_content(slide_id, slide_content)
            
            self.slides_created.append(slide_id)
            self.logger.info(f"Successfully created slide: {slide_id}")
            return slide_id
            
        except Exception as e:
            self.logger.error(f"Failed to create slide: {str(e)}")
            self.errors.append(f"Create slide error: {str(e)}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
    async def _create_slide_structure(self, slide_id: str, insertion_index: int) -> Dict[str, Any]:
        """Create the structure for a slide"""
        title_id = f"{slide_id}_title"
        body_id = f"{slide_id}_body"
        
        requests = [{
            "createSlide": {
                "objectId": slide_id,
                "insertionIndex": insertion_index,
                "slideLayoutReference": {
                    "predefinedLayout": "TITLE_AND_BODY"
                },
                "placeholderIdMappings": [
                    {
                        "layoutPlaceholder": {
                            "type": "TITLE",
                            "index": 0
                        },
                        "objectId": title_id
                    },
                    {
                        "layoutPlaceholder": {
                            "type": "BODY",
                            "index": 0
                        },
                        "objectId": body_id
                    }
                ]
            }
        }]
        
        # Get the batch_update_presentation tool
        tools = await self.slides_workbench.list_tools()
        batch_update_tool = None
        for tool in tools:
            if tool.name == "batch_update_presentation":
                batch_update_tool = tool
                break
                
        if not batch_update_tool:
            raise ValueError("batch_update_presentation tool not found")
            
        # Call the tool
        result = await batch_update_tool.acall(
            presentationId=self.presentation_id,
            requests=requests
        )
        
        return result
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
    async def _add_slide_content(self, slide_id: str, content: SlideContent) -> Dict[str, Any]:
        """Add content to a slide"""
        title_id = f"{slide_id}_title"
        body_id = f"{slide_id}_body"
        
        # Format body content as bullet points if it's a list
        body_text = ""
        if isinstance(content.content, list):
            body_text = "\n".join(f"• {item}" for item in content.content)
        else:
            body_text = content.content
        
        # Create request to add title and content
        requests = [
            {
                "insertText": {
                    "objectId": title_id,
                    "text": content.title
                }
            },
            {
                "insertText": {
                    "objectId": body_id,
                    "text": body_text
                }
            }
        ]
        
        # Add speaker notes if provided
        if content.speaker_notes:
            requests.append({
                "createSpeakerNotesText": {
                    "speakerNotesObjectId": f"{slide_id}_notes",
                    "text": content.speaker_notes
                }
            })
        
        # Get the batch_update_presentation tool
        tools = await self.slides_workbench.list_tools()
        batch_update_tool = None
        for tool in tools:
            if tool.name == "batch_update_presentation":
                batch_update_tool = tool
                break
                
        if not batch_update_tool:
            raise ValueError("batch_update_presentation tool not found")
            
        # Call the tool
        result = await batch_update_tool.acall(
            presentationId=self.presentation_id,
            requests=requests
        )
        
        return result
    
    async def run_complete_workflow(self, presentation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete presentation creation workflow
        
        Args:
            presentation_data: Dict containing presentation title and slides content
            
        Returns:
            Dict with status and results
        """
        self.logger.info("Starting presentation creation workflow")
        start_time = time.time()
        
        try:
            # Step 1: Create the presentation
            presentation_id = await self.create_presentation(presentation_data["title"])
            self.presentation_id = presentation_id
            
            # Step 2: Get presentation details to understand current state
            presentation_details = await self.get_presentation_details(presentation_id)
            
            # Step 3: Create all slides
            for slide_data in presentation_data["slides"]:
                slide_content = SlideContent(
                    title=slide_data["title"],
                    content=slide_data["content"],
                    speaker_notes=slide_data.get("speaker_notes", "")
                )
                await self.create_slide(slide_content)
                
                # Small delay between slides to avoid rate limiting
                await asyncio.sleep(0.5)
            
            # Step 4: Get the presentation URL
            presentation_url = f"https://docs.google.com/presentation/d/{presentation_id}/edit"
            
            duration = time.time() - start_time
            self.logger.info(f"Workflow completed successfully in {duration:.2f} seconds")
            
            return {
                "status": "success",
                "presentation_id": presentation_id,
                "slides_created": len(self.slides_created),
                "duration_seconds": duration,
                "url": presentation_url,
                "errors": self.errors if self.errors else None
            }
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Workflow failed after {duration:.2f} seconds: {str(e)}")
            
            return {
                "status": "error",
                "error": str(e),
                "presentation_id": self.presentation_id,
                "slides_created": len(self.slides_created),
                "duration_seconds": duration,
                "errors": self.errors
            }

class PresentationAgent:
    def __init__(self):
        # Initialize clients and workbenches as None - will set up later
        self.model_client = None
        self.slides_workbench = None
        self.github_workbench = None
        self.elevenlabs_workbench = None
        
        # Initialize LLM config
        self.llm_config = {
            "config_list": [{
                "model": "gpt-4o-mini",
                "api_key": os.getenv("OPENAI_API_KEY")
            }],
        }
        
        # Will initialize agents after tools are loaded
        self.researcher = None
        self.presentation_creator = None
        self.voiceover_creator = None
        # self.user_proxy = None
    
    async def initialize(self):
        """Initialize workbenches, tools, and agents"""
        try:
            # Initialize OpenAI client first
            self.model_client = OpenAIChatCompletionClient(
                model="gpt-4o-mini",
            )
            
            # Initialize all workbenches
            await self.initialize_workbenches()
            
            # Initialize agents with tools
            await self.initialize_agents()
            
            logger.info("System initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            await self.cleanup()
            raise
    
    async def initialize_workbenches(self):
        """Initialize all MCP workbenches"""
        try:
            # Initialize GitHub workbench
            logger.info("Initializing GitHub workbench...")
            github_params = StdioServerParams(
                command=os.getenv("GITHUB_MCP_URL"),
                args=["stdio"],
                env={
                    "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PAT")
                }
            )
            self.github_workbench = McpWorkbench(server_params=github_params)
            await self.github_workbench.start()
            
            # Initialize Google Slides workbench
            logger.info("Initializing Google Slides workbench...")
            slides_server_path = os.path.join(
                "/home/erudman21/school/genai/GenAI-Multi-Agent-System-Demo/mcp_servers/google-slides-mcp/build/index.js"
            )
            slides_params = StdioServerParams(
                command="node",
                args=[
                    slides_server_path,
                    "--google-client-id", os.getenv("GOOGLE_CLIENT_ID"),
                    "--google-client-secret", os.getenv("GOOGLE_CLIENT_SECRET"),
                    "--google-refresh-token", os.getenv("GOOGLE_REFRESH_TOKEN")
                ]
            )
            self.slides_workbench = McpWorkbench(server_params=slides_params)
            await self.slides_workbench.start()
            
            # Initialize ElevenLabs workbench
            logger.info("Initializing ElevenLabs workbench...")
            elevenlabs_params = StdioServerParams(
                command="uv",
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
            self.elevenlabs_workbench = McpWorkbench(server_params=elevenlabs_params)
            await self.elevenlabs_workbench.start()
            
            logger.info("All workbenches initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize workbenches: {e}")
            await self.cleanup()
            raise
    
    async def initialize_agents(self):
        """Initialize agents with tools"""
        try:
            # Create GitHub tools
            github_tools = await self.github_workbench.list_tools()
            logger.info(f"Created {len(github_tools)} GitHub tools")
            
            # Create Slides tools
            slides_tools = await self.slides_workbench.list_tools()
            logger.info(f"Created {len(slides_tools)} Slides tools")
            
            # Create ElevenLabs tools
            elevenlabs_tools = await self.elevenlabs_workbench.list_tools()
            logger.info(f"Created {len(elevenlabs_tools)} ElevenLabs tools")
            
            # Create Researcher Agent with GitHub tools
            self.researcher = AssistantAgent(
                name="researcher",
                model_client=self.model_client,
                workbench=self.github_workbench,
                description="Expert at researching GitHub repositories and gathering information",
                system_message=f"""You are a skilled researcher who finds relevant information from GitHub repositories.
                Use the available GitHub tools to gather information about the repository.
                
                You have access to the following tools to research the GitHub repository:
                
                {github_tools}
                
                Important GitHub Tool Tips:
                - Use search_repositories to find repository information
                - Use get_file_contents to read specific files (especially README.md)
                  * Format: owner/repo/path (e.g. "robot-coder/assignment-project/README.md")
                - Use list_commits to see recent changes
                  * Format: owner/repo
                  
                When research is complete, clearly state "RESEARCH COMPLETE".
                
                Always provide the full repository path including owner and repo name when using GitHub tools.
                """,
                reflect_on_tool_use=True
            )
            
            # Create Presentation Creator Agent
            self.presentation_creator = AssistantAgent(
                name="presentation_creator",
                model_client=self.model_client,
                workbench=self.slides_workbench,
                description="Expert at creating Google Slides presentations",
                system_message=f"""You are an expert at creating Google Slides presentations.
                Your task is to create a professional presentation based on the research provided.
                You have access to the following tools to create the presentation in Google Slides:
                
                {slides_tools}
                
                 Important Google Slides Tool Tips:
                - FIRST use `create_presentation` to create a new presentation.
                - Save the `presentationId` returned by `create_presentation` for all subsequent calls.
                - Before adding any new slides or content, call `get_presentation` to understand the current state, including the number of slides, their `objectId`s, and the `objectId`s of any placeholder elements on the default first slide.

                For each new slide (beyond the first default one), generate a truly unique object ID:
                * ALWAYS include a timestamp or random number for each slide (e.g., "slide_overview_{{timestamp}}", "slide_features_random123").
                * NEVER reuse the same ID twice.

                **CRITICAL EXECUTION FLOW FOR MODIFYING THE PRESENTATION:**
                You MUST use `batch_update_presentation` for ANY action that creates a slide or adds/changes content on a slide.

                **Overall Workflow for Building the Presentation:**
                1.  Use `create_presentation` to get a `presentationId`. Report this ID.
                2.  Call `get_presentation` to get the `objectId` of the initial default slide (usually "p").
                3.  **Populate the Title Slide (e.g., slide "p"):**
                    * State your intent: "Populating Title Slide (ID: 'p') with provided title and subtitle."
                    * IMMEDIATELY IN THIS SAME RESPONSE, make ONE `batch_update_presentation` call with all necessary `insertText` requests for the title and subtitle on slide "p".
                        * Example `requests` for Title Slide content: `[{{ "insertText": {{"objectId": "p", "placeholder": {{"type": "CENTERED_TITLE"}}, "text": "Your Main Presentation Title"}} }}, {{ "insertText": {{"objectId": "p", "placeholder": {{"type": "SUBTITLE"}}, "text": "Your Presentation Subtitle"}} }}]`
                    * Report success or failure of this tool call. WAIT for confirmation/next instruction only if the interaction model requires it, otherwise proceed if you have the next piece of information.

                4.  **For EACH subsequent content slide you need to add (e.g., Overview, Features, etc.), perform the following two sub-steps IN IMMEDIATE SUCCESSION for THAT SLIDE before moving to the next planned slide:**
                    
                    **Sub-step A: Announce and Create Slide Structure**
                        i.  State your intent: "Creating structure for '[Name of Slide, e.g., Project Overview]' slide."
                        ii. Determine a **new, unique `objectId`** for this slide (e.g., `slide_overview_{{timestamp}}`).
                        iii.Determine the `insertionIndex` (this will be the current total number of slides, e.g., if 1 slide exists, next index is 1; if 2 slides exist, next index is 2, and so on).
                        iv. Determine the appropriate `slideLayoutReference` (e.g., `TITLE_AND_BODY`).
                        v.  IMMEDIATELY IN THIS SAME RESPONSE, make ONE `batch_update_presentation` call with a **single `createSlide` request**.
                            * Example: `requests: [{{ "createSlide": {{"objectId": "your_unique_slide_id_here", "insertionIndex": current_total_slides, "slideLayoutReference": {{"predefinedLayout": "TITLE_AND_BODY"}}}} }}]`
                        vi. Report the new slide's `objectId` and success/failure of this `createSlide` tool call.

                    **Sub-step B: Announce and Populate That Same Slide's Content**
                        i.  State your intent: "Adding content to slide '[Name of Slide]' (ID: 'the_id_you_just_created')."
                        ii. Using the `objectId` of the slide you *just created* in Sub-step A, determine the placeholder `elementId`s for its title and body (or use generic placeholder locators like `placeholder: {{"type": "TITLE"}}` and `placeholder: {{"type": "BODY", "index": 0}}`).
                        iii.IMMEDIATELY IN THIS SAME RESPONSE, make ONE `batch_update_presentation` call with all necessary `insertText` requests for THIS slide's title and body.
                            * Example: `requests: [{{ "insertText": {{"objectId": "your_new_slide_id_from_A", "placeholder": {{"type": "TITLE"}}, "text": "Actual Slide Title Text"}} }}, {{ "insertText": {{"objectId": "your_new_slide_id_from_A", "placeholder": {{"type": "BODY"}}, "text": "Bullet 1\\nBullet 2"}} }}]`
                        iv. Report success or failure of this `insertText` tool call.
                
                5.  Repeat step 4 (Sub-steps A and B) for every subsequent slide. Complete all actions (structure creation and content population) for one slide before announcing readiness for the next slide's information or moving to the next planned slide.

                **General Principles for Tool Use (REINFORCED):**
                - **INTENT MUST BE FOLLOWED BY IMMEDIATE TOOL CALL IN THE SAME RESPONSE:** If you state you are about to perform an action that requires a tool (like creating a slide or inserting text), that tool call MUST be part of the same message/turn. Do not state an intent and then end your turn without making the call.
                - Report each tool call and its result clearly (e.g., "Successfully created slide 'X' with ID 'Y'", "Successfully added content to slide 'Y'").
                
                Your presentations should be concise, visually appealing, and focused on the key aspects of the project.
                Begin with an introduction slide that clearly states the purpose and goals of the project.
                Include slides that demonstrate the functionality and features of the implemented solution.
                Highlight technical challenges and how they were overcome in the implementation.
                Use bullet points rather than long paragraphs to maintain audience engagement.
                Include code snippets only when they illustrate important concepts or techniques.
                End with a conclusion that summarizes the achievements and potential future enhancements.
                Each slide should have a clear purpose and contribute to the overall narrative.
                Focus on demonstrating the value and capabilities of the solution rather than implementation details.
                
                CRITICAL: Wait for the researcher to complete their work before starting the presentation creation.
                When ALL slides are created AND populated, and you have verified this, state "PRESENTATION COMPLETE".
                """,
                reflect_on_tool_use=True
            )
            
            # Create Voiceover Creator Agent with ElevenLabs tools
            self.voiceover_creator = AssistantAgent(
                name="voiceover_creator",
                model_client=self.model_client,
                workbench=self.elevenlabs_workbench,
                description="Expert at creating voiceovers for presentations",
                system_message=f"""You are an expert at creating voiceovers for presentations.
                Your task is to generate high-quality voiceovers for presentation slides.
                
                You have access to the following tools to create the presentation in Google Slides:
                
                {elevenlabs_tools}
                
                Important ElevenLabs Tool Tips:
                - Use generate_audio_simple for basic text-to-speech
                - Use generate_audio_script for more complex multi-voice scenarios
                - When you have completed the voiceovers, state "VOICEOVER COMPLETE"
                
                CRITICAL: Wait for the presentation creator to complete their work before starting.
                CRITICAL: When you have completed the voiceovers, state "VOICEOVER COMPLETE"
                """,
                reflect_on_tool_use=True
            )
            
            # Create User Proxy Agent
            # self.user_proxy = UserProxyAgent(
            #     name="user_proxy"
            # )
            
            logger.info("Agents initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            await self.cleanup()
            raise
    
    class IgnoreFirstMessageTermination:
        def __init__(self, text_to_find):
            self.text_to_find = text_to_find
            self.message_count = 0
            
        def __call__(self, message):
            self.message_count += 1
            if self.message_count <= 1:
                return False  # Ignore first message
            return self.text_to_find in message.content
    
    async def extract_research_data(self, messages) -> Dict[str, Any]:
        """
        Extract research information from group chat messages
        
        Args:
            messages: List of messages from the group chat
            
        Returns:
            Dict with organized research data for presentation creation
        """
        # Default structure in case we can't extract enough information
        research_data = {
            "title": "GitHub Repository Overview",
            "sections": [
                {
                    "title": "Overview",
                    "content": ["Repository information"],
                    "notes": "This is an overview of the repository"
                },
                {
                    "title": "Features",
                    "content": ["Project features and functionality"],
                    "notes": "The main features of this project."
                },
                {
                    "title": "Technologies",
                    "content": ["Technologies used in this project"],
                    "notes": "List of technologies and frameworks."
                },
                {
                    "title": "Installation",
                    "content": ["Steps to install and set up"],
                    "notes": "Instructions for installation."
                },
                {
                    "title": "Conclusion",
                    "content": ["Summary of the project"],
                    "notes": "Final thoughts and next steps."
                }
            ]
        }
        
        try:
            # First, collect all researcher messages
            research_content = []
            for message in messages:
                try:
                    if not hasattr(message, 'source'):
                        continue
                        
                    if message.source == "researcher":
                        # Extract the actual content string
                        if hasattr(message, 'content'):
                            content = message.content
                            # Handle different content types
                            if isinstance(content, str):
                                research_content.append(content)
                            elif isinstance(content, list):
                                # If content is a list, join its string elements
                                string_items = [str(item) for item in content if item is not None]
                                if string_items:
                                    research_content.append(" ".join(string_items))
                            elif isinstance(content, dict):
                                # If content is a dict, convert to string representation
                                research_content.append(json.dumps(content))
                            else:
                                research_content.append(str(content))
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue
            
            if not research_content:
                logger.warning("No research content found in messages")
                return research_data
                
            combined_research = "\n\n".join(research_content)
            
            # Use the LLM to extract structured data from research
            try:
                # Construct a prompt to extract information
                prompt = f"""Extract structured information from this research about a GitHub repository:

{combined_research}

Format your response as a JSON object with the following structure:
{{
    "title": "Repository Name and Purpose",
    "sections": [
        {{
            "title": "Overview",
            "content": ["Bullet 1", "Bullet 2", ...],
            "notes": "Speaker notes for this section"
        }},
        {{
            "title": "Features",
            "content": ["Feature 1", "Feature 2", ...],
            "notes": "Speaker notes for this section"
        }}
    ]
}}

Include sections for Overview, Features, Technologies Used, Installation, and Conclusion at minimum.
Ensure content items are formatted as bullet points, not paragraphs.
Return ONLY the JSON, no explanation."""

                # Create a special message for this analysis
                messages = [{"role": "user", "content": prompt}]
                response = await self.model_client.create(messages=messages)
                
                try:
                    # Extract JSON from the response
                    content = response.choices[0].message.content
                    
                    # Try to find JSON in the content
                    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
                    if json_match:
                        json_content = json_match.group(1)
                    else:
                        json_content = content
                        
                    extracted_data = json.loads(json_content)
                    
                    # Validate the structure
                    if "title" in extracted_data and "sections" in extracted_data:
                        research_data = extracted_data
                        logger.info("Successfully extracted structured research data")
                    else:
                        logger.warning("Extracted data missing required fields, using default structure")
                except json.JSONDecodeError:
                    logger.warning("Failed to parse LLM response as JSON, using default structure")
                    
            except Exception as e:
                logger.error(f"Error extracting research data from LLM: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error extracting research data: {str(e)}")
            
        return research_data
        
    async def create_presentation_sequential(self, research_data):
        """
        Execute the presentation creation as a sequential workflow.
        This can be called from your group chat when it's time to create the presentation.
        
        Args:
            research_data: Dict with presentation content information
            
        Returns:
            Dict with results of the presentation creation
        """
        logger.info("Starting sequential presentation creation")
        
        # Initialize the workflow
        workflow = PresentationWorkflow(
            slides_workbench=self.slides_workbench,
            logger=logger
        )
        
        # Format the research data into the expected presentation format
        presentation_data = {
            "title": research_data.get("title", "Presentation"),
            "slides": []
        }
        
        # Extract slides from research data
        for section in research_data.get("sections", []):
            slide = {
                "title": section.get("title", "Slide"),
                "content": section.get("content", []),
                "speaker_notes": section.get("notes", "")
            }
            presentation_data["slides"].append(slide)
        
        # Run the workflow
        result = await workflow.run_complete_workflow(presentation_data)
        
        # Log the result
        if result["status"] == "success":
            logger.info(f"Presentation created successfully: {result['url']}")
        else:
            logger.error(f"Presentation creation failed: {result['error']}")
        
        return result
    
    async def create_presentation(self, task_description: str):
        """Run the multi-agent workflow to create a presentation"""
        logger.info(f"\n{'='*60}\nTASK: {task_description}\n{'='*60}")
        
        # Create the group chat
        groupchat = RoundRobinGroupChat(
            participants=[self.researcher, self.presentation_creator],
            max_turns=20,
            termination_condition=TextMentionTermination("PRESENTATION COMPLETE")
        )
        
        # Define initial task message with more explicit instructions
        initial_message = f"""
        Task: {task_description}
        
        Please follow this workflow:
        
        1. ResearcherAgent: First, examine the GitHub repository and gather information about:
        - Project overview and purpose
        - Key features and functionality
        - Technologies used
        - Installation and setup instructions
        
        2. PresentationCreatorAgent: Once research is complete, create a Google Slides presentation:
        - Utilize the research from the research agent
        - Create layouts and use themes that are visually appealing and appropriate for the project
        
        3. VoiceoverCreatorAgent: Once the presentation is complete, create voiceovers:
        - Use generate_audio_simple for basic text-to-speech of slide content
        - Use generate_audio_script for more complex multi-voice scenarios
        
        Important: Each agent should explicitly call their tools and report the results of each tool call.
        If any agent encounters an error, they should:
        1. Report the error clearly
        2. Explain what they were trying to do
        3. Ask for help if needed
        """
        
        try:
            # STEP 1: Perform research with group chat
            logger.info(f"\n{'='*60}\nSTARTING RESEARCH PHASE\n{'='*60}")
            messages = []
            
            try:
                async for message in groupchat.run_stream(task=initial_message):
                    # Add message to our collection
                    if hasattr(message, 'source') and hasattr(message, 'content'):
                        messages.append(message)
                        logger.info(f"Message from {message.source}: {message.content}")
            except Exception as e:
                logger.error(f"Error in research phase: {str(e)}")
                # Continue anyway if we got some research data
            
            # STEP 2: Extract research data for presentation creation
            logger.info(f"\n{'='*60}\nEXTRACTING RESEARCH DATA\n{'='*60}")
            
            # Create default research data as fallback
            research_data = {
                "title": "GitHub Repository Overview",
                "sections": [
                    {
                        "title": "Overview",
                        "content": ["A GitHub repository for code collaboration"],
                        "notes": "This repository contains project code and documentation."
                    },
                    {
                        "title": "Features",
                        "content": ["Project features and functionality"],
                        "notes": "The main features of this project."
                    },
                    {
                        "title": "Technologies",
                        "content": ["Technologies used in this project"],
                        "notes": "List of technologies and frameworks."
                    },
                    {
                        "title": "Installation",
                        "content": ["Steps to install and set up"],
                        "notes": "Instructions for installation."
                    },
                    {
                        "title": "Conclusion",
                        "content": ["Summary of the project"],
                        "notes": "Final thoughts and next steps."
                    }
                ]
            }
            
            try:
                extracted_data = await self.extract_research_data(messages)
                if extracted_data:
                    research_data = extracted_data
            except Exception as e:
                logger.error(f"Error extracting research data: {str(e)}")
                # Continue with default research data
            
            # STEP 3: Create presentation using sequential workflow
            logger.info(f"\n{'='*60}\nSTARTING SEQUENTIAL PRESENTATION CREATION\n{'='*60}")
            presentation_result = await self.create_presentation_sequential(research_data)
            
            # STEP 4: Create voiceovers
            if presentation_result["status"] == "success":
                logger.info(f"\n{'='*60}\nPRESENTATION CREATION COMPLETED\n{'='*60}")
                logger.info(f"Presentation available at: {presentation_result['url']}")
                
                # Now run the voiceover creation separately
                voiceover_chat = RoundRobinGroupChat(
                    participants=[self.voiceover_creator],
                    max_turns=20,
                    termination_condition=TextMentionTermination("VOICEOVER COMPLETE")
                )
                
                logger.info(f"\n{'='*60}\nSTARTING VOICEOVER CREATION\n{'='*60}")
                
                voiceover_message = f"""
                Now that the presentation is complete, please create voiceovers for each slide.
                The presentation is available at: {presentation_result['url']}
                
                Here's the structure of the presentation:
                Title: {research_data['title']}
                
                Slides:
                {json.dumps([section['title'] for section in research_data['sections']], indent=2)}
                
                For each slide, create a voiceover based on the slide content and speaker notes.
                """
                
                voiceover_messages = []
                try:
                    async for message in voiceover_chat.run_stream(task=voiceover_message):
                        if hasattr(message, 'source') and hasattr(message, 'content'):
                            voiceover_messages.append(message)
                            logger.info(f"Message from {message.source}: {message.content}")
                except Exception as e:
                    logger.error(f"Error in voiceover creation: {str(e)}")
                
                logger.info(f"\n{'='*60}\nVOICEOVER CREATION COMPLETED\n{'='*60}")
                
                # Return combined messages and results
                return {
                    "status": "success",
                    "research_messages": messages,
                    "voiceover_messages": voiceover_messages,
                    "presentation_url": presentation_result["url"],
                    "presentation_id": presentation_result["presentation_id"]
                }
            else:
                logger.error(f"Presentation creation failed: {presentation_result['error']}")
                return {
                    "status": "error",
                    "error": presentation_result["error"],
                    "research_messages": messages
                }
                
        except Exception as e:
            logger.error(f"Error in workflow: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.github_workbench:
                await self.github_workbench.stop()
            
            if self.slides_workbench:
                await self.slides_workbench.stop()
                
            if self.elevenlabs_workbench:
                await self.elevenlabs_workbench.stop()
            
            logger.info("All resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise

async def main():
    # Initialize the presentation agent
    agent = PresentationAgent()
    
    try:
        # Initialize workbenches, tools, and agents
        logger.info(f"\n{'='*60}\nINITIALIZING SYSTEM\n{'='*60}")
        await agent.initialize()
        
        # Example task
        task = """
        Create a presentation about the GitHub Repository: https://github.com/robot-coder/assignment-project.
        The presentation should include:
        - Key features of the project
        - Technologies used
        - Installation instructions
        - Contribution guidelines
        """
        
        logger.info(f"\n{'='*60}\nSTARTING PRESENTATION CREATION WORKFLOW\n{'='*60}")
        
        # Run the presentation creation workflow
        result = await agent.create_presentation(task)
        
        logger.info(f"\n{'='*60}\nPRESENTATION CREATION WORKFLOW COMPLETED\n{'='*60}")
        logger.info(f"Result: {result}")
    
    except Exception as e:
        logger.error(f"Error in main workflow: {e}", exc_info=True)
    
    finally:
        # Clean up resources
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())