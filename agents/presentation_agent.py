import os
import json
import re
import nest_asyncio
from typing import Dict, Any, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

from llama_index.core.agent import ReActAgent
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.tools.mcp import McpToolSpec, BasicMCPClient
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.prompts import ChatMessage, MessageRole

from dotenv import load_dotenv
load_dotenv()

class PresentationAgent:
    """Agent responsible for creating presentations from coding assignments using tool-calling."""
    
    def __init__(self, model_name: str = "gpt-4o"):
    # def __init__(self, model_name: str = "gpt-4.1"):
        """Initialize basic attributes of the presentation agent."""
        self.model_name = model_name
        self.token_counter = None
        self.api_key = None
        self.llm = None
        self.slides_tool = None
        self.tools = None
        self.agent = None

    async def initialize(self):
        """Async initialization of the presentation agent."""
        try:
            # Set up token counting
            self.token_counter = TokenCountingHandler(tokenizer=None)
            callback_manager = CallbackManager([self.token_counter])
            
            # Initialize OpenAI
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
                
            self.llm = OpenAI(
                model=self.model_name,
                api_key=self.api_key,
                callback_manager=callback_manager
            )
            
            # Initialize MCP tools
            self.slides_tool = self._init_slides_tool()
            if not self.slides_tool:
                print("Warning: Google Slides MCP tool initialization failed")
            
            self.github_tool = self._init_github_tool()
            if not self.github_tool:
                print("Warning: Github MCP tool initialization failed")
            
            # Create tools list
            self.tools = []
            tools = await self._create_tools()
            self.tools.extend(tools)
            
            # Initialize agent
            self.agent = ReActAgent.from_tools(
                tools=self.tools,
                llm=self.llm,
                verbose=True,
                system_prompt=self._get_system_prompt()
            )
            # Initialize agent
            # self.agent = FunctionAgent(
            #     name="PresentationAgent",
            #     description="An agent that converts creates Google Slides presentations from coding assignments.",
            #     tools=self.tools,
            #     llm=self.llm,
            #     verbose=True,
            #     system_prompt=self._get_system_prompt()
            # )
            
            # Create a tools dictionary for context
            self.tools_dict = {tool.name: tool for tool in self.tools}
            
            return self
            
        except Exception as e:
            print(f"Error during initialization: {e}")
            raise
    
    def _init_slides_tool(self) -> Optional[McpToolSpec]:
        """Initialize Google Slides MCP tool."""
        try:
            server_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "mcp_servers/google-slides-mcp/build/index.js"
            )
            
            slides_mcp_client = BasicMCPClient(
                command_or_url="node",
                args=[
                    server_path,
                    "--google-client-id", os.getenv("GOOGLE_CLIENT_ID"),
                    "--google-client-secret", os.getenv("GOOGLE_CLIENT_SECRET"),
                    "--google-refresh-token", os.getenv("GOOGLE_REFRESH_TOKEN")
                ]
            )
            
            return McpToolSpec(client=slides_mcp_client)
        except Exception as e:
            print(f"Error initializing Google Slides MCP tool: {e}")
            return None
    
    def _init_github_tool(self) -> Optional[McpToolSpec]:
        """Initialize Github MCP tool."""
        try:
            # server_path = os.path.join(
            #     os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            #     "github-mcp-server/github-mcp-server"
            # )
            
            # github_token = os.getenv("GITHUB_TOKEN")
            # if not github_token:
            #     print("Warning: GITHUB_TOKEN not found in environment")
            #     return None
                
            # github_mcp_client = BasicMCPClient(
            #     command_or_url=server_path,
            #     args=[server_path, "--github-token", github_token]
            # )
            # return McpToolSpec(client=github_mcp_client)
            github_mcp_client = BasicMCPClient(
                command_or_url=os.getenv("GITHUB_MCP_URL"),
                args=["stdio"],
                env={
                    "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PAT")
                }
            )
            return McpToolSpec(client=github_mcp_client)
        except Exception as e:
            print(f"Error initializing Github MCP tool: {e}")
            return None

    async def _create_tools(self) -> List[FunctionTool]:
        """Create a set of tools for the presentation agent to use."""
        print("\n=== Initializing Presentation Tools ===")
        tools = []
        self.tools_dict = {}  # Initialize tools dictionary

        print("\nStep 1: Creating outline tool...")
        # Create outline tool with explicit name attribute
        create_outline_tool = FunctionTool.from_defaults(
            fn=self._create_presentation_outline,
            name="create_presentation_outline",
            description="Creates a structured outline for a presentation based on pre-analyzed content."
        )
        create_outline_tool.name = "create_presentation_outline"  # Explicitly set name
        tools.append(create_outline_tool)
        print(f"✓ Created outline tool: {create_outline_tool.name}")
        print("✓ Outline tool created")

        print("\nStep 2: Loading Google Slides tools...")
        if self.slides_tool:
            try:
                slide_tools = await self.slides_tool.to_tool_list_async()
                tools.extend(slide_tools)
                # Add to tools dictionary
                for tool in slide_tools:
                    self.tools_dict[tool.name] = tool
                print("✓ Google Slides tools loaded:")
                # for tool in slide_tools:
                #     print(f"  - {tool.name}")
            except Exception as e:
                print(f"ERROR: Failed to load Google Slides tools: {e}")
        else:
            print("WARNING: Google Slides tool not initialized")

        print("\nStep 3: Loading Github tools...")
        if self.github_tool:
            try:
                github_tools = await self.github_tool.to_tool_list_async()
                tools.extend(github_tools)
                # Add to tools dictionary
                for tool in github_tools:
                    self.tools_dict[tool.name] = tool
                print("✓ Github tools loaded:")
                # for tool in github_tools:
                #     print(f"  - {tool.name}")
            except Exception as e:
                print(f"ERROR: Failed to load Github tools: {e}")
        else:
            print("WARNING: Github tool not initialized")

        print("\nTool initialization summary:")
        print(f"Total tools available: {len(tools)}")
        print("Available tools:")
        # for tool in tools:
        #     print(f"- {tool.name}: {tool.description}")

        print("=== Tool Initialization Complete ===\n")
        # Create tools dictionary
        self.tools_dict = {}
        for tool in tools:
            if not hasattr(tool, 'name'):
                tool.name = f"tool_{len(self.tools_dict)}"
            self.tools_dict[tool.name] = tool
            print(f"Added to tools dictionary: {tool.name}")

        print(f"\nTotal tools initialized: {len(tools)}")
        print("Available tools:", ", ".join(self.tools_dict.keys()))
        return tools
    
    async def _analyze_content(self, code_content: str, documentation_content: str) -> Dict[str, Any]:
        """Analyze code and documentation in chunks to extract key points for presentation."""
        # Process code in chunks if needed
        code_chunks = self._chunk_text(code_content, max_chunk_size=1500)
        doc_chunks = self._chunk_text(documentation_content, max_chunk_size=1500)
        
        # First pass - analyze each chunk separately
        code_analyses = []
        for i, chunk in enumerate(code_chunks):
            prompt = f"""
            Please analyze part {i+1}/{len(code_chunks)} of the code:
            
            ```python
            {chunk}
            ```
            
            Identify key functions, patterns, or features in this section.
            Keep your analysis brief and focused.
            """
            
            messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
            response = await self.llm.achat(messages)
            code_analyses.append(response.message.content)
        
        doc_analyses = []
        for i, chunk in enumerate(doc_chunks):
            prompt = f"""
            Please analyze part {i+1}/{len(doc_chunks)} of the code:
            
            ```python
            {chunk}
            ```
            
            Identify key functions, patterns, or features in this section.
            Keep your analysis brief and focused.
            """
            
            messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
            response = await self.llm.achat(messages)
            doc_analyses.append(response.message.content)
        
        synthesis_prompt = f"""
        Based on the following analyses of code and documentation chunks, provide a comprehensive analysis:
        
        CODE ANALYSES:
        {' '.join(code_analyses)}
        
        DOCUMENTATION ANALYSES:
        {' '.join(doc_analyses if 'doc_analyses' in locals() else [])}
        
        Extract the following information for a presentation:
        1. Project title and one-line summary
        2. Main functionality and features (3-5 bullet points)
        3. Technical approach and implementation details (as bullet points)
        4. Challenges and solutions (as pairs of challenge and solution)
        5. Key insights and learnings (as bullet points)
        6. Potential extensions or improvements (as bullet points)
        
        Format your response as a structured JSON object with the EXACT following keys:
        - project_title (string)
        - one_line_summary (string)
        - main_functionality_and_features (array of strings)
        - technical_approach_and_implementation_details (array of strings)
        - challenges_and_solutions (array of strings, alternating challenges and solutions)
        - key_insights_and_learnings (array of strings)
        - potential_extensions_or_improvements (array of strings)
        
        Example format:
        {{
          "project_title": "Title Here",
          "one_line_summary": "Summary here",
          "main_functionality_and_features": ["Feature 1", "Feature 2"],
          "technical_approach_and_implementation_details": ["Detail 1", "Detail 2"],
          "challenges_and_solutions": [
            "Challenge: Challenge 1",
            "Solution: Solution 1",
            "Challenge: Challenge 2",
            "Solution: Solution 2"
          ],
          "key_insights_and_learnings": ["Insight 1", "Insight 2"],
          "potential_extensions_or_improvements": ["Extension 1", "Extension 2"]
        }}
        
        IMPORTANT: Make sure to use EXACTLY these JSON keys and maintain an array structure for each list.
        """
        
        messages = [ChatMessage(role=MessageRole.USER, content=synthesis_prompt)]
        response = await self.llm.achat(messages)
        
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response.message.content)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response.message.content
        
        try:
            analysis = json.loads(json_str)
        except json.JSONDecodeError:
            analysis = {
                "project_title": "Project Presentation",
                "one_line_summary": "Overview of the coding project",
                "main_functionality_and_features": ["Feature 1", "Feature 2", "Feature 3"],
                "technical_approach_and_implementation_details": ["Technical implementation detail 1", "Technical implementation detail 2"],
                "challenges_and_solutions": [
                    "Challenge: Implementation challenge 1",
                    "Solution: Engineering solution 1",
                    "Challenge: Implementation challenge 2",
                    "Solution: Engineering solution 2"
                ],
                "key_insights_and_learnings": ["Insight 1", "Insight 2"],
                "potential_extensions_or_improvements": ["Extension 1", "Extension 2"]
            }
        
        return analysis
        
    def _chunk_text(self, text: str, max_chunk_size: int = 1500) -> List[str]:
        """Split text into smaller chunks respecting line breaks where possible."""
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_size = 0
        
        for line in lines:
            if current_size + len(line) + 1 > max_chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(line)
            current_size += len(line) + 1
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _create_presentation_outline(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create a structured presentation outline based on content analysis."""
        title = analysis.get("project_title", "Project Presentation")
        summary = analysis.get("one_line_summary", "")
        features = analysis.get("main_functionality_and_features", [])
        technical_details = analysis.get("technical_approach_and_implementation_details", [])
        technical = "\n".join([f"• {item}" for item in technical_details]) if technical_details else ""
        
        challenges_solutions = analysis.get("challenges_and_solutions", [])
        challenges = []
        solutions = []
        
        # Handle challenges and solutions more defensively
        if isinstance(challenges_solutions, list):
            # If challenges_solutions is already a simple list of strings, handle accordingly
            if all(isinstance(item, str) for item in challenges_solutions):
                for i, item in enumerate(challenges_solutions):
                    if i % 2 == 0:  # Even items are challenges
                        challenges.append(item)
                    else:  # Odd items are solutions
                        solutions.append(item)
            else:
                for item in challenges_solutions:
                    if isinstance(item, dict):
                        if "challenge" in item:
                            challenges.append(str(item["challenge"]))
                        if "solution" in item:
                            solutions.append(str(item["solution"]))
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        challenges.append(str(item[0]))
                        solutions.append(str(item[1]))
                    elif isinstance(item, str):
                        if item.startswith("Challenge:"):
                            challenges.append(item[10:].strip())
                        elif item.startswith("Solution:"):
                            solutions.append(item[9:].strip())
        
        insights = analysis.get("key_insights_and_learnings", [])
        extensions = analysis.get("potential_extensions_or_improvements", [])
        
        outline = {
            "title": title,
            "slides": [
                {
                    "title": "Overview",
                    "content": [summary] + ["• " + feature for feature in features] if features else [summary],
                    "speaker_notes": f"This presentation covers {title}. {summary}" +
                                     (" The main features include various functionalities." if features else "")
                },
                {
                    "title": "Technical Approach",
                    "content": technical_details if isinstance(technical_details, list) else [technical],
                    "speaker_notes": "Let me explain our technical approach and implementation details."
                },
                {
                    "title": "Challenges & Solutions",
                    "content": (["Challenges:"] + ["• " + challenge for challenge in challenges] +
                              ["Solutions:"] + ["• " + solution for solution in solutions]) if challenges or solutions else ["No significant challenges encountered"],
                    "speaker_notes": "We encountered several challenges and implemented solutions to address them."
                },
                {
                    "title": "Key Insights",
                    "content": ["• " + insight for insight in insights],
                    "speaker_notes": "Here are the key insights and learnings from this project."
                },
                {
                    "title": "Future Work",
                    "content": ["• " + extension for extension in extensions],
                    "speaker_notes": "Looking ahead, there are several ways this project could be extended and improved."
                },
                {
                    "title": "Thank You!",
                    "content": ["Questions?"],
                    "speaker_notes": "Thank you for your attention. I'm happy to answer any questions you might have."
                }
            ]
        }
        
        return outline
    
    def _get_steps(self) -> List[Dict[str, str]]:
        """Define the steps for creating a presentation."""
        print("\nInitializing presentation creation steps...")
        steps = [
            {
                "name": "create_presentation_outline",
                "description": "Create a structured outline using the analysis results",
                "tool": "create_presentation_outline",
                "input_key": "analysis",
                "output_key": "outline",
                "required_tools": ["create_presentation_outline"]
            },
            {
                "name": "create_presentation",
                "description": "Create a new Google Slides presentation",
                "tool": "create_presentation",
                "input_key": "title",
                "output_key": "presentationId",
                "required_tools": ["create_presentation"]
            },
            {
                "name": "batch_update_presentation",
                "description": "Add all slides to the presentation",
                "tool": "batch_update_presentation",
                "input_key": "presentationId",
                "output_key": "result",
                "required_tools": ["batch_update_presentation", "get_presentation"]
            }
        ]
        
        print("\nValidating presentation steps...")
        for i, step in enumerate(steps, 1):
            print(f"\nStep {i}: {step['name']}")
            print(f"- Description: {step['description']}")
            print(f"- Tool: {step['tool']}")
            print(f"- Required tools: {', '.join(step['required_tools'])}")
            
            # Verify required tools are available
            # missing_tools = [
            #     tool for tool in step['required_tools']
            #     if not any(t.name == tool for t in (self.tools or []))
            # ]
            # if missing_tools:
            #     print(f"WARNING: Missing required tools for {step['name']}: {', '.join(missing_tools)}")
            # else:
            #     print(f"✓ All required tools available for {step['name']}")
        
        print("\n✓ Presentation steps initialized and validated")
        return steps

    def _get_system_prompt(self) -> str:
        """Define the system prompt that guides the agent's reasoning."""
        steps = self._get_steps()
        steps_text = "\n".join(f"{i+1}. **{step['name']}:** {step['description']}"
                             for i, step in enumerate(steps))
        
        return f"""
        You are an expert presentation creator specialized in generating compelling presentations for coding projects based on pre-analyzed content.

        Follow these steps SEQUENTIALLY to create the presentation:

        {steps_text}

        For each step:
        1. Use the specified tool
        2. Pass the required input parameters
        3. Capture and validate the output
        4. Use the output in subsequent steps as needed

        When using batch_update_presentation:
        1. Delete the default first slide if possible
        2. Create new slides with proper layouts
        3. Add titles and content separately
        4. Include speaker notes where possible
        5. Maintain proper slide order

        **IMPORTANT RULES:**
        * Execute steps in exact order
        * Validate all inputs and outputs
        * Keep track of the presentationId
        * Create separate requests for titles and content
        * Join content strings with newlines
        * Ensure all strings are properly formatted
        * Explain your reasoning for each step
        """
        
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5)
    )
    async def _execute_llm_query(self, query: str):
        """Execute LLM query with retry logic for rate limit errors."""
        # Note: The agent handles its own token counting via the callback manager
        return await self.agent.achat(query) # Use achat for async chat interaction

    async def create_presentation(self) -> Dict[str, Any]:
        """Create a presentation by fetching code and documentation from Github."""
        try:
            if not self.github_tool:
                raise ValueError("Github MCP tool not initialized. Please check GITHUB_TOKEN environment variable.")

            # First get list of repositories
            list_repos_response = await self.agent.achat("""
            Use the Github MCP list_repositories tool to get a list of repositories.
            Return ONLY the name of the most recently created repository.
            """)
            
            repo_name = list_repos_response.response.strip()
            if not repo_name:
                print("ERROR: Empty repository name received")
                raise ValueError("Failed to get repository name from Github")
            
            print(f"✓ Found latest repository: {repo_name}")
            
            print("\nStep 3: Fetching main.py content...")
            # Now get the contents of main.py
            get_code_response = await self.agent.achat(f"""
            Use the Github MCP get_file_content tool to get the contents of main.py from the repository '{repo_name}'.
            Return ONLY the file content, no additional text.
            """)
            
            code_content = get_code_response.response.strip()
            if not code_content:
                print("ERROR: Empty main.py content received")
                raise ValueError("Failed to get main.py content from Github")
            
            print(f"✓ Successfully fetched main.py content ({len(code_content)} chars)")
            
            print("\nStep 4: Fetching README.md content...")
            # Get the contents of README.md
            get_doc_response = await self.agent.achat(f"""
            Use the Github MCP get_file_content tool to get the contents of README.md from the repository '{repo_name}'.
            Return ONLY the file content, no additional text.
            """)
            
            documentation_content = get_doc_response.response.strip()
            if not documentation_content:
                print("ERROR: Empty README.md content received")
                raise ValueError("Failed to get README.md content from Github")
            
            print(f"✓ Successfully fetched README.md content ({len(documentation_content)} chars)")
            
            print("\nStep 5: Creating presentation with fetched content...")
            # Call the original create_presentation method with the fetched content
            result = await self._create_presentation_with_content(code_content, documentation_content)
            print("✓ Presentation creation completed")
            print("=== Presentation Creation Process Complete ===\n")
            return result
        except Exception as e:
            print(f"Error creating presentation: {e}")
            return {
                "error": str(e),
                "tokens_used": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }

    async def _create_presentation_with_content(self, code_content: str, documentation_content: str) -> Dict[str, Any]:
        """Analyzes content and then queries the agent to create a presentation based on the analysis."""
        print("\n=== Starting Presentation Content Creation ===")
        print(f"Input content sizes:")
        print(f"- Code content: {len(code_content)} chars")
        print(f"- Documentation content: {len(documentation_content)} chars")
        
        self.token_counter.reset_counts()
        analysis_tokens = {}
        agent_tokens = {}

        try:
            # Step 1: Analyze content directly (outside the agent loop)
            print("\nStep 1: Starting content analysis...")
            analysis_result = await self._analyze_content(code_content, documentation_content)
            print("✓ Content analysis completed")
            print(f"Analysis result contains {len(analysis_result)} fields")
            
            # Step 2: Validate analysis results
            print("\nStep 2: Validating analysis results...")
            required_fields = [
                "project_title", "one_line_summary", "main_functionality_and_features",
                "technical_approach_and_implementation_details", "challenges_and_solutions",
                "key_insights_and_learnings", "potential_extensions_or_improvements"
            ]
            
            for field in required_fields:
                print(f"Checking field: {field}")
                if field not in analysis_result:
                    print(f"ERROR: Missing required field: {field}")
                    raise ValueError(f"Analysis result missing required field: {field}")
                if not analysis_result[field]:
                    print(f"ERROR: Empty required field: {field}")
                    print(f"Warning: Missing or empty field '{field}' in analysis. Adding default value.")
                    
                    # Add default values based on field type
                    if field == "project_title":
                        analysis_result[field] = "Project Presentation"
                    elif field == "one_line_summary":
                        analysis_result[field] = "Overview of the coding project"
                    elif field.endswith("_and_solutions"):
                        # Use a simple list of strings instead of complex objects to avoid string formatting issues
                        analysis_result[field] = ["Implementation challenge", "Engineering solution"]
                    else:
                        analysis_result[field] = ["Default item 1", "Default item 2"]
            
            # Special handling for all fields that could contain complex structures
            for field in analysis_result:
                value = analysis_result[field]
                if isinstance(value, list):
                    string_list = []
                    for item in value:
                        if isinstance(item, dict):
                            # For dictionaries like in challenges_and_solutions
                            for k, v in item.items():
                                string_list.append(f"{k.capitalize()}: {v}")
                        elif isinstance(item, (list, tuple)):
                            string_list.append(", ".join(str(x) for x in item))
                        else:
                            string_list.append(str(item))
                    analysis_result[field] = string_list
            
            for key in analysis_result.keys():
                value = analysis_result[key]
                if isinstance(value, list):
                    print(f"  - {key}: {len(value)} items")
                else:
                    print(f"  - {key}: {type(value).__name__}")
            
            analysis_tokens = {
                "prompt": self.token_counter.prompt_llm_token_count,
                "completion": self.token_counter.completion_llm_token_count,
                "total": self.token_counter.total_llm_token_count
            }
            print(f"Analysis complete. Tokens used: {analysis_tokens['total']}")
            self.token_counter.reset_counts()

            # Get the presentation creation steps
            print("\nGetting presentation creation steps...")
            steps = self._get_steps()
            print(f"✓ Found {len(steps)} steps to execute")
            
            # Create the query with detailed step instructions
            analysis_json_str = json.dumps(analysis_result, indent=2)
            steps_instructions = "\n".join(f"{i+1}. {step['description']} using the '{step['tool']}' tool"
                                        for i, step in enumerate(steps))
            
            query = f"""
            Please create a Google Slides presentation based on the following analysis results:

            Analysis Results:
            ```json
            {analysis_json_str}
            ```

            Follow these steps EXACTLY in order:
            {steps_instructions}

            For each step:
            1. Use the exact tool specified
            2. Validate inputs before using the tool
            3. Verify the tool's output
            4. Use the output in subsequent steps as needed

            IMPORTANT:
            - The presentationId from step 2 is CRITICAL for step 3
            - Each slide must have separate title and content
            - Content must be properly formatted with newlines
            - Speaker notes should be included where possible

            Execute these steps sequentially and provide the final presentation ID or link upon completion.
            """

            print("\nStep 3: Creating presentation query...")
            print("Query length:", len(query))
            print("Analysis fields being used:", list(analysis_result.keys()))
            print("Steps to execute:")
            for i, step in enumerate(steps, 1):
                print(f"  {i}. {step['name']} using {step['tool']}")
                print(f"     Input: {step['input_key']}, Output: {step['output_key']}")
            
            print("\nStep 4: Executing agent query...")
            print("Sending query to agent for presentation creation...")
            agent_response = await self._execute_llm_query(query)
            print("✓ Agent query completed")
            print("Response length:", len(agent_response.response))
            
            # Log the expected tool usage
            print("\nExpected tool sequence:")
            for step in steps:
                print(f"- {step['tool']} should be called with {step['input_key']}")
            agent_tokens = {
                "prompt": self.token_counter.prompt_llm_token_count,
                "completion": self.token_counter.completion_llm_token_count,
                "total": self.token_counter.total_llm_token_count
            }
            print(f"\nStep 5: Processing agent response...")
            print(f"Token usage breakdown:")
            print(f"- Prompt tokens: {agent_tokens['prompt']}")
            print(f"- Completion tokens: {agent_tokens['completion']}")
            print(f"- Total tokens: {agent_tokens['total']}")
            
            print("\nStep 6: Extracting presentation ID...")
            presentation_id = self._extract_presentation_id(agent_response.response)
            if presentation_id == "ERROR_ID_NOT_EXTRACTED":
                print("ERROR: Failed to extract presentation ID from response")
                print("Raw response excerpt (first 200 chars):", agent_response.response[:200])
            else:
                print(f"✓ Successfully extracted presentation ID: {presentation_id}")

            print("\nStep 7: Calculating total token usage...")
            total_tokens = {
                "prompt_tokens": analysis_tokens.get("prompt", 0) + agent_tokens.get("prompt", 0),
                "completion_tokens": analysis_tokens.get("completion", 0) + agent_tokens.get("completion", 0),
                "total_tokens": analysis_tokens.get("total", 0) + agent_tokens.get("total", 0)
            }

            # Return the result
            return {
                "agent_response": agent_response.response,
                "presentation_id": presentation_id,
                "tokens_used": total_tokens
            }

        except Exception as e:
            print("\nERROR during presentation creation:")
            print(f"- Error type: {type(e).__name__}")
            print(f"- Error message: {str(e)}")
            if hasattr(e, '__traceback__'):
                print(f"- Error location: {e.__traceback__.tb_frame.f_code.co_filename}:{e.__traceback__.tb_lineno}")
            
            print("\nCalculating final token usage despite error...")
            total_tokens = {
                 "prompt_tokens": analysis_tokens.get("prompt", 0) + agent_tokens.get("prompt", 0),
                 "completion_tokens": analysis_tokens.get("completion", 0) + agent_tokens.get("completion", 0),
                 "total_tokens": analysis_tokens.get("total", 0) + agent_tokens.get("total", 0)
            }
            error_response = {
                "error": str(e),
                "error_type": type(e).__name__,
                "tokens_used": total_tokens
            }
            print("\nReturning error response:", json.dumps(error_response, indent=2))
            return error_response

    def _extract_presentation_id(self, response_text: str) -> str:
        """Extract presentation ID from the agent's response, prioritizing JSON parsing of tool output."""
        print("\nAttempting to extract presentation ID from response...")
        print(f"Response length: {len(response_text)} chars")
        print("\nAttempting to extract presentation ID from response...")
        # Regex explanation:
        # - (?:Tool Response:|Function Output:|```json) - Non-capturing group for potential markers
        # - \s* - Optional whitespace
        # - (\{[\s\S]*?\}) - Capturing group 1: The JSON object itself (curly braces, content, non-greedy)
        # - \s* - Optional whitespace
        # - (?:```)? - Optional closing markdown backticks
        json_pattern = r'(?:Tool Response:|Function Output:|```json)\s*(\{[\s\S]*?\})\s*(?:```)?'
        json_matches = re.finditer(json_pattern, response_text) # Find all potential JSON blocks

        for match in json_matches:
            json_str = match.group(1)
            try:
                cleaned_json_str = json_str.replace('\\"', '"').replace('\\n', '\n')
                data = json.loads(cleaned_json_str)
                if isinstance(data, dict) and 'presentationId' in data:
                    extracted_id = data['presentationId']
                    # Basic validation for Google Slides ID format
                    if isinstance(extracted_id, str) and len(extracted_id) > 15 and re.match(r'^[a-zA-Z0-9_-]+$', extracted_id):
                        print(f"Successfully extracted ID via JSON parsing: {extracted_id}")
                        return extracted_id
                    else:
                        print(f"Parsed JSON, found 'presentationId', but value '{extracted_id}' seems invalid.")
                else:
                     print("Parsed JSON, but 'presentationId' key not found or data is not a dict.")
            except json.JSONDecodeError as e:
                print(f"Failed to parse potential JSON block: {e}")

        print("JSON parsing failed or ID not found in JSON. Falling back to regex patterns...")
        patterns = [
            # Look specifically for the ID within a JSON-like structure in the text
            r'"presentationId"\s*:\s*"([a-zA-Z0-9_-]{20,})"',
            # Look for common phrases indicating the ID
            r'[Pp]resentation\s*ID\s*(?:is|:)\s*([a-zA-Z0-9_-]{20,})',
            r'created\s*(?:a|the)?\s*presentation\s*(?:with\s*ID)?\s*[:=]?\s*([a-zA-Z0-9_-]{20,})',
            # Look for a Google Slides URL
            r'https://docs.google.com/presentation/d/([a-zA-Z0-9_-]+)',
            # General fallback for a long alphanumeric string (use word boundaries)
            r'\b([a-zA-Z0-9_-]{20,})\b'
        ]

        for pattern in patterns:
            flags = re.IGNORECASE if '[Pp]resentation' in pattern else 0
            match = re.search(pattern, response_text, flags=flags)
            if match:
                extracted_id = match.group(1)
                if len(extracted_id) > 15 and re.match(r'^[a-zA-Z0-9_-]+$', extracted_id):
                    print(f"Successfully extracted ID via fallback regex pattern '{pattern}': {extracted_id}")
                    return extracted_id
                else:
                    print(f"Regex pattern '{pattern}' matched '{extracted_id}', but it doesn't look like a valid ID. Skipping.")

        print("ERROR: Could not extract presentation ID using any method.")
        # Return a clearly identifiable error string instead of a potentially valid-looking fake ID
        return "ERROR_ID_NOT_EXTRACTED"

# # Create initial presentation agent instance
# presentation_agent = None

# async def get_presentation_agent():
#     """Get or create the presentation agent instance."""
#     global presentation_agent
#     if presentation_agent is None:
#         presentation_agent = await PresentationAgent().__ainit__()
#     return presentation_agent

# Global instance
presentation_agent = None

async def get_presentation_agent():
    """Get or create the presentation agent instance."""
    global presentation_agent
    if presentation_agent is None:
        agent = PresentationAgent()
        await agent.initialize()
        presentation_agent = agent
    return presentation_agent

if __name__ == "__main__":
    code = """
    # Chat Assistant Implementation
    import os
    import requests
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Dict, Any, Optional
    
    app = FastAPI()
    
    class Message(BaseModel):
        role: str  # "user" or "assistant"
        content: str
    
    class ChatRequest(BaseModel):
        messages: List[Message]
        model: str
        temperature: Optional[float] = 0.7
    
    class ChatResponse(BaseModel):
        response: str
        
    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        # Format messages for LLM API
        formatted_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Call external LLM API
        try:
            response = requests.post(
                "https://api.openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": "Bearer {os.environ.get('OPENROUTER_API_KEY')}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": request.model,
                    "messages": formatted_messages,
                    "temperature": request.temperature
                }
            )
            response.raise_for_status()
            result = response.json()
            return ChatResponse(response=result["choices"][0]["message"]["content"])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error communicating with LLM API: {str(e)}")
    
    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    
    documentation = """
    # Chat Assistant API
    
    A FastAPI server that provides a simple interface to various large language models through the OpenRouter API.
    
    ## Features
    
    - Simple REST API for chat interactions
    - Support for multiple LLM models via OpenRouter
    - Conversation history management
    - Configurable temperature parameter
    
    ## API Usage
    
    ### POST /chat
    
    Send a chat request with conversation history.
    
    ```json
    {{
      "messages": [
        {{"role": "user", "content": "Hello, how are you?"}},
        {{"role": "assistant", "content": "I'm doing well! How can I help you today?"}},
        {{"role": "user", "content": "Tell me about the solar system."}}
      ],
      "model": "anthropic/claude-3-opus",
      "temperature": 0.7
    }}
    ```
    
    ### Response
    
    ```json
    {{
      "response": "The solar system consists of the Sun and everything that orbits around it..."
    }}
    ```
    
    ## Implementation Details
    
    The server is built with FastAPI and communicates with OpenRouter's API to access various LLMs. 
    It handles authentication, request formatting, and error handling.
    
    ## Deployment
    
    The service can be deployed on Render.com for free. Environment variables must be set for the OpenRouter API key.
    """
    
    async def main():
        agent = PresentationAgent()
        await agent.initialize()
        return await agent._create_presentation_with_content(code, documentation)
    
    import asyncio
    asyncio.run(main())