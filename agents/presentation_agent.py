import os
import json
import re
from typing import Dict, Any, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from llama_index.core.agent import ReActAgent
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
        """Initialize the presentation agent with specified LLM and tools."""
        # Set up token counting
        self.token_counter = TokenCountingHandler(tokenizer=None)
        callback_manager = CallbackManager([self.token_counter])
        
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
            
        self.llm = OpenAI(
            model=model_name,
            api_key=self.api_key,
            callback_manager=callback_manager
        )
        
        self.slides_tool = self._init_slides_tool()
        self.tools = self._create_tools()
        
        self.agent = ReActAgent.from_tools(
            tools=self.tools,
            llm=self.llm,
            verbose=True,
            system_prompt=self._get_system_prompt()
        )
    
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
    
    def _create_tools(self) -> List[FunctionTool]:
        """Create a set of tools for the presentation agent to use."""
        tools = []

        create_outline_tool = FunctionTool.from_defaults(
            name="create_presentation_outline",
            description="Creates a structured outline for a presentation based on pre-analyzed content.",
            fn=self._create_presentation_outline
        )
        tools.append(create_outline_tool)

        if self.slides_tool:
            tools.extend(self.slides_tool.to_tool_list())

        return tools
    
    def _analyze_content(self, code_content: str, documentation_content: str) -> Dict[str, Any]:
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
            response = self.llm.chat(messages)
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
            response = self.llm.chat(messages)
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
        response = self.llm.chat(messages)
        
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
    
    def _get_system_prompt(self) -> str:
        """Define the system prompt that guides the agent's reasoning."""
        return """
        You are an expert presentation creator specialized in generating compelling presentations for coding projects based on pre-analyzed content.

        Your task is to create a professional presentation using the provided analysis results by following these steps SEQUENTIALLY:

        1.  **Create Outline:** Use the `create_presentation_outline` tool.
            *   Input: The analysis results provided in the user query.
            *   Output: A structured outline containing a list of slides, each with `title`, `content` (as a list of strings), and `speaker_notes`.

        2.  **Create Presentation:** Use the `create_presentation` tool.
            *   Input: The `title` from the outline generated in Step 1.
            *   Output: A JSON object containing the `presentationId`. **CRITICAL: You MUST capture this `presentationId` from the tool's response.**

        3.  **Populate Slides:** Use the `batch_update_presentation` tool.
            *   Input:
                *   `presentationId`: The **exact** `presentationId` obtained from the output of Step 2.
                *   `requests`: Construct an array of requests to perform the following actions **IN ORDER**:
                    *   **a) Delete Default Slide:** Add a `deleteObject` request targeting the default first slide. You might need to get the ID of the first slide first using `getPresentation` or assume a common default ID pattern if the tool supports it (check tool description if unsure, otherwise skip deletion if too complex). If skipping deletion, ensure your first `createSlide` uses `insertionIndex: 0`.
                    *   **b) Create and Populate Slides:** For **EACH** slide in the outline from Step 1:
                        *   Create a `createSlide` request. Specify a layout (e.g., `TITLE_AND_BODY`). Use `placeholderIdMappings` to assign unique `objectId`s to the `TITLE` and `BODY` placeholders (e.g., `slide_1_title`, `slide_1_body`). Use `insertionIndex` to control slide order.
                        *   Create an `insertText` request for the slide's `title`, targeting the `objectId` you assigned to the `TITLE` placeholder.
                        *   Create a **SEPARATE** `insertText` request for the slide's `content`. **THIS STEP IS CRITICAL AND MUST NOT BE SKIPPED**. You must:
                            *   Join the list of content strings with newlines (`\n`)
                            *   Target the `objectId` you assigned to the `BODY` placeholder in your placeholderIdMappings
                            *   Make sure this is a separate request from the title insertText request
                        *   (Optional but recommended) Create an `updateShapeProperties` request to set the speaker notes, targeting the slide's notes page element (requires knowing its objectId, often related to the slideId). If unsure how, skip speaker notes for now.
            *   Output: Confirmation that slides were updated.

        **IMPORTANT RULES:**
        *   Execute steps in order.
        *   Use the **exact** `presentationId` from Step 2 in Step 3.
        *   For Step 3b, ensure you generate **BOTH** `insertText` requests (title and body) for **EVERY** slide in the outline. Join body content list with newlines.
        *   The most common mistake is forgetting to create a separate `insertText` request for the slide content or forgetting to target the correct placeholder ID for the body content.
        *   When joining content strings with newlines, make sure they are all plain strings (not objects).
        *   Explain your reasoning, especially how you construct the `requests` array for `batch_update_presentation`.
        """
        
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5)
    )
    def _execute_llm_query(self, query: str):
        """Execute LLM query with retry logic for rate limit errors."""
        # Note: The agent handles its own token counting via the callback manager
        return self.agent.chat(query) # Use chat for conversational interaction

    def create_presentation(self, code_content: str, documentation_content: str) -> Dict[str, Any]:
        """Analyzes content and then queries the agent to create a presentation based on the analysis."""
        self.token_counter.reset_counts()
        analysis_tokens = {}
        agent_tokens = {}

        try:
            # Step 1: Analyze content directly (outside the agent loop)
            analysis_result = self._analyze_content(code_content, documentation_content)
            
            # Validate and ensure required fields exist in the analysis
            required_fields = [
                "project_title", "one_line_summary", "main_functionality_and_features",
                "technical_approach_and_implementation_details", "challenges_and_solutions",
                "key_insights_and_learnings", "potential_extensions_or_improvements"
            ]
            
            for field in required_fields:
                if field not in analysis_result or not analysis_result[field]:
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

            analysis_json_str = json.dumps(analysis_result, indent=2)
            query = f"""
            Please create a Google Slides presentation based on the following analysis results:

            Analysis Results:
            ```json
            {analysis_json_str}
            ```

            Follow these steps using the available tools:
            1. Create a presentation outline using the 'create_presentation_outline' tool with the provided analysis results.
            2. Create a new Google Slides presentation using the 'create_presentation' tool with the title from the outline.
            3. Add all the slides from the outline to the newly created presentation using the 'batch_update_presentation' tool. Make sure to include titles, content, and speaker notes for each slide as specified in the outline.

            Execute these steps sequentially and provide the final presentation ID or link upon completion.
            """

            print("\nStep 2: Querying agent to create presentation from analysis...")
            # Execute the single query, letting the agent handle the steps
            agent_response = self._execute_llm_query(query)
            agent_tokens = {
                "prompt": self.token_counter.prompt_llm_token_count,
                "completion": self.token_counter.completion_llm_token_count,
                "total": self.token_counter.total_llm_token_count
            }
            print(f"Agent finished processing. Tokens used: {agent_tokens['total']}")
            presentation_id = self._extract_presentation_id(agent_response.response)

            # Combine token counts
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
            print(f"Error during presentation creation: {e}")
            # Combine token counts even in case of error
            total_tokens = {
                 "prompt_tokens": analysis_tokens.get("prompt", 0) + agent_tokens.get("prompt", 0),
                 "completion_tokens": analysis_tokens.get("completion", 0) + agent_tokens.get("completion", 0),
                 "total_tokens": analysis_tokens.get("total", 0) + agent_tokens.get("total", 0)
            }
            return {
                "error": str(e),
                "tokens_used": total_tokens
            }

    def _extract_presentation_id(self, response_text: str) -> str:
        """Extract presentation ID from the agent's response, prioritizing JSON parsing of tool output."""
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
    
    agent = PresentationAgent()
    agent.create_presentation(code, documentation)