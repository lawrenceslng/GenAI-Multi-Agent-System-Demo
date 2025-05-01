"""
Presentation Agent for Multi-Agent Homework System
Responsible for creating presentations based on code and documentation outputs
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
import pprint
import traceback

from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.tools import FunctionTool, ToolOutput
from llama_index.core.agent import ReActAgent
from llama_index.tools.mcp import McpToolSpec, BasicMCPClient

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

llm = OpenAI(model="gpt-4.1-nano-2025-04-14", api_key=os.getenv("OPENAI_API_KEY"))

class PresentationAgent:
    """Agent responsible for creating presentations from coding assignments."""
    
    def __init__(self, model_name: str = "gpt-4o"):
        """Initialize the presentation agent with specified LLM."""
        # Initialize the LLM
        self.api_key = os.getenv("OPENAI_API_KEY") 
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
            
        # Set up token counting
        self.token_counter = TokenCountingHandler(
            tokenizer=None  # will use default tiktoken
        )
        callback_manager = CallbackManager([self.token_counter])
        
        # Initialize LLM
        self.llm = OpenAI(
            model=model_name,
            api_key=self.api_key,
            temperature=0.3,
            callback_manager=callback_manager
        )
        
        # Initialize the Google Slides MCP tool
        self.slides_tool = self._init_slides_tool()
        
        # Initialize agent tools
        self.tools = self._create_tools()
        
        # Create the agent
        self.agent = ReActAgent.from_tools(
            tools=self.tools,
            llm=self.llm,
            verbose=True,
            system_prompt=self._get_system_prompt(),
        )

    def _init_slides_tool(self) -> Optional[McpToolSpec]:
        """Initialize Google Slides MCP tool."""
        try:
            # Start the MCP server as a subprocess
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
                ],
            )

            return McpToolSpec(client=slides_mcp_client)
        except Exception as e:
            print(f"Error initializing Google Slides MCP tool: {e}")
            return None

    def _create_tools(self) -> List[FunctionTool]:
        """Create tools for the presentation agent."""
        tools = []
        
        # Add custom analysis tool
        analyze_content_tool = FunctionTool.from_defaults(
            name="analyze_content",
            description="Analyzes code and documentation to extract key points for presentation",
            fn=self._analyze_content
        )
        tools.append(analyze_content_tool)
        
        # Add outline generation tool
        create_outline_tool = FunctionTool.from_defaults(
            name="create_presentation_outline",
            description="Creates a structured outline for a presentation based on content analysis",
            fn=self._create_presentation_outline
        )
        tools.append(create_outline_tool)
        
        # Add Google Slides tools if available
        if self.slides_tool:
            # Get the tools from the spec
            tools_list = self.slides_tool.to_tool_list()
            tools.extend(tools_list)
            
        return tools

    def _analyze_content(self, code_content: str, documentation_content: str) -> Dict[str, Any]:
        """Analyze code and documentation to extract key points for presentation."""
        prompt = f"""
        Please analyze the following code and documentation for a presentation:
        
        CODE:
        ```python
        {code_content[:3000]}  # Limit to prevent token overflow
        ```
        
        DOCUMENTATION:
        {documentation_content[:3000]}  # Limit to prevent token overflow
        
        Extract the following information for a presentation:
        1. Project title and one-line summary
        2. Main functionality and features (3-5 bullet points)
        3. Technical approach and implementation details
        4. Challenges and solutions
        5. Key insights and learnings
        6. Potential extensions or improvements
        
        Format your response as a structured JSON object.
        """
        
        messages = [
            ChatMessage(role=MessageRole.USER, content=prompt)
        ]
        
        response = self.llm.chat(messages)
        
        # Extract JSON from response
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response.message.content)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response.message.content
        
        try:
            analysis = json.loads(json_str)
        except json.JSONDecodeError:
            # Create structured output if JSON parsing fails
            analysis = {
                "title": "Project Presentation",
                "summary": "Overview of the coding project",
                "main_features": ["Feature 1", "Feature 2", "Feature 3"],
                "technical_approach": "Technical implementation details",
                "challenges": ["Challenge 1", "Challenge 2"],
                "solutions": ["Solution 1", "Solution 2"],
                "key_insights": ["Insight 1", "Insight 2"],
                "potential_extensions": ["Extension 1", "Extension 2"]
            }
        
        return analysis
    
    def _create_presentation_outline(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create a structured presentation outline based on content analysis."""
        # Extract data from analysis
        title = analysis.get("title", "Project Presentation")
        summary = analysis.get("summary", "")
        features = analysis.get("main_features", [])
        technical = analysis.get("technical_approach", "")
        challenges = analysis.get("challenges", [])
        solutions = analysis.get("solutions", [])
        insights = analysis.get("key_insights", [])
        extensions = analysis.get("potential_extensions", [])
        
        # Create slide outline
        outline = {
            "title": title,
            "slides": [
                {
                    "title": "Overview",
                    "content": [summary] + ["• " + feature for feature in features],
                    "speaker_notes": f"This presentation covers {title}. {summary} The main features include: " + 
                                     ", ".join(features) + "."
                },
                {
                    "title": "Technical Approach",
                    "content": [technical],
                    "speaker_notes": f"Let me explain our technical approach. {technical}"
                },
                {
                    "title": "Challenges & Solutions",
                    "content": ["Challenges:"] + ["• " + challenge for challenge in challenges] +
                              ["Solutions:"] + ["• " + solution for solution in solutions],
                    "speaker_notes": f"We encountered several challenges including: {', '.join(challenges)}. " +
                                    f"Our solutions involved: {', '.join(solutions)}."
                },
                {
                    "title": "Key Insights",
                    "content": ["• " + insight for insight in insights],
                    "speaker_notes": f"The key insights from this project were: {', '.join(insights)}."
                },
                {
                    "title": "Future Work",
                    "content": ["• " + extension for extension in extensions],
                    "speaker_notes": f"Looking ahead, we could extend this project by: {', '.join(extensions)}."
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
        """Get the system prompt for the presentation agent."""
        return """
        You are an expert presentation creator tasked with generating compelling presentations for coding projects.
        
        Your goal is to:
        1. Analyze code and documentation to extract key project information
        2. Create a well-structured presentation outline
        3. Generate professional slides using Google Slides (if available)
        
        Your presentations should:
        - Be clear, concise, and engaging
        - Highlight the most important aspects of the project
        - Explain technical concepts in an accessible way
        - Provide insights into challenges and solutions
        - Include speaker notes for each slide
        
        Follow these steps:
        1. Analyze the code and documentation using the analyze_content tool
        2. Create a presentation outline using the create_presentation_outline tool
        3. Generate slides using the tools from Google Slides MCP:
           - create_presentation: Create a new presentation
           - add_slide: Add slides to the presentation
           - batch_update_presentation: For more complex slide modifications
        
        Be creative but professional in your presentation style. Use clear titles, concise bullet points,
        and appropriate technical depth for the target audience.
        """
    
    def create_google_slides(self, outline: Dict[str, Any]) -> Dict[str, Any]:
        """Create a Google Slides presentation based on the outline."""
        presentation_id = None
        results = {} # Initialize results dictionary
        create_presentation_tool = None
        batch_update_tool = None

        # --- 1. Check if Slides Tool Spec is available ---
        if not self.slides_tool:
            print("--- ERROR: Google Slides MCP tool spec not initialized. ---")
            return {"error": "Google Slides MCP tools not available"}

        # --- 2. Retrieve Specific MCP Tools ---
        try:
            print("--- DEBUG: Retrieving Google Slides MCP tools ---")
            tools = self.slides_tool.to_tool_list()
            # Use next with a default of None and check after
            create_presentation_tool = next((t for t in tools if t.metadata.name == "create_presentation"), None)
            batch_update_tool = next((t for t in tools if t.metadata.name == "batch_update_presentation"), None)

            if not create_presentation_tool:
                print("--- ERROR: 'create_presentation' tool not found in MCP spec. ---")
                return {"error": "Required 'create_presentation' Google Slides MCP tool not found"}
            if not batch_update_tool:
                print("--- ERROR: 'batch_update_presentation' tool not found in MCP spec. ---")
                return {"error": "Required 'batch_update_presentation' Google Slides MCP tool not found"}
            print("--- DEBUG: Google Slides MCP tools retrieved successfully ---")

        except Exception as e:
             print(f"--- ERROR: Exception while retrieving tools from McpToolSpec: {e} ---")
             traceback.print_exc()
             return {"error": f"Failed to retrieve tools from McpToolSpec: {e}"}

        # --- 3. Create the Presentation using the Retrieved Tool ---
        presentation_title = outline.get("title", "Project Presentation")
        raw_create_output = None # For error reporting
        create_response_str = None

        try:
            print(f"--- DEBUG: Calling create_presentation_tool with title: '{presentation_title}' ---")
            # *** Use the retrieved tool ***
            tool_output_object = create_presentation_tool(title=presentation_title)
            raw_tool_output = tool_output_object # Store for potential error reporting
            print(f"--- DEBUG: Raw output type from self.create_presentation_tool: {type(tool_output_object)} ---")

            # --- Revised Handling V3: Regex Extraction ---
            create_response_str = None # Initialize string variable
            if isinstance(tool_output_object, ToolOutput):
                print("--- DEBUG: create_presentation_tool returned ToolOutput ---")
                content_value_str = tool_output_object.content # We know this is a string

                print(f"--- DEBUG (Inspection): Type of ToolOutput.content: {type(content_value_str)} ---")
                log_snippet_repr = repr(content_value_str[:300]) # Log prefix of the repr string
                print(f"--- DEBUG (Inspection): Value of ToolOutput.content (string repr): {log_snippet_repr}... ---")

                # Use regex to find the JSON part within text='{...}' inside the repr string
                # This regex looks for text=' followed by { ... } ending just before ')]
                match = re.search(r"text='(\{[\s\S]*?\})'\)\]$", content_value_str.strip()) # Added strip() for safety

                if match:
                    extracted_json_text = match.group(1) # Get the captured JSON block string
                    # Basic check for validity
                    if extracted_json_text.startswith('{') and extracted_json_text.endswith('}'):
                        print("--- DEBUG: Extracted potential JSON using regex from ToolOutput.content repr ---")
                        # Assign the extracted JSON string
                        create_response_str = extracted_json_text
                        # Note: json.loads handles standard JSON escapes like \n, \t, \\, \"
                    else:
                        print(f"--- WARNING: Regex match did not produce valid start/end braces. Match: {extracted_json_text[:100]}... ---")
                        # Keep create_response_str as None
                else:
                    print("--- WARNING: Regex did not find expected JSON pattern in ToolOutput.content string repr. ---")
                    # Keep create_response_str as None

                # Check if extraction failed
                if create_response_str is None:
                    print(f"--- ERROR: Could not extract JSON string from ToolOutput.content using regex. Content was: {content_value_str} ---")
                    return {"error": "Could not extract JSON string from ToolOutput content via regex", "response": content_value_str}
            # --- *** END REVISED HANDLING V3 *** ---

            elif isinstance(tool_output_object, str):
                # This case is less likely now but kept as fallback
                print("--- DEBUG: create_presentation_tool returned str directly ---")
                create_response_str = tool_output_object
            else:
                # Handle unexpected return type immediately
                print(f"--- ERROR: Unexpected return type from create_presentation tool: {type(tool_output_object)} ---")
                return {"error": f"Unexpected return type: {type(tool_output_object)}", "response": repr(tool_output_object)}

            # --- Now, ONLY proceed if we have a valid string ---
            if create_response_str is not None and isinstance(create_response_str, str):
                # (Rest of the json.loads logic remains the same)
                log_snippet = create_response_str[:200].replace('\n', '\\n')
                print(f"--- DEBUG: Attempting json.loads on string: '{log_snippet}...' ---")
                try:
                    create_response = json.loads(create_response_str)
                    presentation_id = create_response.get("presentationId") # Use .get for safety
                    if not presentation_id:
                        print(f"--- ERROR: Failed to get presentationId from parsed JSON. JSON was: {create_response} ---")
                        return {"error": "Failed to create presentation (no presentationId in response)", "response": create_response} # Return parsed JSON
                    print(f"--- DEBUG: Successfully created presentation ID: {presentation_id} ---")
                except json.JSONDecodeError as json_e:
                    print(f"--- ERROR: JSONDecodeError parsing tool response: {json_e} ---")
                    print(f"--- ERROR: String content that failed parsing: {create_response_str}") # Log the full string
                    return {"error": "Invalid JSON response from create_presentation tool", "response": create_response_str} # Return raw string
            else:
                # This case signifies an error in the handling logic above if reached.
                print("--- ERROR: create_response_str is not a valid string after tool call handling ---")
                return {"error": "Failed to get a valid string response from create_presentation tool", "response": repr(tool_output_object)}

        except Exception as e:
            # Catch other potential errors during the tool call itself
            print(f"--- ERROR: Exception during create_presentation tool call/processing: {e} ---")
            traceback.print_exc() # Log the full traceback
            raw_output_repr = repr(raw_create_output) if raw_create_output is not None else "N/A"
            return {"error": f"Error calling/processing create_presentation_tool: {e}", "raw_output": raw_output_repr}

        # --- Crucial Check: Ensure presentation_id was successfully obtained before proceeding ---
        if not presentation_id:
            print("--- ERROR: presentation_id not set after creation attempt, cannot proceed with slides ---")
            # The specific error should have been returned above, but add a fallback.
            return {"error": "Failed to obtain presentation ID, cannot add slides", "details": "Check previous logs for specific error during creation"}

        # --- 4. Prepare and Send Batch Update Requests ---
        print(f"--- DEBUG: Proceeding to build batch requests for presentation {presentation_id} ---")
        slides = outline.get("slides", [])
        # Initialize results *after* confirming presentation_id
        results = {"presentation_id": presentation_id, "slides": []}

        # Create batch requests for all slides (This part remains the same as your original logic)
        batch_requests = []
        for i, slide in enumerate(slides):
            slide_title = slide.get("title", "")
            # Join content lines, ensuring they are strings
            content = "\n".join(map(str, slide.get("content", [])))
            speaker_notes = slide.get("speaker_notes", "")

            slide_obj_id = f"slide_{i+1}" # Use a consistent variable
            title_obj_id = f"title_{i+1}"
            body_obj_id = f"body_{i+1}"
            notes_obj_id = f"notes_{i+1}"

            # Create slide request
            batch_requests.append({
                "createSlide": {
                    "objectId": slide_obj_id,
                    "slideLayoutReference": {"predefinedLayout": "TITLE_AND_BODY"},
                    "placeholderIdMappings": [
                        {"layoutPlaceholder": {"type": "TITLE"}, "objectId": title_obj_id},
                        {"layoutPlaceholder": {"type": "BODY"}, "objectId": body_obj_id}
                    ]
                }
            })
            # Add title text request
            if slide_title: # Avoid inserting empty text
                 batch_requests.append({"insertText": {"objectId": title_obj_id, "text": slide_title}})
            # Add body text request
            if content: # Avoid inserting empty text
                batch_requests.append({"insertText": {"objectId": body_obj_id, "text": content}})
            # Add speaker notes request if available
            if speaker_notes:
                batch_requests.append({
                    "createSpeakerNotes": { # Simpler request if text comes after
                         "slideObjectId": slide_obj_id,
                         "objectId": notes_obj_id
                     }
                 })
                batch_requests.append({
                    "insertText": {
                         "objectId": notes_obj_id,
                         "text": speaker_notes,
                         "insertionIndex": 0 # Start inserting at the beginning of notes textbox
                     }
                 })
                # Note: The original `createSpeakerNotesWithText` is fine too, this is just an alternative structure.

            results["slides"].append({
                "title": slide_title,
                "slide_id": slide_obj_id, # Use the variable
                "status": "pending"
            })

        # Send the batch update if requests were generated
        if batch_requests:
            print(f"--- DEBUG: Sending {len(batch_requests)} batch requests for presentation {presentation_id} ---")
            batch_response_str = None
            raw_batch_output = None
            try:
                # Call the retrieved batch update tool
                batch_tool_output_or_str = batch_update_tool(
                    presentationId=presentation_id,
                    requests=batch_requests
                )
                raw_batch_output = batch_tool_output_or_str
                print(f"--- DEBUG: Raw output type from batch_update_tool: {type(batch_tool_output_or_str)} ---")

                # Handle ToolOutput vs str for batch_update_tool response
                if isinstance(batch_tool_output_or_str, ToolOutput):
                    print("--- DEBUG: batch_update_tool returned ToolOutput ---")
                    if isinstance(batch_tool_output_or_str.content, str):
                         batch_response_str = batch_tool_output_or_str.content
                         # Decide if ToolOutput always means error for batch, or if content needs parsing
                         # For now, assume content might be valid JSON or error JSON
                    else:
                         print(f"--- WARNING: batch_update_tool ToolOutput content is not a string, type: {type(batch_tool_output_or_str.content)}. Treating as potential error info. ---")
                         # Store the non-string content representation in details
                         results["error"] = "Batch update tool returned non-string ToolOutput content"
                         results["details"] = repr(batch_tool_output_or_str.content)
                         for slide_res in results["slides"]: slide_res["status"] = "batch_failed_non_string_output"
                         # Return early if this is considered a fatal error
                         # return results
                elif isinstance(batch_tool_output_or_str, str):
                    print("--- DEBUG: batch_update_tool returned str ---")
                    batch_response_str = batch_tool_output_or_str
                else:
                    # Handle unexpected return type
                    results["error"] = f"Unexpected return type from batch_update tool: {type(batch_tool_output_or_str)}"
                    results["details"] = repr(batch_tool_output_or_str) # Store raw output representation
                    for slide_res in results["slides"]: slide_res["status"] = "batch_type_error"
                    return results # Return results dict with error added

                # Try parsing the batch response string if we got one
                if batch_response_str:
                    print(f"--- DEBUG: Attempting json.loads on batch response string (first 200 chars): '{batch_response_str[:200].replace(r'%5Cn', r'\\n') }...' ---") # Added replace for better readability if encoded newlines exist
                    try:
                        batch_response = json.loads(batch_response_str)
                        print("--- DEBUG: Batch update response JSON parsed successfully. ---")
                        # Optional: Check batch_response content for API-level errors from Google Slides if needed
                        # e.g., check if batch_response['replies'] contains errors

                        # Update status in results on successful parsing (might refine based on actual API errors)
                        if "error" not in results: # Don't overwrite previous errors unless success confirmed
                           for i in range(len(results["slides"])):
                               results["slides"][i]["status"] = "created" # Or check replies status
                    except json.JSONDecodeError as batch_json_e:
                        print(f"--- WARNING: JSONDecodeError parsing batch_update tool response: {batch_json_e}. Presentation might be partially created. ---")
                        # Error parsing the string response from batch update
                        results["warning"] = "Invalid JSON response from batch_update_presentation tool (slides might be created/partially created)" # Changed to warning
                        results["details"] = batch_response_str # Store the raw string
                        for slide_res in results["slides"]: slide_res["status"] = "batch_json_error"
                        # Don't return immediately, let the function return the results dict containing warning info
                # else: If batch_response_str is None (e.g., due to non-string ToolOutput handled above), error is already set.

            except Exception as batch_e:
                # Catch other potential errors during batch update tool call/processing
                print(f"--- ERROR: Exception during batch_update call or processing: {batch_e} ---")
                traceback.print_exc()
                results["error"] = f"Error during batch_update call or processing: {batch_e}"
                results["raw_batch_output"] = repr(raw_batch_output) if raw_batch_output else "N/A"
                for slide_res in results["slides"]: slide_res["status"] = "batch_exception"
                # Don't return immediately

        # --- 5. Return Results ---
        # Return the results dictionary, which includes presentation_id
        # and slide statuses, potentially with error/warning/details fields.
        print("--- DEBUG: Returning results from create_google_slides ---")
        return results
    
    def create_presentation(self, code_content: str, documentation_content: str) -> Dict[str, Any]:
        """Create a complete presentation from code and documentation."""
        # Analyze the content
        analysis = self._analyze_content(code_content, documentation_content)
        
        # Create presentation outline
        outline = self._create_presentation_outline(analysis)
        
        results = {
            "analysis": analysis,
            "outline": outline,
            "slides": None
        }
        
        # Create Google Slides if available
        if self.slides_tool:
            slides_result = self.create_google_slides(outline)
            results["slides"] = slides_result
        
        # Add token usage statistics
        results["tokens_used"] = {
            "prompt_tokens": self.token_counter.prompt_llm_token_count,
            "completion_tokens": self.token_counter.completion_llm_token_count,
            "total_tokens": self.token_counter.total_llm_token_count
        }
        
        return results


def handle_task(task_description: str) -> str:
    """
    Main entry point for the Presentation Agent.
    Args:
        task_description: Description of the presentation task, should include
                          references to code and documentation results
    Returns:
        Generated presentation information as a JSON string
    """
    agent = None # Initialize agent to None
    result = None # Initialize result to None

    try:
        print("--- DEBUG: Initializing PresentationAgent ---")
        agent = PresentationAgent()
        print("--- DEBUG: PresentationAgent Initialized ---")

        # Extract code and documentation
        code_match = re.search(r'CODE:\s*```(?:python)?\s*([\s\S]*?)\s*```', task_description)
        doc_match = re.search(r'DOCUMENTATION:\s*```(?:markdown)?\s*([\s\S]*?)\s*```', task_description)
        code_content = code_match.group(1) if code_match else "# No code provided"
        documentation_content = doc_match.group(1) if doc_match else "No documentation provided"

        print("--- DEBUG: Extracted code and documentation ---")
        print(f"--- DEBUG: Code provided: {bool(code_content and code_content != '# No code provided')}")
        print(f"--- DEBUG: Docs provided: {bool(documentation_content and documentation_content != 'No documentation provided')}")

        # Generate presentation
        print("--- DEBUG: Calling agent.create_presentation ---")
        result = agent.create_presentation(code_content, documentation_content)
        print("--- DEBUG: agent.create_presentation call completed ---")

        # *** Inspect the result before trying to dump it ***
        print(f"--- DEBUG: Type of result: {type(result)}")
        print("--- DEBUG: Content of result before json.dumps:")
        # Use pprint for better dictionary visualization
        pprint.pprint(result)

        # Format the result as JSON
        print("--- DEBUG: Attempting json.dumps ---")
        json_result = json.dumps(result, indent=2)
        print("--- DEBUG: json.dumps successful ---") # Only prints if it works
        return json_result

    except Exception as e:
        # *** Enhanced Error Logging ***
        print(f"\n--- ERROR caught in handle_task ---")
        print(f"--- ERROR Type: {type(e)}")
        print(f"--- ERROR Details: {e}")
        print("--- ERROR Traceback: ---")
        traceback.print_exc() # Print the full traceback to see where the error originated

        # Try to represent the problematic result object safely if it exists
        safe_result_repr = "Result object not available or not yet assigned"
        if 'result' in locals() and result is not None:
             print("--- ERROR: Inspecting result object at time of error ---")
             pprint.pprint(result) # Try printing the structure again
             try:
                 # Try a safer representation in case pprint fails
                 safe_result_repr = repr(result)
             except Exception as repr_e:
                 safe_result_repr = f"Could not represent result object: {repr_e}"

        # Create a JSON-safe error dictionary
        error_result = {
            "error": f"Caught exception: {type(e).__name__} - {e}",
            "status": "failed",
            "result_at_error_repr": safe_result_repr # Include safe representation
        }
        # Return the error info as JSON
        return json.dumps(error_result, indent=2)

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
    test_task = """
    Create a presentation for a project that implements a chat assistant.
    
    CODE:
    ```python
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
                    "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
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
    ```
    
    DOCUMENTATION:
    ```markdown
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
    {
      "messages": [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well! How can I help you today?"},
        {"role": "user", "content": "Tell me about the solar system."}
      ],
      "model": "anthropic/claude-3-opus",
      "temperature": 0.7
    }
    ```
    
    ### Response
    
    ```json
    {
      "response": "The solar system consists of the Sun and everything that orbits around it..."
    }
    ```
    
    ## Implementation Details
    
    The server is built with FastAPI and communicates with OpenRouter's API to access various LLMs. 
    It handles authentication, request formatting, and error handling.
    
    ## Deployment
    
    The service can be deployed on Render.com for free. Environment variables must be set for the OpenRouter API key.
    ```
    """
    
    print(handle_task(test_task))