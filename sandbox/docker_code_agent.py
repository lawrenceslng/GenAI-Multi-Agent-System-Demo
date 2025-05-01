"""
Code Agent Template for agent to be run inside Docker Container for Multi-Agent Homework System
Responsible for generating and testing Python code in a Docker sandbox
"""

import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import re

from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.workflow import Context
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

load_dotenv()

def setup_logging(logger_name: str = "code_agent") -> logging.Logger:
    """Set up logging for the code agent."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # Console handler with simple formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

# Initialize LLM and MCP Client
llm = OpenAI(model="gpt-4.1-nano-2025-04-14", api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Github MCP client
# Set up logger for MCP initialization
logger = setup_logging("github_mcp")
github_mcp_client = BasicMCPClient(
    command_or_url=os.getenv("GITHUB_MCP_URL"),
    args=["stdio"],
    env={
        "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PAT")
    }
)
github_mcp_tool = McpToolSpec(client=github_mcp_client)

class CodeAgent:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.sandbox_dir = Path("sandbox")
        self.project_dir = self.sandbox_dir / "project"
        self.instructions_file = self.sandbox_dir / "instructions"
        self.logger = logger or logging.getLogger("code_agent")
        
        # Ensure project directory exists
        self.project_dir.mkdir(exist_ok=True)
        
    def read_instructions(self) -> str:
        """Read instructions from the instructions file."""
        if not self.instructions_file.exists():
            raise FileNotFoundError("Instructions file not found")
        return self.instructions_file.read_text()
        
    async def analyze_requirements(self, task_description: str) -> Dict:
        """Extract coding requirements from task description."""
        self.logger.info("Analyzing requirements from task description")
        
        try:
            requirements = await llm.acomplete(
                f"""Analyze the following assignment and extract key coding requirements:
                {task_description}
                
                Return your analysis in this exact JSON format:
                {{
                    "main_task": "Brief description of the main task",
                    "files_to_create": ["main.py", "requirements.txt", "README.md"],
                    "required_libraries": ["list", "of", "required", "python", "libraries"]
                }}
                
                Ensure your response is valid JSON with these exact keys.
                """
            )
            
            # Parse requirements
            requirements_dict = json.loads(requirements.text)
            self.logger.info("Requirements analysis completed")
            return requirements_dict
            
        except Exception as e:
            self.logger.error(f"Error in analyze_requirements: {str(e)}")
            raise

    async def generate_code(self, requirements: Dict) -> Dict[str, str]:
        """Generate code files based on requirements."""
        self.logger.info("Generating code files")
        generated_files = {}
        
        try:
            for file_name in requirements["files_to_create"]:
                self.logger.info(f"Generating {file_name}")
                
                # Generate content for this file
                if file_name == "requirements.txt":
                    # For requirements.txt, just list the required libraries
                    content = "\n".join(requirements["required_libraries"])
                else:
                    # For code files, use LLM to generate content
                    response = await llm.acomplete(
                        f"""Generate the content for {file_name} based on these requirements:
                        {json.dumps(requirements, indent=2)}
                        
                        Follow these guidelines:
                        1. Include proper imports for required libraries
                        2. Add type hints and docstrings
                        3. Include error handling
                        4. Make code modular and well-structured
                        
                        Return only the code, no explanations or markdown formatting.
                        """
                    )
                    content = response.text
                
                # Save file to project directory
                file_path = self.project_dir / file_name
                file_path.write_text(content)
                generated_files[file_name] = content
                
            self.logger.info("Code generation completed")
            return generated_files
            
        except Exception as e:
            self.logger.error(f"Error in generate_code: {str(e)}")
            raise

async def handle_task(ctx: Context) -> str:
    """
    Main entry point for the Code Agent.
    Reads instructions, generates code, and saves to project directory.
    """
    try:
        # Set up logging
        logger = setup_logging()
        logger.info("Starting code agent task")
        
        # Get the agent instance
        agent_instance = await get_code_agent()

        # Create code agent instance
        code_agent = CodeAgent(logger=logger)
        
        # Read instructions
        instructions = code_agent.read_instructions()
        logger.info("Read instructions successfully")
        
        # Analyze requirements
        requirements = await code_agent.analyze_requirements(instructions)
        logger.info("Requirements analyzed")
        
        # Generate code files
        generated_files = await code_agent.generate_code(requirements)
        logger.info("Code files generated")
        
        # Create Github repository
        try:
            repo_name = f"coding-assignment-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            # Get Github tools
            logger.info("Initializing Github MCP tools...")
            github_tools = await github_mcp_tool.to_tool_list_async()
            if not github_tools:
                raise Exception("No tools returned from Github MCP server")
            
            # Log available tools
            tool_names = [t._metadata.name for t in github_tools if hasattr(t, '_metadata') and hasattr(t._metadata, 'name')]
            logger.info(f"Retrieved {len(github_tools)} tools from Github MCP server")
            logger.info(f"Available tools: {', '.join(tool_names)}")
            
            # Verify required tools are available
            required_tools = ['create_repository', 'create_or_update_file']
            missing_tools = [tool for tool in required_tools if tool not in tool_names]
            if missing_tools:
                raise Exception(f"Missing required tools: {', '.join(missing_tools)}. Available tools: {', '.join(tool_names)}")
            
            # Get required tool instances
            create_repo_tool = next(t for t in github_tools if hasattr(t, '_metadata') and hasattr(t._metadata, 'name') and t._metadata.name == "create_repository")
            create_file_tool = next(t for t in github_tools if hasattr(t, '_metadata') and hasattr(t._metadata, 'name') and t._metadata.name == "create_or_update_file")
            logger.info("Required Github tools found")
            
            # Create repository
            response = await create_repo_tool.acall(name=repo_name, private=False)
            if not hasattr(response, 'content') or not response.content:
                raise Exception(f"Unexpected repository creation response format: {response}")
            
            print(response)
            print(type(response))
            print('-'*50)
            print(response.content)
            print(type(response.content))
            print('-'*50)
            print(response.raw_input)
            print(type(response.raw_input))
            print('-'*50)
            print(response.raw_output)
            print(type(response.raw_output))
            
            print('-'*50)
            
            # text_content = next((c for c in response.content if hasattr(c, 'type') and c.type == 'text'), None)
            # text_content = next((c for c in response.content if c.type == 'text'), None)
            # if not text_content:
            #     raise Exception("No text content found in repository creation response")
            
            # Parse repository data
            # repo_data = json.loads(response.content[0].text)
            # repo_result = {'html_url': repo_data['html_url']}
            # logger.info(f"Repository created: {repo_result['html_url']}")

            # Step 1: Use regex to extract the JSON part
            match = re.search(r"text='({.*?})'", response.content)

            if match:
                json_string = match.group(1)  # the content inside `text='...'`
                
                # Step 2: Parse JSON
                try:
                    data = json.loads(json_string)
                    repo_owner = data["owner"]["login"]
                    repo_name = data["name"]
                    repo_url = data["html_url"]
                    print("Repo Owner:", repo_owner)
                    print("Repo Name:", repo_name)
                    print("Repo URL:", repo_url)
                except json.JSONDecodeError as e:
                    print("Error decoding JSON:", e)
            else:
                print("Could not find JSON content in the response.")
            
            for file_name, content in generated_files.items():
                try:
                    logger.info(f"Creating file: {file_name}")
                    
                    await create_file_tool.acall(
                        owner=repo_owner,
                        repo=repo_name,
                        path=file_name,
                        content=content,
                        message=f"Add {file_name}",
                        branch="main"
                    )
                    logger.info(f"File {file_name} created successfully")
                except Exception as e:
                    logger.error(f"Error creating file {file_name}: {str(e)}")
                    continue
            
            logger.info("All files pushed successfully")
            
        except Exception as e:
            logger.error(f"Github repository operation failed: {str(e)}")
            raise
        
        # Prepare success response
        result = {
            "status": "success",
            "message": "Code generation and Github push completed",
            "files_generated": list(generated_files.keys()),
            "requirements": requirements,
            "github_repository": repo_url
        }
        logger.info("sending back result")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error in handle_task: {str(e)}")
        error_result = {
            "status": "error",
            "error": str(e)
        }
        return json.dumps(error_result, indent=2)

# Create the Code Agent with LlamaIndex
async def create_agent():
    # # Get Github tools
    # github_tools = await github_mcp_tool.to_tool_list_async()
    
    return FunctionAgent(
        name="CodeAgent",
        description="Generates code based on assignment instructions and manages Github repositories",
        system_prompt="""You are an expert Python programmer that generates code for assignments and manages Github repositories.
        
        Your workflow:
        1. Read instructions from the sandbox/instructions file
        2. Analyze requirements and determine needed files
        3. Generate code and save to sandbox/project directory
        4. Create a Github repository and push the code
        5. Return a status report with repository URL
        
        Always ensure code is:
        - Well-structured and documented
        - Includes proper error handling
        - Uses type hints
        - Follows Python best practices
        - Properly version controlled with Git""",
        tools=[
            FunctionTool.from_defaults(fn=handle_task, name="HandleCodingTask"),
            # *github_tools
        ],
        llm=llm,
        verbose=True
    )

# Create initial code agent instance
code_agent = None

async def get_code_agent():
    """Get or create the code agent instance."""
    global code_agent
    if code_agent is None:
        code_agent = await create_agent()
    return code_agent

if __name__ == "__main__":
    # For testing the agent directly
    import asyncio
    
    async def test_agent():
        agent = await get_code_agent()
        workflow = AgentWorkflow(agents=[agent], root_agent=agent.name)
        context = Context(workflow=workflow)
        result = await handle_task(context)
        print(result)
    
    asyncio.run(test_agent())