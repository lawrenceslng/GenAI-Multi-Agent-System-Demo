"""
Code Agent Template for agent to be run inside Docker Container for Multi-Agent Homework System
Responsible for generating and testing Python code in a Docker sandbox
"""

import json
import os
import logging
import socket
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
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
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

def log_container_info():
    """Log information about the Docker container environment."""
    logger = setup_logging("container_info")
    logger.info("🐳 Docker Container Initialized")
    logger.info("================================")
    logger.info(f"🖥️  Hostname: {socket.gethostname()}")
    logger.info(f"👤 Running as user: {os.getuid()}")
    logger.info(f"📁 Working directory: {os.getcwd()}")
    logger.info(f"📚 Sandbox directory mounted: {SANDBOX_DIR.exists()}")
    logger.info(f"💾 Workspace directory: {WORKSPACE_DIR}")
    logger.info("================================")

# Initialize LLM and MCP Client
llm = OpenAI(model="gpt-4.1-nano-2025-04-14", api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Github MCP client
logger = setup_logging("github_mcp")
github_mcp_client = BasicMCPClient(
    command_or_url=os.getenv("GITHUB_MCP_URL"),
    args=["stdio"],
    env={
        "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PAT")
    }
)
github_mcp_tool = McpToolSpec(client=github_mcp_client)

# Define paths - sandbox is mounted read-only
SANDBOX_DIR = Path("sandbox")  # Mounted read-only from host
WORKSPACE_DIR = Path("workspace")  # Container's writable directory
INSTRUCTIONS_FILE = SANDBOX_DIR / "instructions"

def normalize_requirements(requirements: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize requirements dictionary to ensure consistent keys."""
    normalized = {
        "main_task": requirements.get("main_task", ""),
        "files_to_create": requirements.get("files_to_create") or requirements.get("files", []),
        "required_libraries": requirements.get("required_libraries") or requirements.get("libraries", [])
    }
    return normalized

async def read_instructions() -> str:
    """Tool to read instructions from the instructions file."""
    logger = setup_logging("read_instructions")
    logger.info("Reading instructions")
    
    if not INSTRUCTIONS_FILE.exists():
        raise FileNotFoundError("Instructions file not found")
    
    instructions = INSTRUCTIONS_FILE.read_text()
    logger.info("Instructions read successfully")
    return instructions

async def analyze_requirements(task_description: str) -> Dict:
    """Tool to analyze and extract coding requirements from task description."""
    logger = setup_logging("analyze_requirements")
    logger.info("Analyzing requirements from task description")
    
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
        
        requirements_dict = json.loads(requirements.text)
        normalized = normalize_requirements(requirements_dict)
        logger.info("Requirements analysis completed")
        return normalized
        
    except Exception as e:
        logger.error(f"Error in analyze_requirements: {str(e)}")
        raise

async def generate_code(requirements: Dict) -> Dict[str, str]:
    """Tool to generate code files and store them in memory."""
    logger = setup_logging("generate_code")
    logger.info("Generating code files")
    generated_files = {}
    
    try:
        normalized = normalize_requirements(requirements)
        for file_name in normalized["files_to_create"]:
            logger.info(f"Generating {file_name}")
            
            if file_name == "requirements.txt":
                content = "\n".join(normalized["required_libraries"])
            else:
                response = await llm.acomplete(
                    f"""Generate the content for {file_name} based on these requirements:
                    {json.dumps(normalized, indent=2)}
                    
                    Follow these guidelines:
                    1. Include proper imports for required libraries
                    2. Add type hints and docstrings
                    3. Include error handling
                    4. Make code modular and well-structured
                    
                    Return only the code, no explanations or markdown formatting.
                    """
                )
                content = response.text
            
            # Store generated content in memory
            generated_files[file_name] = content
            logger.info(f"Generated {file_name}")
            
        logger.info("Code generation completed")
        return generated_files
        
    except Exception as e:
        logger.error(f"Error in generate_code: {str(e)}")
        raise

async def generate_repo_name(requirements: Dict) -> str:
    """Tool to generate a creative repository name based on project requirements."""
    logger = setup_logging("generate_repo_name")
    logger.info("Generating creative repository name")
    
    try:
        normalized = normalize_requirements(requirements)
        response = await llm.acomplete(
            f"""Based on this project description and requirements:
            {json.dumps(normalized, indent=2)}
            
            Generate a creative, memorable, and professional repository name that:
            1. Reflects the main purpose of the project
            2. Is concise (2-4 words)
            3. Uses kebab-case (words separated by hyphens)
            4. Avoids generic terms like "project" or "app"
            5. Is unique and catchy
            
            Return only the repository name, no explanations.
            Example formats: "quantum-pathfinder", "neural-butterfly", "swift-data-weaver"
            """
        )
        
        repo_name = response.text.strip().lower()
        repo_name = re.sub(r'[^a-z0-9-]', '-', repo_name)
        repo_name = re.sub(r'-+', '-', repo_name)
        repo_name = repo_name.strip('-')
        
        logger.info(f"Generated repository name: {repo_name}")
        return repo_name
        
    except Exception as e:
        logger.error(f"Error generating repository name: {str(e)}")
        raise

async def create_github_repo(requirements: Dict) -> Dict[str, str]:
    """Tool to create a new Github repository with a creative name."""
    logger = setup_logging("create_github_repo")
    logger.info("Creating Github repository")
    
    try:
        repo_name = await generate_repo_name(requirements)
        
        github_tools = await github_mcp_tool.to_tool_list_async()
        if not github_tools:
            raise Exception("No tools returned from Github MCP server")
        
        create_repo_tool = next(t for t in github_tools if hasattr(t, '_metadata') and hasattr(t._metadata, 'name') and t._metadata.name == "create_repository")
        
        response = await create_repo_tool.acall(name=repo_name, private=False)
        
        match = re.search(r"text='({.*?})'", response.content)
        if not match:
            raise Exception("Could not find JSON content in the response")
            
        data = json.loads(match.group(1))
        repo_info = {
            "owner": data["owner"]["login"],
            "name": data["name"],
            "url": data["html_url"]
        }
        
        logger.info(f"Repository created: {repo_info['url']}")
        return repo_info
        
    except Exception as e:
        logger.error(f"Error creating Github repository: {str(e)}")
        raise

async def push_to_github(repo_info: Dict[str, str], files: Dict[str, str]) -> None:
    """Tool to push generated files to Github repository."""
    logger = setup_logging("push_to_github")
    logger.info("Pushing files to Github")
    
    try:
        github_tools = await github_mcp_tool.to_tool_list_async()
        create_file_tool = next(t for t in github_tools if hasattr(t, '_metadata') and hasattr(t._metadata, 'name') and t._metadata.name == "create_or_update_file")
        
        for file_name, content in files.items():
            logger.info(f"Pushing file: {file_name}")
            try:
                await create_file_tool.acall(
                    owner=repo_info["owner"],
                    repo=repo_info["name"],
                    path=file_name,
                    content=content,
                    message=f"Add {file_name}",
                    branch="main"
                )
                logger.info(f"Successfully pushed {file_name}")
            except Exception as e:
                logger.error(f"Error pushing {file_name}: {str(e)}")
                raise
            
        logger.info("All files pushed successfully")
        
    except Exception as e:
        logger.error(f"Error pushing to Github: {str(e)}")
        raise

async def handle_task(ctx: Context) -> str:
    """
    Main entry point for the Code Agent.
    Provides access to all tools and lets the agent decide how to use them.
    """
    logger = setup_logging()
    logger.info("Starting code agent task")
    
    try:
        # Log container initialization info
        log_container_info()
        
        # Get instructions first
        instructions = await read_instructions()
        logger.info("Read instructions successfully")
        
        # Analyze requirements
        requirements = await analyze_requirements(instructions)
        logger.info("Analyzed requirements")
        
        # Generate code files (stored in memory)
        generated_files = await generate_code(requirements)
        logger.info(f"Generated files: {list(generated_files.keys())}")
        
        # Create Github repository
        repo_info = await create_github_repo(requirements)
        logger.info(f"Created repository: {repo_info['url']}")
        
        # Push files directly to Github
        await push_to_github(repo_info, generated_files)
        logger.info("Pushed files to Github")
        
        # Prepare success response
        result = {
            "status": "success",
            "message": "Code generation and Github push completed",
            "files_generated": list(generated_files.keys()),
            "requirements": requirements,
            "github_repository": repo_info["url"]
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error in handle_task: {str(e)}")
        error_result = {
            "status": "error",
            "error": str(e)
        }
        return json.dumps(error_result, indent=2)

async def create_agent():
    """Create the Code Agent with LlamaIndex."""
    return FunctionAgent(
        name="CodeAgent",
        description="Expert Python programmer that generates code and manages Github repositories",
        system_prompt="""You are an expert Python programmer with strong problem-solving skills and deep understanding of software development best practices.

Your goal is to complete coding assignments by using the available tools in the most effective way. You have complete autonomy in deciding:

1. What information you need to gather
2. Which tools to use and in what order
3. How to structure the solution
4. When to create the repository and push code

Available Tools:
- read_instructions: Get the assignment details
- analyze_requirements: Extract structured requirements
- generate_code: Create well-structured Python code
- create_github_repo: Make a Github repo with creative name
- push_to_github: Push code to Github

Key Principles:
- You can use tools in any order that makes sense
- You can skip tools if they're not needed
- You can use tools multiple times if necessary
- You should make decisions based on the context
- You should handle errors gracefully

For example:
- You might want to analyze requirements before reading instructions
- You might want to create the repo after generating some code
- You might want to push code in multiple commits
- You might want to generate code files in a specific order

Think strategically about:
- What information you need at each step
- How to handle dependencies between steps
- When to commit code to Github
- How to structure the solution effectively

Always ensure code is:
- Well-structured and documented
- Includes proper error handling
- Uses type hints
- Follows Python best practices
- Properly version controlled""",
        tools=[
            FunctionTool.from_defaults(fn=read_instructions, name="read_instructions"),
            FunctionTool.from_defaults(fn=analyze_requirements, name="analyze_requirements"),
            FunctionTool.from_defaults(fn=generate_code, name="generate_code"),
            FunctionTool.from_defaults(fn=create_github_repo, name="create_github_repo"),
            FunctionTool.from_defaults(fn=push_to_github, name="push_to_github"),
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