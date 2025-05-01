"""
Code Agent Template for agent to be run inside Docker Container for Multi-Agent Homework System
Responsible for generating and testing Python code in a Docker sandbox
"""

import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.workflow import Context

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

# Initialize LLM
llm = OpenAI(model="gpt-4.1-nano-2025-04-14", api_key=os.getenv("OPENAI_API_KEY"))

# llm = OpenRouter(
    # api_key=os.getenv("OPENROUTER_API_KEY"),
    # max_tokens=256,
    # context_window=4096,
    # model="google/gemini-2.0-flash-exp:free",
# )

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
        
        # Create agent instance
        agent = CodeAgent(logger=logger)
        
        # Read instructions
        instructions = agent.read_instructions()
        logger.info("Read instructions successfully")
        
        # Analyze requirements
        requirements = await agent.analyze_requirements(instructions)
        logger.info("Requirements analyzed")
        
        # Generate code files
        generated_files = await agent.generate_code(requirements)
        logger.info("Code files generated")
        
        # Prepare success response
        result = {
            "status": "success",
            "message": "Code generation completed",
            "files_generated": list(generated_files.keys()),
            "requirements": requirements
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
code_agent = FunctionAgent(
    name="CodeAgent",
    description="Generates code based on assignment instructions",
    system_prompt="""You are an expert Python programmer that generates code for assignments.
    
    Your workflow:
    1. Read instructions from the sandbox/instructions file
    2. Analyze requirements and determine needed files
    3. Generate code and save to sandbox/project directory
    4. Return a status report
    
    Always ensure code is:
    - Well-structured and documented
    - Includes proper error handling
    - Uses type hints
    - Follows Python best practices""",
    tools=[
        FunctionTool.from_defaults(fn=handle_task, name="HandleCodingTask")
    ],
    llm=llm,
    verbose=True
)

if __name__ == "__main__":
    # For testing the agent directly
    import asyncio
    
    async def test_agent():
        workflow = AgentWorkflow(agents=[code_agent], root_agent=code_agent.name)
        context = Context(workflow=workflow)
        result = await handle_task(context)
        print(result)
    
    asyncio.run(test_agent())