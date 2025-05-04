#!/usr/bin/env python3
"""
Multi-Agent Homework System Orchestrator
Coordinates multiple agents to process homework assignments
"""

from dotenv import load_dotenv
load_dotenv()

import os
import sys
import json
import argparse
import re
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from enum import Enum

from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.workflow import Context

# Import specialized agents
from sandbox.docker_code_agent import get_code_agent
from agents.presentation_agent import get_presentation_agent
from agents.voiceover_agent import mcp_tool, get_agent, generate_and_save_audio

# Initialize LLM
llm = OpenAI(model="gpt-4.1-nano-2025-04-14", api_key=os.getenv("OPENAI_API_KEY"))
# Uncomment to use OpenRouter instead
# llm = OpenRouter(
#     api_key=os.getenv("OPENROUTER_API_KEY"),
#     max_tokens=256,
#     context_window=4096,
#     model="google/gemini-2.0-flash-exp:free",
# )

class TaskType(Enum):
    CODE = "code"
    DOCUMENTATION = "documentation"
    PRESENTATION = "presentation"
    VOICEOVER = "voiceover"

class OrchestratorAgent:
    def __init__(self):
        self.assignment_vault = Path("assignment_vault")
        self.result_dir = Path("result")
        self.sandbox_dir = Path("sandbox")
        
    def create_result_directory(self, assignment_name: str) -> Path:
        """Create a timestamped result directory for the current assignment."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = self.result_dir / f"{assignment_name}_{timestamp}"
        result_path.mkdir(parents=True, exist_ok=True)
        return result_path

    def read_assignment(self, assignment_file: str) -> str:
        """Read the assignment description from file."""
        file_path = self.assignment_vault / assignment_file
        if not file_path.exists():
            raise FileNotFoundError(f"Assignment file not found: {file_path}")
        
        return file_path.read_text()
        
    async def analyze_assignment(self, assignment_text: str) -> Dict:
        """Analyze assignment and come up with a creative implementation idea."""
        print("\n=== Starting Assignment Analysis ===")
        print(f"Assignment text length: {len(assignment_text)} chars")
        
        try:
            print("\nStep 1: Sending analysis request to LLM...")
            analysis = await llm.acomplete(
                f"""Analyze this assignment and propose a creative implementation:
                {assignment_text}
                
                Return your analysis in this exact JSON format:
                {{
                    "main_objective": "Brief description of the overall goal",
                    "creative_idea": {{
                        "summary": "Brief description of your creative implementation idea",
                        "why_interesting": "Why this would be an interesting solution"
                    }},
                    "required_tasks": [
                        {{
                            "type": "code|documentation|presentation|voiceover",
                            "description": "Specific task description",
                            "priority": 1-5 (1 highest)
                        }}
                    ]
                }}
                
                Important:
                - Come up with an original and interesting way to satisfy the requirements
                - Make it specific enough that a coding agent can implement it
                - Keep it realistic for the scope of the assignment
                - ALWAYS include at least these three task types: "code", "presentation", and "voiceover"
                  (The "code" task should have highest priority, followed by "presentation", then "voiceover")
                
                Ensure your response is valid JSON with these exact keys.
                """
            )
            print("✓ LLM analysis completed")
            
            print("\nStep 2: Parsing LLM response...")
            try:
                result = json.loads(analysis.text)
                print("✓ JSON parsing successful")
                
                # Validate required fields
                print("\nStep 3: Validating analysis structure...")
                required_fields = ["main_objective", "creative_idea", "required_tasks"]
                for field in required_fields:
                    if field not in result:
                        print(f"ERROR: Missing required field '{field}'")
                        raise ValueError(f"Analysis missing required field: {field}")
                    print(f"✓ Found required field: {field}")
                
                if "creative_idea" in result:
                    if not isinstance(result["creative_idea"], dict):
                        print("ERROR: 'creative_idea' is not a dictionary")
                        raise ValueError("Invalid creative_idea format")
                    for subfield in ["summary", "why_interesting"]:
                        if subfield not in result["creative_idea"]:
                            print(f"ERROR: Missing creative_idea subfield '{subfield}'")
                            raise ValueError(f"Creative idea missing required field: {subfield}")
                        print(f"✓ Found creative_idea subfield: {subfield}")
                
                if "required_tasks" in result:
                    tasks = result["required_tasks"]
                    print(f"\nFound {len(tasks)} required tasks:")
                    for i, task in enumerate(tasks, 1):
                        print(f"Task {i}:")
                        print(f"- Type: {task.get('type', 'Not specified')}")
                        print(f"- Priority: {task.get('priority', 'Not specified')}")
                
                print("\n✓ Analysis validation complete")
                print("=== Assignment Analysis Complete ===\n")
                return result
                
            except json.JSONDecodeError as e:
                print(f"ERROR: Failed to parse JSON response: {e}")
                print("Raw response:", analysis.text[:200] + "..." if len(analysis.text) > 200 else analysis.text)
                raise
            
        except Exception as e:
            print("\nERROR during assignment analysis:")
            print(f"- Error type: {type(e).__name__}")
            print(f"- Error message: {str(e)}")
            if hasattr(e, '__traceback__'):
                print(f"- Error location: {e.__traceback__.tb_frame.f_code.co_filename}:{e.__traceback__.tb_lineno}")
            print("=== Assignment Analysis Failed ===\n")
            raise Exception(f"Error analyzing assignment: {str(e)}")

    async def get_agent_for_task(self, task_type: str):
        """Get the appropriate agent for a given task type."""
        if task_type == TaskType.CODE.value:
            return await get_code_agent()
        elif task_type == TaskType.VOICEOVER.value:
            return await get_agent(mcp_tool)
        elif task_type == TaskType.PRESENTATION.value:
            return await get_presentation_agent()
        agents = {}
        return agents.get(task_type)

    async def execute_task(self, task: Dict, context: Context) -> Dict:
        """Execute a single task using the appropriate agent."""
        print(f"\n=== Executing Task: {task['type']} ===")
        print(f"Task description: {task.get('description', 'No description provided')}")
        print(f"Task priority: {task.get('priority', 'No priority set')}")
        
        print("\nInitializing agent...")
        agent = await self.get_agent_for_task(task["type"])
        if not agent:
            error_msg = f"No agent found for task type: {task['type']}"
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg)
        print("✓ Agent initialized successfully")

        # Update context with task-specific information
        print("\nUpdating context with task information...")
        await context.set("task", task)
        await context.set("agent", agent)
        
        # Create agent-specific context if needed
        if task["type"] == TaskType.VOICEOVER.value:
            try:
                agent_context = await context.get("agent_context")
            except ValueError:
                # Initialize agent_context if it doesn't exist
                agent_context = Context(agent)
                await context.set("agent_context", agent_context)
        
        print("✓ Context updated")
        
        try:
            print("\nExecuting task with specialized agent...")
            if task["type"] == TaskType.CODE.value:
                # For code agent, use handle_task
                from sandbox.docker_code_agent import handle_task
                result = await handle_task(context)
            elif task["type"] == TaskType.VOICEOVER.value:
                # For voiceover agent, use generate_and_save_audio
                # First, get the code task result from the context
                state = await context.get("state", {})
                completed_tasks = state.get("completed_tasks", [])
                
                # Find the most recent code task result
                code_task_result = None
                for completed_task in reversed(completed_tasks):
                    if completed_task.get("task_type") == TaskType.CODE.value:
                        code_task_result = completed_task
                        break
                
                if not code_task_result:
                    raise ValueError("No code task results found. Cannot create voiceover without code.")
                
                # Extract code result information
                code_result = code_task_result.get("result", {})
                if isinstance(code_result, str):
                    try:
                        code_result = json.loads(code_result)
                    except json.JSONDecodeError:
                        print("ERROR: Failed to parse code task result as JSON")
                        raise ValueError("Failed to parse code task result as JSON")
                
                # Create a script that explains the code and points out challenges and implementation details
                # Find the most recent presentation task result
                presentation_task_result = None
                for completed_task in reversed(completed_tasks):
                    if completed_task.get("task_type") == TaskType.PRESENTATION.value:
                        presentation_task_result = completed_task
                        break
                
                # Extract presentation information if available
                presentation_info = ""
                if presentation_task_result:
                    presentation_result = presentation_task_result.get("result", {})
                    if isinstance(presentation_result, str):
                        try:
                            presentation_result = json.loads(presentation_result)
                        except json.JSONDecodeError:
                            print("WARNING: Failed to parse presentation task result as JSON")
                    
                    if isinstance(presentation_result, dict):
                        presentation_id = presentation_result.get("presentation_id", "Not available")
                        presentation_info = f"Google Slides Presentation ID: {presentation_id}"
                
                script_prompt = f"""
                Create a detailed script for a voiceover that explains the following code implementation and presentation:
                
                Project Requirements: {code_result.get('requirements', {})}
                Files Generated: {code_result.get('files_generated', [])}
                GitHub Repository: {code_result.get('github_repository', 'Not available')}
                {presentation_info}
                
                The script should:
                1. Introduce the project and its purpose
                2. Explain the overall structure and architecture of the code
                3. Point out interesting implementation details and design choices
                4. Highlight any challenges that were addressed in the implementation
                5. Discuss how the code satisfies the project requirements
                6. Mention that a presentation was created to visualize the project
                7. Conclude with the value and potential applications of the project
                
                Make it engaging and informative for someone who wants to understand both the code and presentation.
                The voiceover should be 1 minute long when spoken.
                """
                
                script_response = await llm.acomplete(script_prompt)
                script = script_response.text
                
                # Get or create the agent context
                try:
                    agent_context = await context.get("agent_context")
                except ValueError:
                    print("Creating new agent context for voiceover task...")
                    agent_context = Context(agent)
                    await context.set("agent_context", agent_context)
                
                # Add retry logic for handling rate limits
                max_retries = 3
                retry_delay = 5  # seconds
                
                for attempt in range(max_retries):
                    try:
                        result = await generate_and_save_audio(script, agent, agent_context)
                        if "error" not in result:
                            break
                        if "quota_exceeded" in str(result.get("error", "")).lower() or "too_many_concurrent_requests" in str(result.get("error", "")).lower():
                            if attempt < max_retries - 1:
                                print(f"Rate limit hit, waiting {retry_delay} seconds before retry {attempt + 1}/{max_retries}...")
                                await asyncio.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                                continue
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"Error during attempt {attempt + 1}: {str(e)}")
                            print(f"Retrying in {retry_delay} seconds...")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2
                        else:
                            raise
            elif task["type"] == TaskType.PRESENTATION.value:
                print("\n=== Starting Presentation Task ===")
                print("Initializing presentation agent...")
                
                # For presentation agent, use _create_presentation_with_content
                from agents.presentation_agent import get_presentation_agent
                presentation_agent = await get_presentation_agent()
                if not presentation_agent:
                    print("ERROR: Failed to initialize presentation agent")
                    raise ValueError("Failed to initialize presentation agent")
                print("✓ Presentation agent initialized successfully")
                
                # Get the code task result from the context
                print("\nGetting code task result from context...")
                state = await context.get("state", {})
                completed_tasks = state.get("completed_tasks", [])
                
                # Find the most recent code task result
                code_task_result = None
                for completed_task in reversed(completed_tasks):
                    if completed_task.get("task_type") == TaskType.CODE.value:
                        code_task_result = completed_task
                        break
                
                # If no code task result in context, check if there are any in the current results
                if not code_task_result:
                    # Try to get the code task result from the orchestrate_tasks method
                    # This is a fallback and might not work if the method doesn't store results
                    print("No code task result found in context, checking current execution...")
                    
                    # Get the code task result from the docker_code_agent
                    from sandbox.docker_code_agent import handle_task
                    code_context = Context()
                    code_task_result = {
                        "task_type": TaskType.CODE.value,
                        "status": "success",
                        "result": await handle_task(code_context)
                    }
                
                if not code_task_result:
                    print("ERROR: No code task results found. Cannot create presentation without code.")
                    raise ValueError("No code task results found. Cannot create presentation without code.")
                
                code_result = code_task_result.get("result", {})
                if isinstance(code_result, str):
                    try:
                        code_result = json.loads(code_result)
                    except json.JSONDecodeError:
                        print("ERROR: Failed to parse code task result as JSON")
                        raise ValueError("Failed to parse code task result as JSON")
                
                # Extract GitHub repository information
                github_repo_url = code_result.get("github_repository")
                if not github_repo_url:
                    print("ERROR: GitHub repository URL not found in code task result")
                    raise ValueError("GitHub repository URL not found in code task result")
                print(f"✓ Found GitHub repository URL: {github_repo_url}")
                
                # Extract owner and repo name from the URL
                match = re.search(r"github\.com/([^/]+)/([^/]+)", github_repo_url)
                if not match:
                    print("ERROR: Failed to extract owner and repo name from GitHub URL")
                    raise ValueError(f"Invalid GitHub URL format: {github_repo_url}")
                
                owner = match.group(1)
                repo_name = match.group(2)
                print(f"✓ Extracted owner: {owner}, repo name: {repo_name}")
                
                # Fetch code and documentation content from GitHub
                print("\nFetching code and documentation content from GitHub...")
                
                # Use the presentation agent's GitHub tool to fetch content
                code_content = await presentation_agent.agent.achat(
                    f"Use the Github MCP get_file_content tool to get the contents of main.py from the repository '{repo_name}' owned by '{owner}'. Return ONLY the file content."
                )
                
                documentation_content = await presentation_agent.agent.achat(
                    f"Use the Github MCP get_file_content tool to get the contents of README.md from the repository '{repo_name}' owned by '{owner}'. Return ONLY the file content."
                )
                
                print("✓ Successfully fetched code and documentation content")
                
                # Create presentation with the fetched content
                print("\nExecuting presentation creation with content...")
                result = await presentation_agent._create_presentation_with_content(
                    code_content.response.strip(),
                    documentation_content.response.strip()
                )
                
                if "error" in result:
                    print(f"\nERROR: Presentation creation failed:")
                    print(f"- Error message: {result['error']}")
                    if "tokens_used" in result:
                        print(f"- Tokens used: {result['tokens_used']}")
                else:
                    print("\n✓ Presentation creation completed successfully")
                    if "presentation_id" in result:
                        print(f"- Presentation ID: {result['presentation_id']}")
                    if "tokens_used" in result:
                        print(f"- Total tokens used: {result['tokens_used']['total_tokens']}")
                
                print("=== Presentation Task Complete ===\n")
            else:
                # For other agents, use acomplete
                result = await agent.acomplete(context=context)
                
            return {
                "task_type": task["type"],
                "status": "success",
                "result": result
            }
        except Exception as e:
            print(f"\nERROR during task execution:")
            print(f"- Error type: {type(e).__name__}")
            print(f"- Error message: {str(e)}")
            if hasattr(e, '__traceback__'):
                print(f"- Error location: {e.__traceback__.tb_frame.f_code.co_filename}:{e.__traceback__.tb_lineno}")
            
            error_response = {
                "task_type": task["type"],
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
            print("\nReturning error response")
            return error_response
        finally:
            print(f"\n=== Task Execution Complete: {task['type']} ===\n")

    async def orchestrate_tasks(self, analysis: Dict, context: Context) -> List[Dict]:
        """Orchestrate execution of all tasks based on dependencies."""
        print("\n=== Starting Task Orchestration ===")
        tasks = analysis["required_tasks"]
        print(f"Total tasks to execute: {len(tasks)}")
        results = []
        completed_types = set()
        
        # Initialize completed_tasks in context if it doesn't exist
        state = await context.get("state", {})
        if "completed_tasks" not in state:
            state["completed_tasks"] = []
        await context.set("state", state)

        # Group tasks by type
        task_groups = {
            "code": [],
            "presentation": [],
            "voiceover": []
        }
        
        for task in tasks:
            if task["type"] in task_groups:
                task_groups[task["type"]].append(task)

        # Execute tasks in order: code -> presentation -> voiceover
        execution_order = ["code", "presentation", "voiceover"]
        
        for task_type in execution_order:
            if task_groups[task_type]:
                print(f"\nProcessing {task_type} tasks...")
                print(f"Found {len(task_groups[task_type])} tasks of type {task_type}")
                
                # Sort tasks within each type by priority
                task_groups[task_type].sort(key=lambda x: x["priority"])
                print(f"Tasks sorted by priority (lowest first)")
                
                if task_type == "code":
                    # For code tasks, combine all descriptions and execute once
                    combined_task = {
                        "type": "code",
                        "description": "\n".join(task["description"] for task in task_groups["code"]),
                        "priority": min(task["priority"] for task in task_groups["code"])
                    }
                    print("\nExecuting combined code tasks...")
                    result = await self.execute_task(combined_task, context)
                    results.append(result)
                    
                    # Store the result in the context for later tasks
                    state = await context.get("state", {})
                    state["completed_tasks"].append(result)
                    await context.set("state", state)
                    
                    print(f"✓ Code tasks completed with status: {result['status']}")
                else:
                    # For other types, execute each task separately
                    for i, task in enumerate(task_groups[task_type], 1):
                        print(f"\nExecuting {task_type} task {i}/{len(task_groups[task_type])}...")
                        print(f"Task priority: {task['priority']}")
                        result = await self.execute_task(task, context)
                        results.append(result)
                        
                        # Store the result in the context for later tasks
                        state = await context.get("state", {})
                        state["completed_tasks"].append(result)
                        await context.set("state", state)
                        
                        print(f"✓ Task completed with status: {result['status']}")
                        if "error" in result:
                            print(f"ERROR: {result['error']}")
                
                completed_types.add(task_type)
                print(f"✓ All {task_type} tasks completed")

        print("\n=== Task Orchestration Complete ===")
        print(f"Completed task types: {', '.join(completed_types)}")
        print(f"Total results: {len(results)}")
        return results

    def write_instructions(self, assignment_text: str, analysis: Dict = None) -> None:
        """Write instructions to sandbox/instructions file."""
        if analysis:
            # Write original assignment and creative idea to instructions file
            content = f"""ORIGINAL ASSIGNMENT:
{assignment_text}

CREATIVE IMPLEMENTATION IDEA:
{analysis['creative_idea']['summary']}

WHY THIS IS INTERESTING:
{analysis['creative_idea']['why_interesting']}
"""
        else:
            # Just write the assignment text
            content = assignment_text
            
        instructions_file = self.sandbox_dir / "instructions"
        instructions_file.write_text(content)

    async def process_assignment(self, assignment_text: str, result_dir: Path) -> Dict[str, str]:
        """
        Process the assignment using the agent workflow.
        Returns dict with results from each agent.
        """
        # Write instructions to sandbox
        self.write_instructions(assignment_text)

        try:
            # Create workflow and context
            workflow = AgentWorkflow(
                agents=[orchestrator_agent],
                root_agent=orchestrator_agent.name
            )
            context = Context(workflow=workflow)
            
            # Set context state
            await context.set("state", {
                "assignment_text": assignment_text,
                "result_dir": str(result_dir)
            })
            
            # Call orchestrator agent's handle_orchestration function
            response = await handle_orchestration(context)
            
            # Parse the response
            try:
                result = json.loads(response)
                if isinstance(result, dict) and "status" in result:
                    return result
            except json.JSONDecodeError:
                pass
            
            # If we couldn't parse JSON or it's not in the expected format,
            # wrap the response in a success result
            return {
                "status": "success",
                "message": "Code generation completed",
                "code": response
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def run(self, assignment_file: str):
        """Main orchestration logic."""
        try:
            # Read assignment
            assignment_text = self.read_assignment(assignment_file)
            
            # Create result directory
            assignment_name = Path(assignment_file).stem
            result_dir = self.create_result_directory(assignment_name)
            
            # Save original assignment
            (result_dir / "original_assignment.txt").write_text(assignment_text)
            
            # Process assignment with agent workflow
            results = await self.process_assignment(assignment_text, result_dir)
            
            # Save results
            if results.get("status") == "success":
                for key, value in results.items():
                    if isinstance(value, (str, dict)):
                        result_file = result_dir / f"{key}_result.txt"
                        result_file.write_text(
                            value if isinstance(value, str) else json.dumps(value, indent=2)
                        )
                
                # Print results summary
                print("\nAssignment Processing Complete!")
                print(f"Results saved in: {result_dir}")
                print("\nCode Generation Summary:")
                print("="*50)
                if "code" in results:
                    print(results["code"][:200] + "..." if len(results["code"]) > 200 else results["code"])
                if "test_results" in results:
                    print("\nTest Results:")
                    print(results["test_results"])
            else:
                print("\nAssignment Processing Failed!")
                print(f"Error details saved in: {result_dir}/error_result.txt")
                if "error" in results:
                    print(f"Error: {results['error']}")
                
        except Exception as e:
            print(f"Error processing assignment: {e}", file=sys.stderr)
            sys.exit(1)

async def handle_orchestration(ctx: Context) -> str:
    """
    Main entry point for the Orchestrator Agent.
    Analyzes assignment, coordinates tasks between specialized agents.
    """
    print("\n=== Starting Assignment Orchestration ===")
    try:
        print("Step 1: Getting assignment text from context...")
        state = await ctx.get("state")
        assignment_text = state.get("assignment_text")
        if not assignment_text:
            print("ERROR: No assignment text found in context")
            raise ValueError("No assignment text provided in context")
        print(f"✓ Assignment text retrieved ({len(assignment_text)} chars)")

        print("\nStep 2: Creating orchestrator instance...")
        orchestrator = OrchestratorAgent()
        print("✓ Orchestrator instance created")
        
        print("\nStep 3: Analyzing assignment...")
        analysis = await orchestrator.analyze_assignment(assignment_text)
        print("✓ Assignment analysis completed")
        print("Analysis contains:")
        print(f"- Main objective: {analysis.get('main_objective', 'Not found')}")
        print(f"- Creative idea: {analysis.get('creative_idea', {}).get('summary', 'Not found')}")
        print(f"- Required tasks: {len(analysis.get('required_tasks', []))} tasks")
        
        print("\nStep 4: Writing instructions to file...")
        orchestrator.write_instructions(assignment_text, analysis)
        print("✓ Instructions written to file")
        
        print("\nStep 5: Orchestrating tasks...")
        results = await orchestrator.orchestrate_tasks(analysis, ctx)
        print("✓ Task orchestration completed")
        print(f"Total tasks completed: {len(results)}")
        
        # Count successes and failures
        successes = sum(1 for r in results if r.get('status') == 'success')
        failures = sum(1 for r in results if r.get('status') == 'error')
        print(f"Task results: {successes} succeeded, {failures} failed")
        
        print("\nStep 6: Preparing final response...")
        response = {
            "status": "success",
            "analysis": analysis,
            "task_results": results
        }
        
        print("=== Assignment Orchestration Complete ===\n")
        return json.dumps(response, indent=2)
        
    except Exception as e:
        print("\nERROR during orchestration:")
        print(f"- Error type: {type(e).__name__}")
        print(f"- Error message: {str(e)}")
        if hasattr(e, '__traceback__'):
            print(f"- Error location: {e.__traceback__.tb_frame.f_code.co_filename}:{e.__traceback__.tb_lineno}")
        
        error_response = {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }
        print("\nReturning error response")
        print("=== Assignment Orchestration Failed ===\n")
        return json.dumps(error_response, indent=2)

# Create the Orchestrator Agent with LlamaIndex
orchestrator_agent = FunctionAgent(
    name="OrchestratorAgent",
    description="Analyzes assignments and coordinates specialized agents",
    system_prompt="""You are a creative system orchestrator that:
    1. Analyzes assignments to understand core requirements
    2. Proposes creative and interesting implementation ideas
    3. Provides clear direction to specialized agents
    4. Ensures the final solution meets assignment goals
    
    Your workflow:
    1. Read and understand the assignment requirements
    2. Come up with an original and engaging implementation idea
    3. Write clear instructions combining the original assignment and your creative idea
    4. Coordinate specialized agents to bring the idea to life
    5. Ensure quality and completeness of deliverables
    
    Always ensure:
    - Ideas are creative yet practical for the scope
    - Instructions are clear and actionable
    - Implementation satisfies assignment requirements
    - Work is properly coordinated between agents
    - Final solution is cohesive and well-integrated""",
    tools=[
        FunctionTool.from_defaults(fn=handle_orchestration, name="HandleOrchestration")
    ],
    llm=llm,
    verbose=True
)

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Homework System")
    parser.add_argument(
        "--assignment",
        required=True,
        help="Name of the assignment file in assignment_vault directory"
    )
    
    args = parser.parse_args()
    
    # Run orchestrator
    import asyncio
    orchestrator = OrchestratorAgent()
    asyncio.run(orchestrator.run(args.assignment))

if __name__ == "__main__":
    # For normal operation, run the main function
    main()
    
    # For testing the orchestrator directly (uncomment to use)
    # import asyncio
    # async def test_orchestrator():
    #     workflow = AgentWorkflow(agents=[orchestrator_agent], root_agent=orchestrator_agent.name)
    #     context = Context(workflow=workflow)
    #     await context.set("state", {
    #         "assignment_text": "Create a simple web application with documentation"
    #     })
    #     result = await handle_orchestration(context)
    #     print(result)
    # 
    # asyncio.run(test_orchestrator())