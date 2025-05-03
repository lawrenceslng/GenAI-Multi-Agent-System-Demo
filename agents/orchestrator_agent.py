#!/usr/bin/env python3
"""
Orchestrator Agent for Multi-Agent Homework System
Responsible for task analysis, decomposition, and coordination between specialized agents
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from enum import Enum

from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import Context

# Import specialized agents
from sandbox.docker_code_agent import get_code_agent
from agents.documentation_agent import documentation_agent
from agents.presentation_agent import get_presentation_agent
from agents.voiceover_agent import voiceover_agent

# Initialize LLM
llm = OpenAI(model="gpt-4.1-nano-2025-04-14", api_key=os.getenv("OPENAI_API_KEY"))

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
            from agents.voiceover_agent import mcp_tool
            return mcp_tool
        elif task_type == TaskType.PRESENTATION.value:
            return await get_presentation_agent()
        agents = {
            TaskType.DOCUMENTATION.value: documentation_agent,
            TaskType.VOICEOVER.value: voiceover_agent
        }
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
        print("✓ Context updated")
        
        try:
            print("\nExecuting task with specialized agent...")
            if task["type"] == TaskType.CODE.value:
                # For code agent, use handle_task
                from sandbox.docker_code_agent import handle_task
                result = await handle_task(context)
            elif task["type"] == TaskType.VOICEOVER.value:
                # For voiceover agent, use generate_and_save_audio
                from agents.voiceover_agent import get_agent, generate_and_save_audio
                voiceover_agent = await get_agent(agent)  # agent here is the mcp_tool from voiceover_agent
                agent_context = Context(voiceover_agent)
                result = await generate_and_save_audio(task["description"], voiceover_agent, agent_context)
            elif task["type"] == TaskType.PRESENTATION.value:
                print("\n=== Starting Presentation Task ===")
                print("Initializing presentation agent...")
                
                # For presentation agent, use create_presentation
                from agents.presentation_agent import get_presentation_agent
                presentation_agent = await get_presentation_agent()
                if not presentation_agent:
                    print("ERROR: Failed to initialize presentation agent")
                    raise ValueError("Failed to initialize presentation agent")
                print("✓ Presentation agent initialized successfully")
                
                agent_context = Context(presentation_agent)
                print("\nExecuting presentation creation...")
                result = await presentation_agent.create_presentation()
                
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

        # Group tasks by type
        task_groups = {
            "code": [],
            "documentation": [],
            "presentation": [],
            "voiceover": []
        }
        
        for task in tasks:
            if task["type"] in task_groups:
                task_groups[task["type"]].append(task)

        # Execute tasks in order: code -> documentation -> presentation -> voiceover
        execution_order = ["code", "documentation", "presentation", "voiceover"]
        
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
                    print(f"✓ Code tasks completed with status: {result['status']}")
                else:
                    # For other types, execute each task separately
                    for i, task in enumerate(task_groups[task_type], 1):
                        print(f"\nExecuting {task_type} task {i}/{len(task_groups[task_type])}...")
                        print(f"Task priority: {task['priority']}")
                        result = await self.execute_task(task, context)
                        results.append(result)
                        print(f"✓ Task completed with status: {result['status']}")
                        if "error" in result:
                            print(f"ERROR: {result['error']}")
                
                completed_types.add(task_type)
                print(f"✓ All {task_type} tasks completed")

        print("\n=== Task Orchestration Complete ===")
        print(f"Completed task types: {', '.join(completed_types)}")
        print(f"Total results: {len(results)}")
        return results

    def write_instructions(self, assignment_text: str, analysis: Dict) -> None:
        """Write original assignment and creative idea to instructions file."""
        content = f"""ORIGINAL ASSIGNMENT:
{assignment_text}

CREATIVE IMPLEMENTATION IDEA:
{analysis['creative_idea']['summary']}

WHY THIS IS INTERESTING:
{analysis['creative_idea']['why_interesting']}
"""
        instructions_file = self.sandbox_dir / "instructions"
        instructions_file.write_text(content)

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

if __name__ == "__main__":
    # For testing the orchestrator directly
    import asyncio
    
    async def test_orchestrator():
        workflow = AgentWorkflow(agents=[orchestrator_agent], root_agent=orchestrator_agent.name)
        context = Context(workflow=workflow)
        await context.set("state", {
            "assignment_text": "Create a simple web application with documentation"
        })
        result = await handle_orchestration(context)
        print(result)
    
    asyncio.run(test_orchestrator())