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
from agents.presentation_agent import presentation_agent
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
        try:
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
            
            return json.loads(analysis.text)
            
        except Exception as e:
            raise Exception(f"Error analyzing assignment: {str(e)}")

    async def get_agent_for_task(self, task_type: str):
        """Get the appropriate agent for a given task type."""
        if task_type == TaskType.CODE.value:
            return await get_code_agent()
            
        agents = {
            TaskType.DOCUMENTATION.value: documentation_agent,
            TaskType.PRESENTATION.value: presentation_agent,
            TaskType.VOICEOVER.value: voiceover_agent
        }
        return agents.get(task_type)

    async def execute_task(self, task: Dict, context: Context) -> Dict:
        """Execute a single task using the appropriate agent."""
        agent = await self.get_agent_for_task(task["type"])
        if not agent:
            raise ValueError(f"No agent found for task type: {task['type']}")

        # Update context with task-specific information
        await context.set("task", task)
        
        try:
            # Execute task with the specialized agent
            if task["type"] == TaskType.CODE.value:
                # For code agent, use handle_task
                from sandbox.docker_code_agent import handle_task
                result = await handle_task(context)
            else:
                # For other agents, use acomplete
                result = await agent.acomplete(context=context)
                
            return {
                "task_type": task["type"],
                "status": "success",
                "result": result
            }
        except Exception as e:
            return {
                "task_type": task["type"],
                "status": "error",
                "error": str(e)
            }

    async def orchestrate_tasks(self, analysis: Dict, context: Context) -> List[Dict]:
        """Orchestrate execution of all tasks based on dependencies."""
        tasks = analysis["required_tasks"]
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
                # Sort tasks within each type by priority
                task_groups[task_type].sort(key=lambda x: x["priority"])
                
                if task_type == "code":
                    # For code tasks, combine all descriptions and execute once
                    combined_task = {
                        "type": "code",
                        "description": "\n".join(task["description"] for task in task_groups["code"]),
                        "priority": min(task["priority"] for task in task_groups["code"])
                    }
                    result = await self.execute_task(combined_task, context)
                    results.append(result)
                else:
                    # For other types, execute each task separately
                    for task in task_groups[task_type]:
                        result = await self.execute_task(task, context)
                        results.append(result)
                
                completed_types.add(task_type)

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
    try:
        # Get assignment text from context
        state = await ctx.get("state")
        assignment_text = state.get("assignment_text")
        if not assignment_text:
            raise ValueError("No assignment text provided in context")

        # Create orchestrator instance
        orchestrator = OrchestratorAgent()
        
        # Analyze assignment and get creative idea
        analysis = await orchestrator.analyze_assignment(assignment_text)
        
        # Write original assignment and creative idea to instructions
        orchestrator.write_instructions(assignment_text, analysis)
        
        # Orchestrate tasks
        results = await orchestrator.orchestrate_tasks(analysis, ctx)
        
        # Prepare final response
        response = {
            "status": "success",
            "analysis": analysis,
            "task_results": results
        }
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        error_response = {
            "status": "error",
            "error": str(e)
        }
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