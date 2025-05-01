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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from llama_index.llms.openai import OpenAI
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool

# Import agents
from agents.orchestrator_agent import handle_orchestration, orchestrator_agent

llm = OpenAI(model="gpt-4.1-nano-2025-04-14", api_key=os.getenv("OPENAI_API_KEY"))
# Initialize LLM
# llm = OpenRouter(
#     api_key=os.getenv("OPENROUTER_API_KEY"),
#     max_tokens=256,
#     context_window=4096,
#     model="google/gemini-2.0-flash-exp:free",
# )

class AssignmentOrchestrator:
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

    def write_instructions(self, content: str) -> None:
        """Write instructions to sandbox/instructions file."""
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
            
            # Call code agent's handle_task function
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
    orchestrator = AssignmentOrchestrator()
    asyncio.run(orchestrator.run(args.assignment))

if __name__ == "__main__":
    main()