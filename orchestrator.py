#!/usr/bin/env python3
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

class AssignmentOrchestrator:
    def __init__(self):
        self.assignment_vault = Path("assignment_vault")
        self.result_dir = Path("result")
        self.agents_dir = Path("agents")

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

    def parse_tasks(self, assignment_text: str) -> Dict[str, str]:
        """Split assignment into tasks for different agents."""
        return {
            "code": f"Generate Python code for: {assignment_text}",
            "documentation": f"Generate documentation for: {assignment_text}",
            "presentation": f"Generate presentation outline for: {assignment_text}"
        }

    def delegate_tasks(self, tasks: Dict[str, str], result_dir: Path) -> Dict[str, str]:
        """Delegate tasks to appropriate agents and collect results."""
        results = {}
        
        # Import agents dynamically
        for agent_type, task in tasks.items():
            try:
                agent_module = __import__(f"agents.{agent_type}_agent", fromlist=['handle_task'])
                result = agent_module.handle_task(task)
                results[agent_type] = result
                
                # Save result to file
                result_file = result_dir / f"{agent_type}_result.txt"
                result_file.write_text(result)
                
            except ImportError as e:
                print(f"Warning: Could not import {agent_type}_agent: {e}")
                results[agent_type] = f"Error: Agent {agent_type} not available"
                
        return results

    def run(self, assignment_file: str):
        """Main orchestration logic."""
        try:
            # Read assignment
            assignment_text = self.read_assignment(assignment_file)
            
            # Create result directory
            assignment_name = Path(assignment_file).stem
            result_dir = self.create_result_directory(assignment_name)
            
            # Parse into tasks
            tasks = self.parse_tasks(assignment_text)
            
            # Save original assignment
            (result_dir / "original_assignment.txt").write_text(assignment_text)
            
            # Delegate tasks and get results
            results = self.delegate_tasks(tasks, result_dir)
            
            # Print results summary
            print("\nAssignment Processing Complete!")
            print(f"Results saved in: {result_dir}")
            for agent_type, result in results.items():
                print(f"\n{agent_type.upper()} Agent Result Summary:")
                print(f"{'='*50}")
                print(result[:200] + "..." if len(result) > 200 else result)
                
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
    
    orchestrator = AssignmentOrchestrator()
    orchestrator.run(args.assignment)

if __name__ == "__main__":
    main()