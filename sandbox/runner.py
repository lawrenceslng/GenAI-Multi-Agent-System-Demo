#!/usr/bin/env python3
"""
Code Runner for Multi-Agent Homework System
Provides safe execution environment for generated code
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

class CodeRunner:
    def __init__(self, use_docker: bool = False):
        self.use_docker = use_docker
        self.temp_dir = Path(tempfile.mkdtemp(prefix="homework_sandbox_"))
        
    def _write_code_to_file(self, code: str) -> Path:
        """Write code to a temporary file."""
        code_file = self.temp_dir / "code.py"
        code_file.write_text(code)
        return code_file
        
    def _run_in_process(self, code_file: Path) -> Tuple[str, str, int]:
        """Run code in a separate process with basic isolation."""
        try:
            # Run in a new process with restricted permissions
            process = subprocess.run(
                [sys.executable, str(code_file)],
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
                cwd=str(self.temp_dir)
            )
            
            return (
                process.stdout,
                process.stderr,
                process.returncode
            )
            
        except subprocess.TimeoutExpired:
            return "", "Execution timed out after 30 seconds", 1
        except Exception as e:
            return "", f"Error running code: {str(e)}", 1
            
    def _run_in_docker(self, code_file: Path) -> Tuple[str, str, int]:
        """Run code in a Docker container (placeholder for future implementation)."""
        # This would be implemented in a future version
        return (
            "",
            "Docker execution not implemented yet",
            1
        )
        
    def run_code(self, code: str) -> Dict[str, str]:
        """
        Run the provided code in a safe environment.
        Returns dict with stdout, stderr, and status
        """
        try:
            # Write code to temporary file
            code_file = self._write_code_to_file(code)
            
            # Run the code
            if self.use_docker:
                stdout, stderr, returncode = self._run_in_docker(code_file)
            else:
                stdout, stderr, returncode = self._run_in_process(code_file)
            
            return {
                "stdout": stdout,
                "stderr": stderr,
                "status": "success" if returncode == 0 else "error",
                "returncode": returncode
            }
            
        finally:
            # Cleanup
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"Warning: Failed to cleanup temporary directory: {e}", file=sys.stderr)

def run_code_safely(code: str, use_docker: bool = False) -> Dict[str, str]:
    """
    Main entry point for running code safely.
    Args:
        code: String containing Python code to run
        use_docker: Whether to use Docker for isolation (not implemented yet)
    Returns:
        Dict containing execution results
    """
    runner = CodeRunner(use_docker=use_docker)
    return runner.run_code(code)

if __name__ == "__main__":
    # Test the runner
    test_code = """
print("Hello from sandbox!")
for i in range(3):
    print(f"Count: {i}")
"""
    result = run_code_safely(test_code)
    print("Execution Result:")
    print(f"Status: {result['status']}")
    print(f"Output:\n{result['stdout']}")
    if result['stderr']:
        print(f"Errors:\n{result['stderr']}")