
# Multi-Agent Homework System

  

A Python-based system that uses multiple specialized agents to process and complete homework assignments. The system breaks down assignments into subtasks and delegates them to specialized agents for code generation, documentation creation, and presentation preparation.

  

## Features

  

- Command-line interface for submitting assignments

- Multiple specialized agents:

- Code Agent: Generates Python code implementations

- Documentation Agent: Creates comprehensive documentation

- Presentation Agent: Prepares presentation outlines

- Safe code execution environment

- Organized result storage with timestamps

  

## Installation

  

1. Clone the repository:

```bash

git  clone <repository-url>

cd  multi-agent-homework

```

  

2. Create and activate a virtual environment:

```bash

python3.12  -m  venv  genai-mas-venv

source  genai-mas-venv/bin/activate  # On Windows: venv\Scripts\activate

```

  

3. Install dependencies:

```bash

pip  install  -r  requirements.txt

```

  

## Usage

  

1. Place your assignment description in a text file within the `assignment_vault` directory.

  

2. Run the orchestrator with your assignment file:

```bash

python  orchestrator.py  --assignment  your_assignment.txt

```

  

3. Find the results in the `result` directory, organized by assignment name and timestamp.

  

## Project Structure

  

```

multi_agent_homework/

│

├── orchestrator.py # Main coordination logic

├── agents/ # Specialized agents

│ ├── __init__.py

│ ├── code_agent.py # Generates code

│ ├── doc_agent.py # Creates documentation

│ └── presentation_agent.py # Prepares presentations

├── local_autogen/ # Directory for autogen

│ └── main.py # Main/current autogen entrypoint

│ └── test_autogen.py # Initial try/setup with autogen

├── sandbox/

│ └── runner.py # Safe code execution

├── utils/

│ └── container_manager.py # Docker container management

├── requirements.txt

└── README.md

```

  

## Agent Capabilities

  

### Code Agent

- Analyzes coding requirements

- Generates Python implementations

- Executes code in a sandboxed environment

- Provides test results

  

### Documentation Agent

- Generates comprehensive documentation

- Creates structured Markdown content

- Includes metadata and formatting

  

### Presentation Agent

- Creates presentation outlines

- Generates slide content and speaker notes

- Provides estimated duration and structure

  
## AutoGen
We also tried to use autogen and give the agents more autonomy. It worked to some extent, but we faced many issues (as you can see in the recording) like rate limiting and failed api formatting/calls. However, using autogen did seem to provide more creativity from the agents and it seemed like they were better able to resolve errors they faced by themselves and with the help of the group chat.

```
python local_autogen/main.py --assignment assignment_vault/project_1.txt
```
Demo:
https://github.com/user-attachments/assets/9eea8c87-ef5c-4a19-ab64-f8a35433a2f0


## Future Enhancements

  

1. Dynamic Agent Creation

- Create specialized agents based on task requirements

- Support for additional programming languages

  

2. Docker Integration

- Full container-based code isolation

- Secure execution environment

  

3. LlamaIndex Integration

- Enhanced natural language processing

- Improved task understanding and delegation

  

## Contributing

  

1. Fork the repository

2. Create a feature branch

3. Commit your changes

4. Push to the branch

5. Create a Pull Request

  

## License

  

This project is licensed under the MIT License - see the LICENSE file for details.
