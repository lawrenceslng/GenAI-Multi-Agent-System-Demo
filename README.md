# Multi-Agent Homework System

This repository showcases a Python-based multi-agent system that uses multiple specialized agents to process and complete homework assignments. The system breaks down assignments into subtasks and delegates them to specialized agents for code generation, Google Slides presentation generation, and voiceover audio generation for presentation via ElevenLabs.

Two versions of the multi-agent system is built. **Version 1** is built following [LlamaIndex's multi-agent example](https://docs.llamaindex.ai/en/stable/understanding/agent/multi_agent/). A strict workflow is followed with each agent taking turns performing their individual tasks. The downside of this strict workflow is the lack of user interaction and lack of deviation/autonomy for the agents.

**Version 2** is built using [Microsoft's Autogen framework](https://github.com/microsoft/autogen). The goal is to allow the agents more autonomy. 

Both versions uses the following OpenAI LLMs:
- OpenAI 4o
- OpenAI 4.1-nano
- OpenAI 4o-mini

This project serves as a proof-of-concept that a multi-agent system can take a coding-related task and break it down into subtasks to create a full presentation with slide show and voiceover. While we used homework assignments as the basis of this POC, the potential of such a multi-agent system can help in the automatic creation of sample integration walkthroughs or other small MVP projects in industry roles such as solution architect or implementation engineer. The ability to generate not only the code, but a presentation and voiceover audio is a powerful combination when it comes to conveying information to new audiences and presents new opportunities for knowledge sharing and collaboration. 

## Features

- Command-line interface for initializing the multi-agent system
- Multiple specialized agents:
  - Docker Code Agent: Generates Python code implementations within a Docker environment for full isolation of the coding agent; uploads completed code to Github via [Github MCP Server](https://github.com/github/github-mcp-server)
  - Documentation Agent (optional): Creates comprehensive documentation
  - Presentation Agent: Generates Google Slides presentation via [Google Slides MCP Server](https://github.com/matteoantoci/google-slides-mcp)
  - Voiceover Agent: Generates script and audio file for presentation via [ElevenLabs MCP Server](https://github.com/elevenlabs/elevenlabs-mcp)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository>
```

2. Create and activate a virtual environment:
```bash
python3.12 -m venv genai-mas-venv
source genai-mas-venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Version 1 Usage

1. Place your assignment description in a text file within the `assignment_vault` directory.

2. Run the orchestrator with your assignment file:
```bash
python orchestrator_merged.py --assignment <assignment file>
```

3. There are 3 outputs that are generated:
   - [a repository under the Github user robot-coder](https://github.com/robot-coder)
   - a Google Slides Presentation in robot-coder's [Google Drive Account](https://drive.google.com/drive/my-drive)
   - a voiceover mp3 file in `output/audio`

## Version 2 Usage

1. Place your assignment description in a text file within the `assignment_vault` directory.

2. Run the Autogen execution file with your assignment file:
```bash
python local_autogen/main.py --assignment assignment_vault/<assignment file>
```



## Project Structure

```
GenAI-Multi-Agent-System-Demo/
├── assignment_vault/               # directory for assignment text files
├── orchestrator_merged.py          # Main coordination logic
├── agents/                         # Specialized agents
│   ├── __init__.py
│   ├── documentation_agent.py      # Creates documentation
│   ├── presentation_agent.py       # Prepares presentations
│   └── voiceover_agent.py          # Generates script and audio file
├── sandbox/                        # Directory is mapped as volume into Docker environment
│   ├── docker_code_agent.py        # coding agent
│   ├── Dockerfile                  # Docker file for Docker environment
│   ├── instructions                # Assignment instructions are copied here
│   └── run_docker_agent.sh         # Runs the coding agent in a docker container
├── output/audio                    # Directory where audio files are generated
├── requirements.txt
├── PRESENTATION.md
├── .env.example
└── README.md
```

## Agent Capabilities

### Code Agent
- Analyzes coding requirements
- Generates and executes Python code in a sandboxed environment
- Uploads completed code to Github repository

### Documentation Agent
- Generates comprehensive documentation for a given repository
- Creates structured Markdown content
- Includes metadata and formatting

### Presentation Agent
- Creates Google Slides presentation
- Generates slide content and speaker notes
- Provides estimated duration and structure

### Voiceover Agent
- Creates audio file for presentation and code using ElevenLabs 
- Outputs audio file locally 

## Future Enhancements

1. Dynamic Agent Creation
   - Create specialized agents based on task requirements
   - Support for additional programming languages

2. Enable more user interaction
   - Implement chat-based behavior prior to initiating workflows so user can further customize and define behavior

3. Presentation and Voiceover integration
   - Enable automated presentation with voiceover via Playwright or alternative solutions

4. Interactive Voice Agentic AI for Presentation for real-time user-agent interaction 
   - Enable users to interact with the Voice Agent to ask questions in real-time and engage in conversation about produce output.

## Demo 

The following link shows how **Version 1** of this multi-agent system works. The command `python orchestrator_merged.py --assignment project_3.txt` is run, and it kicks off all the necessary agents to complete the task one by one in order. The Github Repository, Google Slides (minus Title Slide), and the voiceover recording are all generated automatically.

[Version 1 Demo Video](https://youtu.be/U3R9durFHp4)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
