# Multi-Agent System Presentation

## Objective

Create a multi-agent system Proof-of-Concept (POC) that can:
- take in a coding-related task description
- break down the task into sub-tasks
- come up with creative implementation/solution
- delegate sub-tasks to appropriate agents
    - perform coding in a sandboxed environment
    - create a presentation to explain the code/solution
    - create a voiceover to accompany the presentation
- package deliverables 

We chose to use coding project assignments to demonstrate this POC because:
- assignment descriptions are available 

## Development Process

### VERSION 1

First, we focus on developing the individual agents:

1. Coding Agent                                                 
    - Launch Docker env       
    - Write code based on project description
    - Upload to GitHub repository via [Github MCP Server](https://github.com/github/github-mcp-server)

2. Presentation Agent 
    - Read code from GitHub repository
    - Generate presentation and slides via [Google Slides MCP Server](https://github.com/matteoantoci/google-slides-mcp)

3. Voiceover Agent
    - Read code & slides
    - Generate script
    - Generate audio file via [ElevenLabs MCP Server](https://github.com/elevenlabs/elevenlabs-mcp)

We then follow the [LlamaIndex Multi-Agent example](https://docs.llamaindex.ai/en/stable/understanding/agent/multi_agent/) to combine the agents together in a workflow.

This results in **VERSION 1** of the POC.

#### What We Found

- rigid architecture
    - flow is too structured

### VERSION 2




## Challenges and Lessons Learned
1. Unofficial MCP servers may not work as well; we could edit the MCP server itself to make this work
    - Google Slides MCP Server unable to create Title Slide
    ![MCP Server unable to create slides](./assets/image.png)
    ![PREDEFINED_LAYOUT value incorrect](./assets/image-1.png)

2. What is the value of multi-agent system at this scale?
    - is it better to have 1 agent with a variety of function-calling/tool-use capabilities or separate into different agents?


3. [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)

## Next Steps