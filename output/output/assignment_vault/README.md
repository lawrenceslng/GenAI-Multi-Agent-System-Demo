# Chat Assistant Project

## Overview
This project implements a Chat Assistant using FastAPI for the backend and JavaScript for the frontend. The assistant mimics the functionality of ChatGPT, allowing users to interact with it in a conversational manner.

## Features
- Front-end JavaScript interface
- Back-end FastAPI server
- Continuous conversation feature
- Placeholder for LLM integration

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/robot-coder/ChatAssistant-Project.git
   cd ChatAssistant-Project
   ```

2. Install the required packages:
   ```bash
   pip install fastapi uvicorn
   ```

3. Run the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

4. Open `index.html` in your browser to interact with the Chat Assistant.

## Deployment
This Chat Assistant will be deployed on Render.com. A link to the deployed site will be provided in the README once available.

## Extensions
- Text file uploads to add to the prompt context.
- Image file uploads for multimodal LLMs.
- Side-by-side LLM response comparison of two models.

## License
This project is licensed under the MIT License.