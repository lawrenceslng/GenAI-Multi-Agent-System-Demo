# README.md

# Web Content Retrieval and Analysis Agent

This project implements an intelligent agent that leverages LlamaIndex to integrate custom Python functions for web content retrieval, key point extraction, and visualization. The agent interacts with an MCP server hosting web automation tools to perform research tasks and generate comprehensive reports.

## Features

- Fetch web content dynamically using Playwright
- Extract key points from web pages
- Visualize data insights with Matplotlib
- Communicate with MCP server for web automation
- Modular and extensible design

## Requirements

Ensure you have Python 3.8+ installed. Install the required libraries:

```bash
pip install -r requirements.txt
```

## Files

- `main.py`: Main script to run the agent
- `requirements.txt`: List of dependencies
- `README.md`: This documentation

## Usage

```bash
python main.py
```

## License

This project is licensed under the MIT License.

---

# main.py

import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from playwright.sync_api import sync_playwright
from llama_index import GPTIndex, ServiceContext
from mcp_client import MCPClient
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)

def fetch_web_content(url: str) -> str:
    """
    Fetches the HTML content of a web page using Playwright.
    Args:
        url (str): The URL of the web page to fetch.
    Returns:
        str: The HTML content of the page.
    Raises:
        RuntimeError: If fetching fails.
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url)
            content = page.content()
            browser.close()
        return content
    except Exception as e:
        logging.error(f"Error fetching web content from {url}: {e}")
        raise RuntimeError(f"Failed to fetch web content: {e}")

def extract_key_points(html_content: str) -> List[str]:
    """
    Extracts key points from HTML content.
    Args:
        html_content (str): HTML content of a web page.
    Returns:
        List[str]: List of key points extracted.
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        paragraphs = soup.find_all('p')
        key_points = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
        # For simplicity, return first 5 key points
        return key_points[:5]
    except Exception as e:
        logging.error(f"Error extracting key points: {e}")
        return []

def visualize_key_points(key_points: List[str]) -> None:
    """
    Creates a bar chart visualization of key points.
    Args:
        key_points (List[str]): List of key points.
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(key_points)), [len(kp) for kp in key_points], color='skyblue')
        plt.yticks(range(len(key_points)), key_points)
        plt.xlabel('Length of Key Point (characters)')
        plt.title('Key Points from Web Content')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Error during visualization: {e}")

def interact_with_mcp_server(command: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """
    Sends a command to the MCP server and retrieves the response.
    Args:
        command (str): The command to execute.
        params (Optional[Dict[str, Any]]): Parameters for the command.
    Returns:
        Any: Response from the MCP server.
    """
    try:
        client = MCPClient()
        response = client.send_command(command, params or {})
        return response
    except Exception as e:
        logging.error(f"Error communicating with MCP server: {e}")
        return None

def main() -> None:
    """
    Main function to orchestrate web retrieval, processing, and visualization.
    """
    url = "https://example.com/research-topic"
    try:
        html_content = fetch_web_content(url)
        key_points = extract_key_points(html_content)
        visualize_key_points(key_points)
        report = {
            "url": url,
            "key_points": key_points
        }
        # Send report to MCP server for further processing
        response = interact_with_mcp_server("generate_report", report)
        if response:
            logging.info("Report successfully generated and sent to MCP server.")
        else:
            logging.warning("Failed to generate report on MCP server.")
    except Exception as e:
        logging.error(f"An error occurred in main: {e}")

if __name__ == "__main__":
    main()

# requirements.txt

llama_index
requests
beautifulsoup4
matplotlib
playwright
mcp_client

# End of README.md