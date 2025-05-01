# main.py

import asyncio
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from llama_index import GPTIndex, ServiceContext, LLMPredictor
from mcp_client import MCPClient
from playwright.async_api import async_playwright
from typing import List, Dict, Any, Optional

class WebContentRetriever:
    """Class to retrieve web content using requests and Playwright."""
    @staticmethod
    def fetch_page_content(url: str) -> Optional[str]:
        """Fetches page content using requests."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    @staticmethod
    async def fetch_dynamic_content(url: str) -> Optional[str]:
        """Fetches dynamic page content using Playwright."""
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                await page.goto(url, timeout=10000)
                content = await page.content()
                await browser.close()
                return content
        except Exception as e:
            print(f"Error fetching dynamic content from {url}: {e}")
            return None

class KeyPointExtractor:
    """Class to extract key points from web content."""
    @staticmethod
    def extract_key_points(html_content: str, max_points: int = 5) -> List[str]:
        """Extracts key points from HTML content."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text()
            sentences = text.split('.')
            key_points = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 20]
            return key_points[:max_points]
        except Exception as e:
            print(f"Error extracting key points: {e}")
            return []

class Visualizer:
    """Class to visualize data."""
    @staticmethod
    def plot_key_points(key_points: List[str]) -> None:
        """Plots key points as a bar chart."""
        try:
            labels = [f"Point {i+1}" for i in range(len(key_points))]
            values = [len(point) for point in key_points]
            plt.figure(figsize=(8, 4))
            plt.bar(labels, values)
            plt.xlabel('Key Points')
            plt.ylabel('Text Length')
            plt.title('Key Point Lengths')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error during visualization: {e}")

class ResearchAgent:
    """Main agent class to coordinate web retrieval, processing, and reporting."""
    def __init__(self, mcp_server_url: str):
        self.mcp_client = MCPClient(mcp_server_url)
        self.service_context = self._create_service_context()

    def _create_service_context(self) -> ServiceContext:
        """Creates a llama_index ServiceContext with a simple predictor."""
        predictor = LLMPredictor()
        return ServiceContext.from_defaults(llm_predictor=predictor)

    def perform_research(self, url: str) -> Dict[str, Any]:
        """Performs research on a given URL."""
        content = WebContentRetriever.fetch_page_content(url)
        if not content:
            print(f"Failed to retrieve content from {url}")
            return {}

        key_points = KeyPointExtractor.extract_key_points(content)
        return {
            'url': url,
            'key_points': key_points,
            'content': content
        }

    def generate_report(self, research_data: Dict[str, Any]) -> None:
        """Generates a report based on research data."""
        url = research_data.get('url')
        key_points = research_data.get('key_points', [])
        print(f"Research Report for {url}")
        for idx, point in enumerate(key_points, 1):
            print(f"{idx}. {point}")

        # Visualize key points
        Visualizer.plot_key_points(key_points)

    def interact_with_mcp(self, command: str) -> Any:
        """Send command to MCP server and get response."""
        try:
            response = self.mcp_client.send_command(command)
            return response
        except Exception as e:
            print(f"Error communicating with MCP server: {e}")
            return None

async def main():
    """Main execution function."""
    mcp_server_url = "http://localhost:8000"  # Replace with actual MCP server URL
    agent = ResearchAgent(mcp_server_url)

    # Example URL to research
    url = "https://example.com"

    # Perform web content retrieval and processing
    research_data = agent.perform_research(url)

    if research_data:
        # Generate report
        agent.generate_report(research_data)

        # Example interaction with MCP server
        command = "perform_web_automation for research report"
        response = agent.interact_with_mcp(command)
        print(f"MCP Response: {response}")

if __name__ == "__main__":
    asyncio.run(main())