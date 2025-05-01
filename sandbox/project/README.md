# README.md

# Content Curation and Summarization Agent

This project develops an intelligent agent that leverages custom Python functions and MCP servers to automate web scraping, content filtering, browser interactions, and article summarization for creating a personalized newsletter.

## Features

- Scrapes articles from various sources using custom functions and MCP servers
- Filters and processes content based on user-defined criteria
- Automates browser interactions with Playwright for dynamic content
- Summarizes articles using NLP libraries (NLTK or spaCy)
- Modular and extensible code structure with error handling

## Requirements

Ensure you have Python 3.8+ installed. Install the required libraries:

```bash
pip install -r requirements.txt
```

## Files

- `main.py`: Main script orchestrating the agent's workflow
- `requirements.txt`: List of dependencies
- `README.md`: This documentation

## Usage

1. Configure your MCP server credentials and endpoints in `main.py`.
2. Customize filtering criteria and summarization parameters as needed.
3. Run the main script:

```bash
python main.py
```

## License

This project is for educational purposes. Modify and extend as needed.

---

# main.py

import asyncio
import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Browser, Page
from llama_index import GPTIndex  # Assuming llama_index is used for summarization
from mcp_server_sdk import MCPClient  # Placeholder for MCP SDK
import logging
from typing import List, Dict, Optional
import nltk
# If using spaCy for summarization, uncomment the following:
# import spacy

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize NLP tools
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# For spaCy, load model if used
# nlp = spacy.load('en_core_web_sm')

# Custom function to fetch articles from MCP server
def fetch_articles_from_mcp(mcp_client: MCPClient, query_params: Dict) -> List[Dict]:
    """
    Fetch articles from MCP server based on query parameters.

    Args:
        mcp_client (MCPClient): Initialized MCP client.
        query_params (Dict): Parameters for querying articles.

    Returns:
        List[Dict]: List of article metadata.
    """
    try:
        articles = mcp_client.query_articles(query_params)
        logger.info(f"Fetched {len(articles)} articles from MCP server.")
        return articles
    except Exception as e:
        logger.error(f"Error fetching articles from MCP: {e}")
        return []

# Custom function to scrape article content
def scrape_article_content(url: str) -> Optional[str]:
    """
    Scrape the main content of an article from its URL.

    Args:
        url (str): URL of the article.

    Returns:
        Optional[str]: Extracted article text or None if failed.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Example: extract all paragraph texts
        paragraphs = soup.find_all('p')
        content = ' '.join(p.get_text() for p in paragraphs)
        return content if content else None
    except requests.RequestException as e:
        logger.error(f"Request error for {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return None

# Custom function to automate browser interactions
async def automate_browser_interaction(url: str) -> Optional[str]:
    """
    Use Playwright to open a page and perform interactions if needed.

    Args:
        url (str): URL to open.

    Returns:
        Optional[str]: Page content after interactions or None if failed.
    """
    try:
        async with async_playwright() as p:
            browser: Browser = await p.chromium.launch(headless=True)
            page: Page = await browser.new_page()
            await page.goto(url, timeout=15000)
            # Add any interactions here if needed
            content = await page.content()
            await browser.close()
            return content
    except Exception as e:
        logger.error(f"Browser automation failed for {url}: {e}")
        return None

# Custom function to filter articles
def filter_articles(articles: List[Dict], keywords: List[str]) -> List[Dict]:
    """
    Filter articles based on presence of keywords in title or content.

    Args:
        articles (List[Dict]): List of article metadata.
        keywords (List[str]): List of keywords to filter by.

    Returns:
        List[Dict]: Filtered list of articles.
    """
    filtered = []
    for article in articles:
        title = article.get('title', '').lower()
        url = article.get('url', '')
        if any(keyword.lower() in title for keyword in keywords):
            filtered.append(article)
        else:
            # Optionally, scrape content and check
            content = scrape_article_content(url)
            if content and any(keyword.lower() in content.lower() for keyword in keywords):
                filtered.append(article)
    logger.info(f"Filtered down to {len(filtered)} articles.")
    return filtered

# Custom function to summarize article content
def summarize_text(text: str, max_sentences: int = 3) -> str:
    """
    Summarize text using NLP libraries.

    Args:
        text (str): Text to summarize.
        max_sentences (int): Max number of sentences in summary.

    Returns:
        str: Summarized text.
    """
    try:
        # Example using nltk
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        summary = ' '.join(sentences[:max_sentences])
        return summary
        # If using spaCy, implement accordingly
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        return text[:200] + '...'  # fallback

# Main orchestration function
async def main():
    """
    Main function to run the content curation and summarization pipeline.
    """
    # Initialize MCP client
    mcp_client = MCPClient(api_key='YOUR_MCP_API_KEY')  # Replace with actual credentials

    # Define query parameters for fetching articles
    query_params = {
        'category': 'technology',
        'limit': 10
    }

    # Fetch articles
    articles = fetch_articles_from_mcp(mcp_client, query_params)

    # Filter articles based on keywords
    keywords = ['AI', 'machine learning', 'automation']
    filtered_articles = filter_articles(articles, keywords)

    # Process each article
    newsletter_content = []
    for article in filtered_articles:
        url = article.get('url')
        title = article.get('title', 'No Title')
        logger.info(f"Processing article: {title}")

        # Scrape content
        content = scrape_article_content(url)
        if not content:
            # Fallback to browser automation if needed
            content = await automate_browser_interaction(url)
        if not content:
            logger.warning(f"Skipping article due to content fetch failure: {title}")
            continue

        # Summarize content
        summary = summarize_text(content)
        newsletter_content.append(f"Title: {title}\nSummary: {summary}\nURL: {url}\n")

    # Compile newsletter
    newsletter = "\n---\n".join(newsletter_content)
    print("Generated Newsletter:\n")
    print(newsletter)

if __name__ == '__main__':
    asyncio.run(main())

# requirements.txt

llama_index
mcp_server_sdk
playwright
beautifulsoup4
requests
nltk
# Uncomment if using spaCy for summarization
# spacy
# en_core_web_sm

# Note: For Playwright, install browsers
# playwright install

