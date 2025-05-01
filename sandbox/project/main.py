import asyncio
import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from llama_index import GPTSimpleVectorIndex, Document
from mcp_server_sdk import MCPClient
import spacy
from typing import List, Dict, Optional

# Load spaCy model for summarization
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If the model isn't installed, provide instructions or handle accordingly
    raise RuntimeError("spaCy model 'en_core_web_sm' not found. Please install it with: python -m spacy download en_core_web_sm")

class ContentScraper:
    """
    A class to scrape and filter web content.
    """

    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client

    async def fetch_page(self, url: str) -> Optional[str]:
        """
        Fetch the HTML content of a webpage asynchronously.
        """
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                await page.goto(url)
                content = await page.content()
                await browser.close()
                return content
        except Exception as e:
            print(f"Error fetching page {url}: {e}")
            return None

    def parse_html(self, html: str) -> str:
        """
        Parse HTML content to extract main text.
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            # Example: extract all paragraph texts
            paragraphs = soup.find_all('p')
            text = ' '.join(p.get_text() for p in paragraphs)
            return text
        except Exception as e:
            print(f"Error parsing HTML: {e}")
            return ""

    def filter_content(self, text: str, keywords: List[str]) -> bool:
        """
        Filter content based on presence of keywords.
        """
        return any(keyword.lower() in text.lower() for keyword in keywords)

class Summarizer:
    """
    A class to summarize text content.
    """

    def __init__(self):
        pass

    def summarize(self, text: str, max_sentences: int = 3) -> str:
        """
        Generate a simple extractive summary.
        """
        try:
            doc = nlp(text)
            sentences = list(doc.sents)
            # Select first N sentences as summary
            summary_sentences = sentences[:max_sentences]
            summary = ' '.join([sent.text for sent in summary_sentences])
            return summary
        except Exception as e:
            print(f"Error during summarization: {e}")
            return ""

class NewsletterAgent:
    """
    Main agent to orchestrate scraping, filtering, summarizing, and content curation.
    """

    def __init__(self, mcp_server_url: str, keywords: List[str]):
        self.mcp_client = MCPClient(server_url=mcp_server_url)
        self.scraper = ContentScraper(self.mcp_client)
        self.summarizer = Summarizer()
        self.keywords = keywords

    async def process_article(self, url: str) -> Optional[Dict]:
        """
        Fetch, filter, and summarize an article.
        """
        html = await self.scraper.fetch_page(url)
        if not html:
            return None
        text = self.scraper.parse_html(html)
        if not self.scraper.filter_content(text, self.keywords):
            return None
        summary = self.summarizer.summarize(text)
        return {
            'url': url,
            'summary': summary
        }

    async def curate_content(self, urls: List[str]) -> List[Dict]:
        """
        Process multiple URLs concurrently.
        """
        tasks = [self.process_article(url) for url in urls]
        results = await asyncio.gather(*tasks)
        # Filter out None results
        return [res for res in results if res]

    def build_index(self, articles: List[Dict]) -> GPTSimpleVectorIndex:
        """
        Build a vector index from summarized articles.
        """
        documents = [Document(text=article['summary'], extra_info={'url': article['url']}) for article in articles]
        index = GPTSimpleVectorIndex.from_documents(documents)
        return index

    def generate_newsletter(self, index: GPTSimpleVectorIndex) -> str:
        """
        Generate a newsletter summary from the index.
        """
        try:
            # Example: retrieve top 3 articles
            top_docs = index.query("Summarize the latest articles", top_k=3)
            newsletter_content = "\n\n".join([doc.text for doc in top_docs])
            return newsletter_content
        except Exception as e:
            print(f"Error generating newsletter: {e}")
            return ""

async def main():
    """
    Main execution function.
    """
    # Example configuration
    mcp_server_url = "http://localhost:8000"
    target_urls = [
        "https://example.com/article1",
        "https://example.com/article2",
        "https://example.com/article3"
    ]
    keywords = ["technology", "science", "innovation"]

    agent = NewsletterAgent(mcp_server_url, keywords)
    articles = await agent.curate_content(target_urls)
    if not articles:
        print("No articles matched the filtering criteria.")
        return
    index = agent.build_index(articles)
    newsletter = agent.generate_newsletter(index)
    print("Generated Newsletter:\n")
    print(newsletter)

if __name__ == "__main__":
    asyncio.run(main())