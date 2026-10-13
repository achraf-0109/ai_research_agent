from langchain.tools import tool
from tavily import TavilyClient
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
from rich import print
import os

import time

# load env variables
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# make the tools

## search tool

def web_search(query: str) -> str:
    """Return top sources that speak about a topic"""
    client = TavilyClient(api_key=TAVILY_API_KEY)
    results = client.search(query=query,
                            max_results=2)

    urls = ""
    for result in results['results']:
        urls += result['url'] + "\n"

    return urls

## scraping tool
def scrape_url(url: str) -> str:
    """Fetch a web page and return its main text content."""
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        return f"Could not fetch {url}: {e}"

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove parts that are not real content
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    
    return f"Content: {text[:3000]}"