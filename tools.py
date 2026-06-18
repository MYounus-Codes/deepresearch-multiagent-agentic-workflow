from langchain.tools import tool
from tavily import TavilyClient
from bs4 import BeautifulSoup
import requests
import os
from dotenv import load_dotenv
load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool()
def search_web(query: str) -> str:
    """Search the web for the given query and return the results. RETURN URL, TITLE, AND SNIPPET"""
    response = tavily.search(query=query, max_results=5)

    out = []

    for r in response["results"]:
        out.append(
            f"Title: {r['title']}\n"
            f"URL: {r['url']}\n"
            f"Snippet   : {r['content'][:400]}\n"
        )
    
    return "\n---\n".join(out)

@tool()
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:3000]
    except Exception as e:
        return f"Could not scrape URL: {str(e)}"
    
