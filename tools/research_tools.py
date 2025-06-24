# tools/research_tools.py

from crewai_tools import SerperDevTool
from crewai_tools.tools.base_tool import BaseTool 
from playwright.sync_api import sync_playwright
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')


class RedditResearchTool(BaseTool):
    name: str = "RedditResearchTool"
    description: str = "Useful to search for real-world opinions, trends, and discussions on Reddit given a topic."

    def _run(self, query: str) -> str:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(f"https://www.reddit.com/search/?q={query}")
            page.wait_for_timeout(3000)
            posts = page.locator("div[data-testid='post-container'] h3")
            titles = posts.all_text_contents()
            browser.close()
            return "\n".join(titles[:5]) if titles else "No Reddit results found."


class QuoraResearchTool(BaseTool):
    name: str = "QuoraResearchTool"
    description: str = "Helpful for finding expert-style answers and discussions from Quora for any topic."

    def _run(self, query: str) -> str:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(f"https://www.quora.com/search?q={query}")
            page.wait_for_timeout(3000)
            answers = page.locator("div.q-box.qu-mb--tiny")
            snippets = answers.all_text_contents()
            browser.close()
            return "\n".join(snippets[:5]) if snippets else "No Quora results found."
    def myrun(self):
        return self._run("Hello World!")


research_tools = [
    RedditResearchTool(),
    QuoraResearchTool(),
    SerperDevTool()
]
