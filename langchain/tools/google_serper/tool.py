"""Tool for the Google Search API."""

import requests
from typing import Optional
from bs4 import BeautifulSoup
from pydantic.fields import Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool


class GoogleSerperRun(BaseTool):
    """Tool that adds the capability to query the Google search API."""

    name = "google_serper"
    description = (
        "A low-cost Google Search API."
        "Useful for when you need to answer questions about current events."
        "Input should be a search query."
    )

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return str(self.search_google(query))

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return str(self.search_google(query))

    @staticmethod
    def search_google(query: str) -> str:
        """Send a GET request to Google Search and parse the results."""
        response = requests.get(f"https://www.google.com/search?q={query}")
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.prettify()


class GoogleSerperResults(BaseTool):
    """Tool that has capability to query the Google Search API
    and get back json."""

    name = "google_serrper_results_json"
    description = (
        "A low-cost Google Search API."
        "Useful for when you need to answer questions about current events."
        "Input should be a search query. Output is a JSON object of the query results"
    )

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return str(self.search_google(query))

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return str(self.search_google(query))

    @staticmethod
    def search_google(query: str) -> str:
        """Send a GET request to Google Search and parse the results."""
        response = requests.get(f"https://www.google.com/search?q={query}")
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.prettify()
