"""Util that calls Google Search."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, root_validator
from bs4 import BeautifulSoup
import requests

from langchain.utils import get_from_dict_or_env


class GoogleSearchAPIWrapper(BaseModel):
    """Wrapper for Google Search API.

    This class now uses requests and BeautifulSoup to directly scrape Google's search results.
    """

    google_search_url: str = 'https://www.google.com/search?q='
    k: int = 10
    siterestrict: bool = False

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def _google_search_results(self, search_term: str, **kwargs: Any) -> List[dict]:
        res = requests.get(self.google_search_url + search_term)
        soup = BeautifulSoup(res.text, 'html.parser')
        search_results = soup.find_all('div', {'class': 'g'})
        return search_results

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        return values

    def run(self, query: str) -> str:
        """Run query through GoogleSearch and parse result."""
        snippets = []
        results = self._google_search_results(query)
        if len(results) == 0:
            return "No good Google Search Result was found"
        for result in results:
            snippet = result.find('span', {'class': 'st'})
            if snippet:
                snippets.append(snippet.text)

        return " ".join(snippets)

    def results(
        self,
        query: str,
        num_results: int,
        search_params: Optional[Dict[str, str]] = None,
    ) -> List[Dict]:
        """Run query through GoogleSearch and return metadata.

        Args:
            query: The query to search for.
            num_results: The number of results to return.
            search_params: Parameters to be passed on search

        Returns:
            A list of dictionaries with the following keys:
                snippet - The description of the result.
                title - The title of the result.
                link - The link to the result.
        """
        metadata_results = []
        results = self._google_search_results(query)
        if len(results) == 0:
            return [{"Result": "No good Google Search Result was found"}]
        for result in results:
            metadata_result = {
                "title": result.find('h3').text if result.find('h3') else '',
                "link": result.find('a')['href'] if result.find('a') else '',
            }
            snippet = result.find('span', {'class': 'st'})
            if snippet:
                metadata_result["snippet"] = snippet.text
            metadata_results.append(metadata_result)

        return metadata_results
