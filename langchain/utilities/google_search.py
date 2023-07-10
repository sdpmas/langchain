"""Util that calls Google Search."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Extra, root_validator
from bs4 import BeautifulSoup
import requests
from langchain.utils import get_from_dict_or_env


class GoogleSearchAPIWrapper(BaseModel):
    """Wrapper for Google Search API."""

    google_api_key: Optional[str] = None
    google_cse_id: Optional[str] = None
    k: int = 10
    siterestrict: bool = False

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def _google_search_results(self, search_term: str, **kwargs: Any) -> List[dict]:
        url = f"https://www.google.com/search?q={search_term}&num={self.k}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        for g in soup.find_all('div', class_='rc'):
            anchors = g.find_all('a')
            if anchors:
                link = anchors[0]['href']
                title = g.find('h3').text
                item = {"title": title, "link": link}
                results.append(item)
        return results

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        google_api_key = get_from_dict_or_env(
            values, "google_api_key", "GOOGLE_API_KEY"
        )
        values["google_api_key"] = google_api_key

        google_cse_id = get_from_dict_or_env(values, "google_cse_id", "GOOGLE_CSE_ID")
        values["google_cse_id"] = google_cse_id

        return values

    def run(self, query: str) -> str:
        """Run query through GoogleSearch and parse result."""
        snippets = []
        results = self._google_search_results(query)
        if len(results) == 0:
            return "No good Google Search Result was found"
        for result in results:
            if "snippet" in result:
                snippets.append(result["snippet"])

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
        results = self._google_search_results(query, num=num_results, **(search_params or {}))
        if len(results) == 0:
            return [{"Result": "No good Google Search Result was found"}]
        for result in results:
            metadata_result = {
                "title": result["title"],
                "link": result["link"],
            }
            if "snippet" in result:
                metadata_result["snippet"] = result["snippet"]
            metadata_results.append(metadata_result)

        return metadata_results
