"""Util that calls Google Search."""
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup
import requests
from pydantic import BaseModel, Extra, root_validator
from langchain.utils import get_from_dict_or_env


class GoogleSearchAPIWrapper(BaseModel):
    """Wrapper for Google Search API.

    This wrapper no longer uses the Google API, but directly sends requests to Google's search URL and parses the resulting HTML.
    """

    google_api_key: Optional[str] = None
    google_cse_id: Optional[str] = None
    k: int = 10
    siterestrict: bool = False

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def _google_search_results(self, search_term: str, **kwargs: Any) -> List[dict]:
        url = f'https://www.google.com/search?q={search_term}'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('div', class_='g')
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
        results = self._google_search_results(query, num=self.k)
        if len(results) == 0:
            return "No good Google Search Result was found"
        for result in results:
            snippet = result.find('span', class_='st')
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
        results = self._google_search_results(
            query, num=num_results, **(search_params or {})
        )
        if len(results) == 0:
            return [{"Result": "No good Google Search Result was found"}]
        for result in results:
            title = result.find('h3')
            link = result.find('a')
            snippet = result.find('span', class_='st')
            metadata_result = {
                "title": title.text if title else "",
                "link": link['href'] if link else "",
                "snippet": snippet.text if snippet else ""
            }
            metadata_results.append(metadata_result)

        return metadata_results
