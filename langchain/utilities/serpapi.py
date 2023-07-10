"""Chain that calls Google Search.

Heavily borrowed from https://github.com/ofirpress/self-ask
"""
import os
import sys
import requests
from typing import Any, Dict, Optional, Tuple
from bs4 import BeautifulSoup

from pydantic import BaseModel, Extra, Field, root_validator

from langchain.utils import get_from_dict_or_env


class HiddenPrints:
    """Context manager to hide prints."""

    def __enter__(self) -> None:
        """Open file to pipe stdout to."""
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, *_: Any) -> None:
        """Close file that stdout was piped to."""
        sys.stdout.close()
        sys.stdout = self._original_stdout


class GoogleSearchWrapper(BaseModel):
    """Wrapper around Google Search."""

    params: dict = Field(
        default={
            "engine": "google",
            "google_domain": "google.com",
            "gl": "us",
            "hl": "en",
        }
    )

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def run(self, query: str, **kwargs: Any) -> str:
        """Run query through Google Search and parse result."""
        return self._process_response(self.results(query))

    def results(self, query: str) -> dict:
        """Run query through Google Search and return the raw result."""
        params = self.get_params(query)
        with HiddenPrints():
            res = requests.get('https://www.google.com/search', params=params)
            soup = BeautifulSoup(res.text, 'html.parser')
        return soup

    def get_params(self, query: str) -> Dict[str, str]:
        """Get parameters for Google Search."""
        _params = {
            "q": query,
        }
        params = {**self.params, **_params}
        return params

    @staticmethod
    def _process_response(soup: BeautifulSoup) -> str:
        """Process response from Google Search."""
        results = soup.find_all('div', class_='g')
        for result in results:
            try:
                title = result.find('h3').get_text()
                link = result.find('a').get('href')
                description = result.find('span', class_='st').get_text()
                return {'title': title, 'link': link, 'description': description}
            except AttributeError:
                pass
        return "No good search result found"

