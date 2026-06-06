from tavily import TavilyClient

from src.config import settings
from src.observability.logger import (
    app_logger
)


class WebSearchAgent:
    """
    Web research agent
    """

    _client = None

    def __init__(self):

        self.api_key = (
            settings
            .TAVILY_API_KEY
        )

        self.top_k = (
            settings
            .WEB_TOP_K
        )

    def _get_client(self):

        if self._client is None:

            self._client = (
                TavilyClient(
                    api_key=(
                        self.api_key
                    )
                )
            )

            app_logger.success(
                "Tavily initialized"
            )

        return self._client

    def search(
        self,
        query: str
    ) -> list:
        """
        Web research
        """

        client = (
            self._get_client()
        )

        response = (
            client.search(
                query=query,

                search_depth=
                "advanced",

                max_results=(
                    self.top_k
                )
            )
        )

        results = []

        for item in (
            response.get(
                "results",
                []
            )
        ):

            results.append({
                "title":
                item.get(
                    "title"
                ),

                "content":
                item.get(
                    "content"
                ),

                "url":
                item.get(
                    "url"
                ),

                "score":
                item.get(
                    "score",
                    0.0
                )
            })

        app_logger.success(
            f"Retrieved "
            f"{len(results)} "
            f"web results"
        )

        return results


web_search_agent = (
    WebSearchAgent()
)