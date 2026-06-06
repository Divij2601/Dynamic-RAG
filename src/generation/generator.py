from groq import Groq

from src.config import settings
from src.observability.logger import (
    app_logger
)


class ResponseGenerator:
    """
    Groq-based grounded
    response generator
    """

    _client = None

    def __init__(self):

        self.model = (
            settings.DEFAULT_LLM
        )

    def _get_client(self):
        """
        Singleton Groq client
        """

        if self._client is None:

            self._client = Groq(
                api_key=(
                    settings
                    .GROQ_API_KEY
                )
            )

            app_logger.success(
                "Groq client initialized"
            )

        return self._client

    def generate(
        self,
        prompt: str
    ) -> str:
        """
        Generate grounded answer
        """

        client = (
            self._get_client()
        )

        response = (
            client.chat.completions.create(
                model=self.model,

                messages=[
                    {
                        "role":
                        "user",

                        "content":
                        prompt
                    }
                ],

                temperature=(
                    settings
                    .TEMPERATURE
                ),

                max_tokens=(
                    settings
                    .MAX_TOKENS
                )
            )
        )

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        app_logger.success(
            "Answer generated"
        )

        return answer


response_generator = (
    ResponseGenerator()
)