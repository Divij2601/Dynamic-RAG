from src.config import settings
from src.models.groq_provider import groq_provider
from src.observability.logger import (
    app_logger
)


class ResponseGenerator:
    """
    Groq-based grounded response generator.
    Delegates the LLM call to the resilient
    GroqProvider (retry / backoff / fallback).
    """

    def __init__(self):

        self.model = settings.DEFAULT_LLM

    def generate(
        self,
        prompt: str
    ) -> str:
        """
        Generate grounded answer.
        """

        answer = groq_provider.complete(
            prompt=prompt,
            model=self.model,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS
        )

        app_logger.success("Answer generated")

        return answer


response_generator = ResponseGenerator()
