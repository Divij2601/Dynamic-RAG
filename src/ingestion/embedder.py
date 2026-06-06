import time
from typing import List, Dict

from sentence_transformers import (
    SentenceTransformer
)

from src.config import settings
from src.observability.logger import (
    app_logger
)


class EmbeddingGenerator:
    """
    Generate embeddings
    for document chunks
    """

    _model = None

    def __init__(self):

        self.model_name = (
            settings.EMBEDDING_MODEL
        )

        self.device = (
            settings.EMBEDDING_DEVICE
        )

        self.batch_size = (
            settings
            .EMBEDDING_BATCH_SIZE
        )

    def _load_model(self):
        """
        Singleton model loading
        """

        if self._model is None:

            app_logger.info(
                f"Loading embedding model: "
                f"{self.model_name}"
            )

            self._model = (
                SentenceTransformer(
                    self.model_name,
                    device=self.device
                )
            )

            app_logger.success(
                "Embedding model loaded"
            )

        return self._model

    def generate_embeddings(
        self,
        chunks: List[Dict]
    ) -> List[Dict]:
        """
        Generate embeddings
        for chunk list
        """

        start_time = (
            time.perf_counter()
        )

        model = (
            self._load_model()
        )

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = (
            model.encode(
                texts,
                batch_size=(
                    self.batch_size
                ),
                show_progress_bar=True,
                normalize_embeddings=True
            )
        )

        enriched_chunks = []

        for chunk, embedding in zip(
            chunks,
            embeddings
        ):

            enriched_chunks.append({
                **chunk,
                "embedding":
                embedding.tolist()
            })

        execution_time = round(
            (
                time.perf_counter()
                - start_time
            ) * 1000,
            2
        )

        app_logger.success(
            f"Generated embeddings "
            f"for "
            f"{len(chunks)} chunks "
            f"in "
            f"{execution_time} ms"
        )
    def generate_query_embedding(
            self,
            query: str
        ) -> list:
            """
            Generate embedding
            for query
            """

            model = self._load_model()

            embedding = model.encode(
                query,
                normalize_embeddings=True
            )

            return embedding.tolist()    

            return enriched_chunks


embedding_generator = (
    EmbeddingGenerator()
)