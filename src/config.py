from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # ==========================
    # App Config
    # ==========================

    APP_NAME: str = "Dynamic-RAG"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    HOST: str = "127.0.0.1"
    PORT: int = 8000

    LOG_LEVEL: str = "INFO"

    # ==========================
    # Database Config
    # ==========================

    MONGO_URI: str

    QDRANT_URL: str
    QDRANT_API_KEY: str = ""

    # ==========================
    # LLM Config
    # ==========================

    LLM_PROVIDER: str = "groq"

    GROQ_API_KEY: str

    DEFAULT_LLM: str
    FAST_MODEL: str
    CRITIC_MODEL: str

    TEMPERATURE: float = 0.2
    MAX_TOKENS: int = 2048

    # ==========================
    # Retrieval Config
    # ==========================

    TOP_K: int = 5
    RERANK_TOP_K: int = 10

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # ==========================
    # Runtime Config
    # ==========================

    REQUEST_TIMEOUT: int = 30
    MAX_RETRIES: int = 2

    # ==========================
    # Pydantic Settings Config
    # ==========================
    # ==========================
    # Embedding Config
    # ==========================

    EMBEDDING_MODEL: str = (
        "BAAI/bge-small-en-v1.5"
    )

    RERANK_MODEL: str = (
       "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    FINAL_TOP_K: int = 5

    
    EMBEDDING_DEVICE: str = "cpu"

    EMBEDDING_BATCH_SIZE: int = 16

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"
    )

    # ==========================
    # Vector DB Config
    # ==========================

    QDRANT_COLLECTION_NAME: str = (
        "dynamic_rag_documents"
    )

    VECTOR_DIMENSION: int = 384

    # ==========================
    # Web Research
    # ==========================

    TAVILY_API_KEY: str = ""

    WEB_TOP_K: int = 5
    


@lru_cache
def get_settings() -> Settings:
    """
    Singleton settings instance
    """
    return Settings()


settings = get_settings()