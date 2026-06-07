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

    # Short description of what the document
    # knowledge base contains. The planner uses
    # this to decide whether a query can be answered
    # internally or needs the web. Update this when
    # the corpus changes.
    KNOWLEDGE_BASE_DESCRIPTION: str = (
        "Documents covering: (1) Geopolitics and world "
        "affairs — country profiles (US, Russia, China, "
        "India, EU), international organizations (UN, "
        "NATO, BRICS, SCO), treaties (Westphalia, NPT, "
        "Paris Agreement, Kyoto), major wars and eras "
        "(WWI, WWII, Cold War, Cuban Missile Crisis, "
        "decolonization), geopolitical concepts, and "
        "global affairs 2025-2026. "
        "(2) Indian history and South Asian studies — "
        "ancient and medieval India, modern Indian "
        "history, India's approach to Asia, political "
        "science and international relations theory "
        "(PSIR). "
        "(3) Geopolitical analysis reports — State of "
        "Power 2025, defense and security studies, "
        "military history. "
        "The knowledge base covers historical facts, "
        "political analysis, and events up to 2026. "
        "It does NOT contain live data like stock prices, "
        "sports results, current population statistics, "
        "private internal documents, or classified information."
    )

    DEFAULT_LLM: str
    FAST_MODEL: str
    CRITIC_MODEL: str

    TEMPERATURE: float = 0.2
    MAX_TOKENS: int = 2048

    # LLM call resilience (rate limits / transient
    # errors). Distinct from the graph's answer-retry
    # loop (MAX_RETRIES below).
    LLM_MAX_RETRIES: int = 3
    LLM_BACKOFF_BASE: float = 2.0
    LLM_RETRY_MAX_WAIT: float = 30.0
    # When a model is rate-limited beyond the max wait,
    # fall back to this model once (it has its own
    # per-model quota). Empty disables fallback.
    LLM_FALLBACK_MODEL: str = "llama-3.1-8b-instant"

    # Conversation memory management
    # When session exceeds this many turns, older
    # turns are compressed into a summary.
    MAX_HISTORY_TURNS: int = 8
    # Recent turns kept verbatim after summarisation.
    SUMMARY_KEEP_RECENT: int = 3

    # ==========================
    # Retrieval Config
    # ==========================

    TOP_K: int = 5

    # Candidate pool size retrieved (hybrid) before
    # reranking down to FINAL_TOP_K. pool=20 ties the
    # best measured retrieval metrics and gives the
    # reranker headroom as the corpus grows.
    RERANK_TOP_K: int = 20

    # Hybrid fusion: "weighted" (score-weighted sum,
    # dense-favored) or "rrf" (reciprocal rank fusion).
    # Measured best on the current corpus: "weighted".
    # RRF is available for larger / more lexical corpora.
    FUSION_MODE: str = "weighted"
    DENSE_WEIGHT: float = 0.7
    SPARSE_WEIGHT: float = 0.3
    RRF_K: int = 60

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

    # Reranker output: how many chunks are passed to
    # the generator. Measured optimal for this corpus:
    # k=8 achieves Recall=1.0 (vs 0.979 at k=5) with
    # stable MRR. Higher adds no recall gain.
    FINAL_TOP_K: int = 8

    
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

    MEMORY_COLLECTION_NAME: str = (
        "dynamic_rag_memory"
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