# ══════════════════════════════════════════════════════════════════
# src/ingestion/indexer.py
# Embeds document chunks and stores them in a Qdrant vector index.
#
# Embedding model: multi-qa-mpnet-base-dot-v1
#   - Trained for asymmetric semantic retrieval (short query → long passage)
#   - Native similarity: dot product (no cosine normalisation overhead)
#   - Same model reused throughout evaluation for threshold calibration
#
# Storage: Qdrant in-memory (POC default).
#   Set QDRANT_URL in .env for persistent/remote deployments.
# ══════════════════════════════════════════════════════════════════

import logging
import os
from langchain.schema import Document
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from src.utils.config import RAGExperimentConfig

logger = logging.getLogger(__name__)

_DISTANCE_MAP = {
    "Dot":    Distance.DOT,
    "Cosine": Distance.COSINE,
    "Euclid": Distance.EUCLID,
}


def build_embedding_model(config: RAGExperimentConfig) -> HuggingFaceEmbeddings:
    """Load the sentence-transformer embedding model specified in config."""
    device = os.getenv("DEVICE", "auto")
    logger.info(f"Loading embedding model: {config.embedding_model} (device={device})")
    return HuggingFaceEmbeddings(
        model_name    = config.embedding_model,
        model_kwargs  = {"device": device},
        encode_kwargs = {"normalize_embeddings": False},
    )


def build_vector_store(
    chunks:     list[Document],
    config:     RAGExperimentConfig,
    collection: str = "persona_lens",
) -> Qdrant:
    """
    Embed all chunks and load them into Qdrant.

    Args:
        chunks     : list of chunked Documents from chunker.py
        config     : experiment config (embedding_model, qdrant_distance)
        collection : Qdrant collection name

    Returns:
        Qdrant vector store ready for retrieval.
    """
    qdrant_url = os.getenv("QDRANT_URL", "")
    qdrant_key = os.getenv("QDRANT_API_KEY", "")

    embedding_fn = build_embedding_model(config)
    distance     = _DISTANCE_MAP.get(config.qdrant_distance, Distance.DOT)

    if qdrant_url:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_key or None)
        logger.info(f"Connecting to remote Qdrant at {qdrant_url}")
    else:
        client = QdrantClient(":memory:")
        logger.info("Using in-memory Qdrant instance (POC mode)")

    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        sample_vec = embedding_fn.embed_query("warmup")
        client.create_collection(
            collection_name = collection,
            vectors_config  = VectorParams(
                size     = len(sample_vec),
                distance = distance,
            ),
        )
        logger.info(
            f"Created collection '{collection}' "
            f"(dim={len(sample_vec)}, distance={distance})"
        )

    vector_store = Qdrant.from_documents(
        documents       = chunks,
        embedding       = embedding_fn,
        collection_name = collection,
        client          = client,
    )

    logger.info(f"Indexed {len(chunks)} chunks into '{collection}'")
    return vector_store
