# ══════════════════════════════════════════════════════════════════
# src/ingestion/chunker.py
# Splits LangChain Documents into fixed-size overlapping chunks.
# Default: 768 tokens / 128 overlap (17% overlap ratio).
#
# Empirical rationale (see docs/experiment_log.md Finding 3):
#   256 tokens  → named-entity fragmentation
#   512 tokens  → borderline on dense technical passages
#   768 tokens  → optimal: best FP3 + NDCG + BERTScore F1
#   1024 tokens → topical dilution; specificity gap (FP6) inflates
# ══════════════════════════════════════════════════════════════════

import logging
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.utils.config import RAGExperimentConfig

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: list[Document],
    config:    RAGExperimentConfig,
) -> list[Document]:
    """
    Split a list of Documents into overlapping chunks.

    Each chunk's metadata is preserved from the source document and
    extended with:
        chunk_index   : sequential position within the source document
        source_doc_id : index of the original document in the input list
        chunk_size    : configured chunk_size for traceability
        start_index   : character offset within the original text
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size      = config.chunk_size,
        chunk_overlap   = config.chunk_overlap,
        length_function = len,
        add_start_index = True,
    )

    chunks: list[Document] = []

    for doc_idx, doc in enumerate(documents):
        doc_chunks = splitter.split_documents([doc])
        for chunk_idx, chunk in enumerate(doc_chunks):
            chunk.metadata["chunk_index"]   = chunk_idx
            chunk.metadata["source_doc_id"] = doc_idx
            chunk.metadata["chunk_size"]    = config.chunk_size
            chunks.append(chunk)

    logger.info(
        f"Chunked {len(documents)} documents → {len(chunks)} chunks "
        f"(size={config.chunk_size}, overlap={config.chunk_overlap}, "
        f"ratio={config.overlap_ratio:.0%})"
    )
    return chunks
