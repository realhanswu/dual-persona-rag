# ══════════════════════════════════════════════════════════════════
# src/generation/chains.py
# Dual-persona LangChain RAG chains.
# Both chains share the same retriever but use separate prompts,
# separate LLM instances, and separate evaluation pipelines.
# ══════════════════════════════════════════════════════════════════

import logging
from langchain.chains import RetrievalQA
from langchain.schema.retriever import BaseRetriever
from langchain_huggingface import HuggingFacePipeline
from src.generation.prompts import get_prompt
from src.utils.config import RAGExperimentConfig

logger = logging.getLogger(__name__)


def build_rag_chain(
    retriever: BaseRetriever,
    llm:       HuggingFacePipeline,
    persona:   str,
    config:    RAGExperimentConfig,
) -> RetrievalQA:
    """Build a RetrievalQA chain for the given persona."""
    prompt = get_prompt(persona)
    temp   = config.temperature_eng if persona == "engineering" else config.temperature_mkt

    chain = RetrievalQA.from_chain_type(
        llm                     = llm,
        chain_type              = "stuff",
        retriever               = retriever,
        return_source_documents = True,
        chain_type_kwargs       = {"prompt": prompt},
    )

    logger.info(
        f"RAG chain built — persona={persona}, "
        f"temperature={temp}, reranker={config.use_reranker}"
    )
    return chain


def run_chain(chain: RetrievalQA, question: str) -> dict:
    """Run one question through the chain and return a structured result dict."""
    result      = chain.invoke({"query": question})
    source_docs = result.get("source_documents", [])
    return {
        "question":               question,
        "rag_answer":             result["result"].strip(),
        "retrieved_chunks":       [doc.page_content for doc in source_docs],
        "context_sources":        [doc.metadata     for doc in source_docs],
        "context_passed_to_llm":  "\n\n".join(doc.page_content for doc in source_docs),
    }
