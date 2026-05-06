# ══════════════════════════════════════════════════════════════════
# src/ingestion/loader.py
# Document loaders for PDF, ArXiv, Wikipedia, and web articles.
# Returns a flat list of LangChain Document objects with
# source_type metadata attached to every page/article.
# ══════════════════════════════════════════════════════════════════

import logging
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    WikipediaLoader,
    ArxivLoader,
)
from langchain.schema import Document

logger = logging.getLogger(__name__)


def load_pdfs(pdf_dir: str) -> list[Document]:
    """Load all PDFs from a directory recursively."""
    docs = []
    for path in Path(pdf_dir).rglob("*.pdf"):
        try:
            loader = PyPDFLoader(str(path))
            pages  = loader.load()
            for p in pages:
                p.metadata["source_type"] = "pdf"
                p.metadata["filename"]    = path.name
            docs.extend(pages)
            logger.info(f"  Loaded PDF: {path.name} ({len(pages)} pages)")
        except Exception as e:
            logger.warning(f"  Failed to load {path.name}: {e}")
    logger.info(f"PDFs loaded: {len(docs)} pages from {pdf_dir}")
    return docs


def load_arxiv_papers(arxiv_ids: list[str], max_docs: int = 5) -> list[Document]:
    """
    Load ArXiv papers by ID list.
    arxiv_ids: e.g. ["2005.11401", "2312.10997"]
    """
    docs = []
    for arxiv_id in arxiv_ids:
        try:
            loader = ArxivLoader(query=arxiv_id, load_max_docs=max_docs)
            papers = loader.load()
            for p in papers:
                p.metadata["source_type"] = "arxiv"
                p.metadata["arxiv_id"]    = arxiv_id
            docs.extend(papers)
            logger.info(f"  Loaded ArXiv {arxiv_id}: {len(papers)} docs")
        except Exception as e:
            logger.warning(f"  Failed ArXiv {arxiv_id}: {e}")
    return docs


def load_wikipedia_articles(topics: list[str], lang: str = "en") -> list[Document]:
    """
    Load Wikipedia articles by topic name.
    topics: e.g. ["Retrieval-augmented generation", "Large language model"]
    """
    docs = []
    for topic in topics:
        try:
            loader   = WikipediaLoader(query=topic, lang=lang, load_max_docs=1)
            articles = loader.load()
            for a in articles:
                a.metadata["source_type"] = "wikipedia"
                a.metadata["topic"]       = topic
            docs.extend(articles)
            logger.info(f"  Loaded Wikipedia: {topic}")
        except Exception as e:
            logger.warning(f"  Failed Wikipedia '{topic}': {e}")
    return docs


def load_web_articles(urls: list[str]) -> list[Document]:
    """Load web articles from a URL list via BeautifulSoup."""
    docs = []
    for url in urls:
        try:
            loader = WebBaseLoader(web_paths=[url])
            pages  = loader.load()
            for p in pages:
                p.metadata["source_type"] = "web"
                p.metadata["url"]         = url
            docs.extend(pages)
            logger.info(f"  Loaded web: {url}")
        except Exception as e:
            logger.warning(f"  Failed web '{url}': {e}")
    return docs


def load_all_corpus(
    pdf_dir:          Optional[str]       = None,
    arxiv_ids:        Optional[list[str]] = None,
    wikipedia_topics: Optional[list[str]] = None,
    web_urls:         Optional[list[str]] = None,
) -> list[Document]:
    """
    Load the full corpus from all configured sources.
    Returns a single flat list of Documents with source_type metadata.
    """
    all_docs: list[Document] = []

    if pdf_dir:
        all_docs.extend(load_pdfs(pdf_dir))
    if arxiv_ids:
        all_docs.extend(load_arxiv_papers(arxiv_ids))
    if wikipedia_topics:
        all_docs.extend(load_wikipedia_articles(wikipedia_topics))
    if web_urls:
        all_docs.extend(load_web_articles(web_urls))

    logger.info(f"Total corpus documents loaded: {len(all_docs)}")
    return all_docs
