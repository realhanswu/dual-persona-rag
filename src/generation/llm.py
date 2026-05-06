# ══════════════════════════════════════════════════════════════════
# src/generation/llm.py
# Loads Mistral-7B-Instruct-v0.3 in 4-bit NF4 via BitsAndBytes.
# engineering → temp=0.0 (deterministic)
# marketing   → temp=0.2 (natural tone variation)
# ══════════════════════════════════════════════════════════════════

import os
import logging
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from langchain_huggingface import HuggingFacePipeline
from src.utils.config import RAGExperimentConfig

logger   = logging.getLogger(__name__)
_MODEL_ID = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")


def _resolve_device() -> str:
    device = os.getenv("DEVICE", "auto")
    return ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device


def load_mistral(
    config:  RAGExperimentConfig,
    persona: str = "engineering",
) -> HuggingFacePipeline:
    """Load and return a Mistral-7B pipeline for the given persona."""
    temperature = (
        config.temperature_eng if persona == "engineering"
        else config.temperature_mkt
    )
    device = _resolve_device()
    logger.info(f"Loading LLM: {_MODEL_ID} | persona={persona} | temp={temperature} | device={device}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID, token=os.getenv("HF_TOKEN"))
    model     = AutoModelForCausalLM.from_pretrained(
        _MODEL_ID,
        quantization_config = bnb_config,
        device_map          = device,
        token               = os.getenv("HF_TOKEN"),
    )

    pipe = pipeline(
        task             = "text-generation",
        model            = model,
        tokenizer        = tokenizer,
        temperature      = temperature,
        do_sample        = temperature > 0.0,
        max_new_tokens   = 512,
        return_full_text = False,
    )

    logger.info(f"LLM loaded: {_MODEL_ID} (4-bit NF4, bfloat16)")
    return HuggingFacePipeline(pipeline=pipe)
