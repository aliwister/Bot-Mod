"""The OpenAI-compatible vLLM clients and the intent list shared by every module.

launch_vllm.sh serves Qwen3-8B, Mistral-7B-Instruct-v0.3 and Llama-3.1-8B-Instruct on ports
BOTMOD_PORT_BASE, +1, +2 (default 8000-8002). BOTMOD_{QWEN,MISTRAL,LLAMA}_PORT point one model
at another replica.
"""
import os
from openai import OpenAI

_LLM_ALIASES = {
    "qwen":    "Qwen/Qwen3-8B",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama":   "meta-llama/Llama-3.1-8B-Instruct",
}


def resolve_model(name: str) -> str:
    return _LLM_ALIASES.get(name.lower(), name)


_PORT = int(os.environ.get("BOTMOD_PORT_BASE", "8000"))
_BASE_URL = {
    "Qwen/Qwen3-8B": f"http://localhost:{os.environ.get('BOTMOD_QWEN_PORT', _PORT)}/v1",
    "mistralai/Mistral-7B-Instruct-v0.3": f"http://localhost:{os.environ.get('BOTMOD_MISTRAL_PORT', _PORT + 1)}/v1",
    "meta-llama/Llama-3.1-8B-Instruct": f"http://localhost:{os.environ.get('BOTMOD_LLAMA_PORT', _PORT + 2)}/v1",
}
_clients: dict[str, OpenAI] = {}


def _client(model: str) -> OpenAI:
    url = _BASE_URL[resolve_model(model)]
    if url not in _clients:
        _clients[url] = OpenAI(base_url=url, api_key="none")
    return _clients[url]


INTENTS = [
    "subtle_promotion",      # product/service persuasion
    "narrative_pushing",     # ideological/political agenda
    "spam",                  # irrelevant or repetitive content
    "elicitation",           # prompting users to reveal information
    "organic_contribution",  # genuine participation
]

# Qwen3 is run without its thinking mode: " /nothink" is appended to every user turn
_NOTHINK_MODELS = {"Qwen/Qwen3-8B"}
