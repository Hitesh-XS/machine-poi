"""
LLM Function Adapters for LightRAG

Provides async LLM functions compatible with LightRAG's requirements
using the existing SteeredLLM infrastructure.
"""

import logging
from typing import TYPE_CHECKING, Optional, Dict, List
import asyncio

if TYPE_CHECKING:
    from .llm_wrapper import SteeredLLM

logger = logging.getLogger("machine_poi.llm_adapters")


def create_ollama_adapter(
    model_name: str = "qwen2.5:7b",
    host: str = "http://localhost:11434",
    timeout: int = 300,
) -> callable:
    """
    Create an async LLM function for LightRAG using Ollama.

    Args:
        model_name: Ollama model name
        host: Ollama host URL
        timeout: Request timeout

    Returns:
        Async function compatible with LightRAG
    """
    import httpx

    async def ollama_complete(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict]] = None,
        **kwargs,
    ) -> str:
        """Async Ollama completion for LightRAG."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history_messages:
            messages.extend(history_messages)

        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{host}/api/chat",
                json={
                    "model": model_name,
                    "messages": messages,
                    "stream": False,
                    "options": kwargs.get("options", {"num_ctx": 8192}),
                },
            )
            response.raise_for_status()
            return response.json()["message"]["content"]

    return ollama_complete


def create_openai_adapter(
    model_name: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> callable:
    """
    Create an async LLM function for LightRAG using OpenAI API.

    Supports all OpenAI models including:
    - gpt-4o, gpt-4o-mini
    - gpt-5.2, chatgpt-5.2 (latest)
    - o1, o1-mini (reasoning models)

    Args:
        model_name: OpenAI model name (e.g., "gpt-5.2", "gpt-4o-mini")
        api_key: API key (uses OPENAI_API_KEY env var if None)
        base_url: Optional custom base URL for API-compatible services

    Returns:
        Async function compatible with LightRAG
    """
    import os
    from openai import AsyncOpenAI

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def openai_complete(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict]] = None,
        **kwargs,
    ) -> str:
        """Async OpenAI completion for LightRAG."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history_messages:
            messages.extend(history_messages)

        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 2048),
        )

        return response.choices[0].message.content

    return openai_complete


def create_gemini_adapter(
    model_name: str = "gemini-2.0-flash",
    api_key: Optional[str] = None,
) -> callable:
    """
    Create an async LLM function for LightRAG using Google Gemini API.

    Supports Gemini models including:
    - gemini-2.0-flash, gemini-2.0-flash-lite
    - gemini-3.0-pro, gemini-3.0-ultra (latest)
    - gemini-1.5-pro, gemini-1.5-flash

    Args:
        model_name: Gemini model name (e.g., "gemini-3.0-pro")
        api_key: API key (uses GOOGLE_API_KEY or GEMINI_API_KEY env var if None)

    Returns:
        Async function compatible with LightRAG
    """
    import os
    from google import genai
    from google.genai import types

    api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    def content(role: str, text: str):
        return types.Content(role=role, parts=[types.Part.from_text(text=text)])

    async def gemini_complete(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict]] = None,
        **kwargs,
    ) -> str:
        """Async Gemini completion for LightRAG."""
        contents = [
            content("user" if msg.get("role") == "user" else "model", msg.get("content", ""))
            for msg in history_messages or []
        ]
        contents.append(content("user", prompt))

        response = await client.aio.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=kwargs.get("temperature", 0.7),
                max_output_tokens=kwargs.get("max_tokens", 2048),
            ),
        )
        return response.text

    return gemini_complete


def create_local_llm_adapter(
    steered_llm: "SteeredLLM",
) -> callable:
    """
    Create an async LLM function for LightRAG using existing SteeredLLM.

    This allows using the same local LLM for both steering and entity extraction.

    Args:
        steered_llm: Existing SteeredLLM instance

    Returns:
        Async function compatible with LightRAG
    """

    async def local_complete(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict]] = None,
        **kwargs,
    ) -> str:
        """Async wrapper for local LLM."""
        # Build full prompt
        full_prompt = ""

        if system_prompt:
            full_prompt += f"System: {system_prompt}\n\n"

        if history_messages:
            for msg in history_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                full_prompt += f"{role.capitalize()}: {content}\n"

        full_prompt += f"User: {prompt}\n\nAssistant:"

        def complete():
            # Extraction output becomes the shared knowledge graph; steered
            # text would carry the intervention into every later retrieval.
            with steered_llm.steering_disabled():
                return steered_llm.generate(
                    prompt=full_prompt,
                    max_new_tokens=kwargs.get("max_tokens", 1024),
                    temperature=kwargs.get("temperature", 0.7),
                )

        # Run in executor to avoid blocking
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, complete)

    return local_complete
