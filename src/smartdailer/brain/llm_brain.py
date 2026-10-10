import os
from typing import List, Dict, Optional
from smartdailer.brain.base import BrainBase


class LLMBrain(BrainBase):
    """
    LLM-powered dialogue brain supporting OpenAI, Ollama, Groq, or any OpenAI-compatible API.
    """

    def __init__(
        self,
        system_prompt: str,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
    ):
        from openai import OpenAI

        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature

        resolved_key = api_key or os.getenv("OPENAI_API_KEY", "dummy-key-for-local")
        resolved_base = base_url or os.getenv("OPENAI_BASE_URL", None)

        self.client = OpenAI(api_key=resolved_key, base_url=resolved_base)

    def respond(self, speaker_name: str, incoming_text: str, history: List[Dict[str, str]]) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]

        for turn in history:
            role = "assistant" if turn["speaker"] == speaker_name else "user"
            messages.append({"role": role, "content": turn["text"]})

        messages.append({"role": "user", "content": incoming_text})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=150,
        )

        reply = response.choices[0].message.content.strip()
        return reply
