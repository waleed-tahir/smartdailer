from abc import ABC, abstractmethod
from typing import List, Dict


class BrainBase(ABC):
    """Abstract Base Class for Voice Agent Dialogue Brains."""

    @abstractmethod
    def respond(self, speaker_name: str, incoming_text: str, history: List[Dict[str, str]]) -> str:
        """
        Generate response text given incoming transcribed text and conversation history.
        history format: [{"speaker": "Agent A", "text": "..."}, ...]
        """
        pass
