from typing import List, Dict, Optional
from smartdailer.brain.base import BrainBase


class ScenarioBrain(BrainBase):
    """
    Scenario-driven dialogue brain for deterministic, zero-API-key testing
    of the 2-agent audio conversation pipeline.
    """

    def __init__(self, persona: str, scripted_replies: Optional[List[str]] = None):
        self.persona = persona
        self.scripted_replies = scripted_replies or []
        self._reply_index = 0

    def respond(self, speaker_name: str, incoming_text: str, history: List[Dict[str, str]]) -> str:
        if self._reply_index < len(self.scripted_replies):
            reply = self.scripted_replies[self._reply_index]
            self._reply_index += 1
            return reply

        # Fallback dynamic conversational responses if script is exhausted
        incoming_lower = incoming_text.lower()
        if "hello" in incoming_lower or "hi" in incoming_lower:
            return f"Hello there! I heard you say '{incoming_text}'. How are you doing today?"
        elif "yes" in incoming_lower or "sure" in incoming_lower:
            return "That sounds great, let us proceed with the details."
        elif "bye" in incoming_lower or "thank" in incoming_lower:
            return "Thank you so much. Have a wonderful day, goodbye!"
        else:
            return f"Understood. Regarding '{incoming_text.strip()}', let's make sure everything is confirmed."
