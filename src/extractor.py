from typing import Literal

from pydantic import BaseModel, Field

from .llm import LLM, structured_call


class ExtractedFact(BaseModel):
    text: str = Field(min_length=5, max_length=200)
    kind: Literal["preference", "fact", "goal", "event"]
    importance: int = Field(ge=1, le=5)


class ExtractionResult(BaseModel):
    facts: list[ExtractedFact] = Field(default_factory=list, max_length=5)


EXTRACT_SYSTEM = (
    "You extract long-term memories about the USER from one chat message. Keep only durable "
    "information: personal facts, preferences, goals, relationships, upcoming events. Ignore "
    "small talk, general knowledge questions and anything about the assistant. Write each fact "
    "in the third person starting with 'User' (for example 'User is vegetarian'). "
    "importance: 1 trivial, 3 useful, 5 critical (allergies, safety, hard constraints). "
    'If there is nothing worth remembering return {"facts": []}.'
)


def extract_facts(user_message: str, llm: LLM) -> list[ExtractedFact]:
    if len(user_message.split()) < 4:   # skip "ok", "thanks", "yes please"
        return []
    result = structured_call(EXTRACT_SYSTEM, user_message, ExtractionResult, llm)
    return result.facts if result else []