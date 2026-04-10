"""Pydantic data models for the Thinker analysis pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import List

from pydantic import BaseModel, ConfigDict, Field


class Fallacy(BaseModel):
    """A single thinking fallacy detected in the input."""

    name: str = Field(..., description="Name of the fallacy")
    explanation: str = Field(..., description="Why this fallacy applies here")


class AnalysisResult(BaseModel):
    """Structured output returned by the LLM for one analysis run."""

    facts: List[str] = Field(default_factory=list, description="Objective facts identified")
    emotions: List[str] = Field(default_factory=list, description="Emotions / emotional language identified")
    assumptions: List[str] = Field(default_factory=list, description="Hidden assumptions identified")
    fallacies: List[Fallacy] = Field(default_factory=list, description="Thinking fallacies detected")
    explanations: List[str] = Field(
        default_factory=list,
        min_length=1,
        max_length=5,
        description="Alternative explanations (ideally 3)",
    )
    rational_conclusion: str = Field("", description="A more rational conclusion")


class AnalysisRecord(BaseModel):
    """A persisted analysis record (maps to SQLite row)."""
    model_config = ConfigDict(protected_namespaces=())

    id: int | None = None
    input_text: str = ""
    language: str = "zh"
    result: AnalysisResult | None = None
    provider_name: str = ""
    model_name: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


class QARecord(BaseModel):
    """A persisted Q&A record (maps to SQLite row)."""
    model_config = ConfigDict(protected_namespaces=())

    id: int | None = None
    book_name: str = ""
    question: str = ""
    answer: str = ""
    language: str = "zh"
    provider_name: str = ""
    model_name: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


class ChatRecord(BaseModel):
    """A persisted free-chat record (maps to SQLite row)."""
    model_config = ConfigDict(protected_namespaces=())

    id: int | None = None
    question: str = ""
    answer: str = ""
    language: str = "zh"
    provider_name: str = ""
    model_name: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


class GatewayQARecord(BaseModel):
    """A persisted gateway Q&A record (maps to SQLite row)."""
    model_config = ConfigDict(protected_namespaces=())

    id: int | None = None
    question: str = ""
    answer: str = ""
    language: str = "zh"
    provider_name: str = ""
    model_name: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


