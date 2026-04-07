"""
models.py — Typed Pydantic models for the Email Triage OpenEnv environment.
"""
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

EmailLabel = Literal["work", "personal", "spam", "newsletter", "urgent", "unlabeled"]

ActionType = Literal[
    "open_email",
    "label_email",
    "reply_email",
    "delete_email",
    "unsubscribe",
    "flag_urgent",
    "archive_email",
    "done",
]


class EmailMessage(BaseModel):
    id: str
    sender: str
    subject: str
    body: str
    timestamp: str
    label: EmailLabel = "unlabeled"
    is_open: bool = False
    is_deleted: bool = False
    is_archived: bool = False
    is_flagged_urgent: bool = False
    is_unsubscribed: bool = False
    reply_body: Optional[str] = None
    requires_reply: bool = False
    is_spam: bool = False
    is_newsletter: bool = False
    true_label: EmailLabel = "unlabeled"  # ground truth for grading


# ---------------------------------------------------------------------------
# OpenEnv interface models
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """What the agent sees at each step."""
    inbox: List[EmailMessage]
    current_email: Optional[EmailMessage] = None
    labels: List[str] = ["work", "personal", "spam", "newsletter", "urgent"]
    step_count: int = 0
    task_id: str = ""
    message: str = ""  # feedback / narration for the agent


class Action(BaseModel):
    """A single action the agent takes."""
    action_type: ActionType
    email_id: Optional[str] = None
    label: Optional[EmailLabel] = None
    body: Optional[str] = None  # for reply_email


class Reward(BaseModel):
    """Reward signal returned after each step."""
    value: float = Field(..., ge=-1.0, le=1.0)
    reason: str = ""
    cumulative: float = 0.0


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = {}
