"""
environment.py — EmailTriageEnv: the core OpenEnv-compliant environment.

Interface:
    reset(task_id)         → Observation
    step(action)           → StepResult(observation, reward, done, info)
    state()                → dict
    grade()                → dict  (final grading, call after done=True)
"""
from __future__ import annotations
import copy
from typing import Dict, Any, List, Optional

from env.models import (
    Action, EmailMessage, EmailLabel, Observation, Reward, StepResult,
)
from env.fixtures import EASY_INBOX, MEDIUM_INBOX, HARD_INBOX
from env import graders

MAX_STEPS = 60  # prevent infinite loops — hard penalty beyond this

TASK_INBOXES = {
    "task_easy":   EASY_INBOX,
    "task_medium": MEDIUM_INBOX,
    "task_hard":   HARD_INBOX,
}

VALID_LABELS: set[str] = {"work", "personal", "spam", "newsletter", "urgent"}


class EmailTriageEnv:
    """
    OpenEnv-compliant email triage environment.

    Observation, Action, and Reward are typed Pydantic models.
    Rewards are shaped — partial credit is given throughout the episode.
    """

    def __init__(self):
        self._task_id: str = ""
        self._inbox: List[EmailMessage] = []
        self._current_email: Optional[EmailMessage] = None
        self._step_count: int = 0
        self._cumulative_reward: float = 0.0
        self._done: bool = False

    # -----------------------------------------------------------------------
    # OpenEnv interface
    # -----------------------------------------------------------------------

    def reset(self, task_id: str = "task_easy") -> Observation:
        """Reset environment for a given task. Returns initial observation."""
        if task_id not in TASK_INBOXES:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {list(TASK_INBOXES)}")
        self._task_id = task_id
        self._inbox = copy.deepcopy(TASK_INBOXES[task_id])
        self._current_email = None
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._done = False
        return self._observe(message=f"Inbox loaded for '{task_id}'. {len(self._inbox)} emails await.")

    def step(self, action: Action) -> StepResult:
        """Apply an action, return (observation, reward, done, info)."""
        if self._done:
            return StepResult(
                observation=self._observe(message="Episode already finished."),
                reward=Reward(value=0.0, reason="Episode done", cumulative=self._cumulative_reward),
                done=True,
                info={"warning": "step() called after done=True"},
            )

        self._step_count += 1

        # Hard loop penalty
        if self._step_count > MAX_STEPS:
            self._done = True
            step_penalty = -0.5
            self._cumulative_reward += step_penalty
            return StepResult(
                observation=self._observe(message="Too many steps — episode terminated."),
                reward=Reward(value=step_penalty, reason="Exceeded max steps", cumulative=self._cumulative_reward),
                done=True,
                info={"error": "max_steps_exceeded"},
            )

        reward_value, reward_reason, done = self._apply_action(action)
        self._cumulative_reward = round(self._cumulative_reward + reward_value, 4)
        self._done = done

        return StepResult(
            observation=self._observe(),
            reward=Reward(value=reward_value, reason=reward_reason, cumulative=self._cumulative_reward),
            done=done,
            info={"step": self._step_count, "task_id": self._task_id},
        )

    def state(self) -> Dict[str, Any]:
        """Return serialisable current state."""
        return {
            "task_id": self._task_id,
            "step_count": self._step_count,
            "done": self._done,
            "cumulative_reward": self._cumulative_reward,
            "inbox": [e.model_dump() for e in self._inbox],
            "current_email_id": self._current_email.id if self._current_email else None,
        }

    def grade(self) -> Dict[str, Any]:
        """Run the task-specific grader and return score + breakdown."""
        grader_map = {
            "task_easy":   graders.grade_easy,
            "task_medium": graders.grade_medium,
            "task_hard":   graders.grade_hard,
        }
        grader = grader_map.get(self._task_id)
        if grader is None:
            return {"score": 0.0, "error": "No grader for task"}
        return grader(self._inbox)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _observe(self, message: str = "") -> Observation:
        visible_inbox = [e for e in self._inbox if not e.is_deleted]
        return Observation(
            inbox=visible_inbox,
            current_email=self._current_email,
            labels=list(VALID_LABELS),
            step_count=self._step_count,
            task_id=self._task_id,
            message=message,
        )

    def _get_email(self, email_id: str) -> Optional[EmailMessage]:
        return next((e for e in self._inbox if e.id == email_id), None)

    def _apply_action(self, action: Action) -> tuple[float, str, bool]:
        """Returns (reward_value, reason, done)."""
        t = action.action_type

        # --- done ---
        if t == "done":
            grading = self.grade()
            final_score = grading.get("score", 0.0)
            bonus = round(final_score * 0.2, 4)  # up to +0.2 bonus for finishing
            return bonus, f"Episode complete. Final grader score: {final_score}", True

        # Require email_id for all other actions
        email = self._get_email(action.email_id) if action.email_id else None
        if email is None and t != "done":
            return -0.05, f"Unknown email_id '{action.email_id}'", False

        if email and email.is_deleted:
            return -0.02, "Email already deleted", False

        # --- open_email ---
        if t == "open_email":
            email.is_open = True
            self._current_email = email
            return 0.0, f"Opened email '{email.id}'", False

        # --- label_email ---
        if t == "label_email":
            if action.label not in VALID_LABELS:
                return -0.05, f"Invalid label '{action.label}'", False
            email.label = action.label
            if email.label == email.true_label:
                return 0.1, f"Correct label '{action.label}' for {email.id}", False
            else:
                return -0.05, f"Wrong label '{action.label}' (expected '{email.true_label}')", False

        # --- reply_email ---
        if t == "reply_email":
            if not email.requires_reply:
                return -0.02, "This email doesn't require a reply", False
            if not action.body or len(action.body.split()) < 5:
                return -0.05, "Reply too short", False
            from env.graders import _reply_quality_score
            q = _reply_quality_score(action.body)
            email.reply_body = action.body
            reward = round(q * 0.15, 4)  # max +0.15 per reply
            return reward, f"Reply quality score: {q:.2f}", False

        # --- delete_email ---
        if t == "delete_email":
            if email.is_spam:
                email.is_deleted = True
                return 0.08, f"Correctly deleted spam email {email.id}", False
            else:
                email.is_deleted = True
                return -0.15, f"Deleted non-spam email {email.id} — penalty!", False

        # --- unsubscribe ---
        if t == "unsubscribe":
            if email.is_newsletter:
                email.is_unsubscribed = True
                return 0.05, f"Unsubscribed from newsletter {email.id}", False
            else:
                return -0.02, "Email is not a newsletter", False

        # --- flag_urgent ---
        if t == "flag_urgent":
            if email.true_label == "urgent":
                email.is_flagged_urgent = True
                return 0.12, f"Correctly flagged urgent email {email.id}", False
            else:
                email.is_flagged_urgent = True
                return -0.05, f"Incorrectly flagged {email.id} as urgent", False

        # --- archive_email ---
        if t == "archive_email":
            email.is_archived = True
            # Small reward if not spam (spam should be deleted, not archived)
            if email.is_spam:
                return -0.02, "Should delete spam, not archive it", False
            return 0.02, f"Archived email {email.id}", False

        return 0.0, "Unknown action", False
