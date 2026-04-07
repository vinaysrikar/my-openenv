"""
graders.py — Programmatic graders for each task (score 0.0 – 1.0).

Graders are deterministic and operate on the final environment state.
"""
from __future__ import annotations
import re
from typing import List
from env.models import EmailMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROFESSIONAL_REPLY_MIN_WORDS = 15
URGENCY_KEYWORDS = re.compile(
    r"(acknowledge|on.it|immediately|looking into|received|understood|will act|responding)",
    re.I,
)


def _reply_quality_score(body: str, context: str = "") -> float:
    """Heuristic reply quality: length, tone, acknowledgement."""
    if not body or not body.strip():
        return 0.0
    words = body.split()
    if len(words) < PROFESSIONAL_REPLY_MIN_WORDS:
        return 0.3
    score = 0.5
    if any(w in body.lower() for w in ["thank", "regards", "best", "sincerely", "hi", "hello", "dear"]):
        score += 0.2
    if len(words) >= 30:
        score += 0.2
    if "?" in body or any(w in body.lower() for w in ["let me know", "please", "confirm", "happy to"]):
        score += 0.1
    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Task 1 — Easy: Basic Label Triage
# ---------------------------------------------------------------------------

def grade_easy(emails: List[EmailMessage]) -> dict:
    """Score: fraction of emails with correct label."""
    correct = sum(1 for e in emails if e.label == e.true_label)
    score = correct / len(emails)
    return {
        "score": round(score, 4),
        "correct_labels": correct,
        "total_emails": len(emails),
        "details": {e.id: {"got": e.label, "expected": e.true_label, "ok": e.label == e.true_label} for e in emails},
    }


# ---------------------------------------------------------------------------
# Task 2 — Medium: Triage + Reply
# ---------------------------------------------------------------------------

def grade_medium(emails: List[EmailMessage]) -> dict:
    """Score: 60% label accuracy + 40% reply quality on emails requiring reply."""
    label_score = grade_easy(emails)["score"]

    reply_emails = [e for e in emails if e.requires_reply]
    if reply_emails:
        reply_scores = [_reply_quality_score(e.reply_body or "") for e in reply_emails]
        reply_score = sum(reply_scores) / len(reply_scores)
    else:
        reply_score = 1.0

    # Penalty: deleting an email that wasn't spam
    non_spam_deleted = sum(1 for e in emails if e.is_deleted and not e.is_spam)
    penalty = min(non_spam_deleted * 0.1, 0.3)

    final = (0.6 * label_score + 0.4 * reply_score) - penalty

    return {
        "score": round(max(0.0, final), 4),
        "label_score": round(label_score, 4),
        "reply_score": round(reply_score, 4),
        "penalty": round(penalty, 4),
        "reply_details": {
            e.id: {"has_reply": bool(e.reply_body), "reply_score": _reply_quality_score(e.reply_body or "")}
            for e in reply_emails
        },
    }


# ---------------------------------------------------------------------------
# Task 3 — Hard: Full Inbox Zero
# ---------------------------------------------------------------------------

def grade_hard(emails: List[EmailMessage]) -> dict:
    """
    Composite score across five criteria:
      - Label accuracy        (25%)
      - Urgent flags          (20%)
      - Reply quality         (25%)
      - Spam deleted          (15%)
      - Newsletters unsubscribed (15%)
    Penalties:
      - Deleting non-spam email: -0.15 each
      - Missing an urgent flag:  -0.10 each (in addition to urgency score)
    """
    # 1. Label accuracy
    label_correct = sum(1 for e in emails if e.label == e.true_label)
    label_score = label_correct / len(emails)

    # 2. Urgent flags — emails with true_label == "urgent" must be flagged
    urgent_emails = [e for e in emails if e.true_label == "urgent"]
    flagged_correctly = sum(1 for e in urgent_emails if e.is_flagged_urgent)
    urgent_score = (flagged_correctly / len(urgent_emails)) if urgent_emails else 1.0

    # 3. Replies
    reply_emails = [e for e in emails if e.requires_reply]
    if reply_emails:
        reply_scores = [_reply_quality_score(e.reply_body or "") for e in reply_emails]
        reply_score = sum(reply_scores) / len(reply_scores)
    else:
        reply_score = 1.0

    # 4. Spam deletion
    spam_emails = [e for e in emails if e.is_spam]
    spam_deleted = sum(1 for e in spam_emails if e.is_deleted)
    spam_score = (spam_deleted / len(spam_emails)) if spam_emails else 1.0

    # 5. Newsletter unsubscribe
    newsletters = [e for e in emails if e.is_newsletter]
    unsub_count = sum(1 for e in newsletters if e.is_unsubscribed)
    newsletter_score = (unsub_count / len(newsletters)) if newsletters else 1.0

    # Penalties
    wrong_deletes = sum(1 for e in emails if e.is_deleted and not e.is_spam)
    missed_urgent = sum(1 for e in urgent_emails if not e.is_flagged_urgent)
    penalty = (wrong_deletes * 0.15) + (missed_urgent * 0.10)

    composite = (
        0.25 * label_score
        + 0.20 * urgent_score
        + 0.25 * reply_score
        + 0.15 * spam_score
        + 0.15 * newsletter_score
    ) - penalty

    return {
        "score": round(max(0.0, min(1.0, composite)), 4),
        "label_score": round(label_score, 4),
        "urgent_score": round(urgent_score, 4),
        "reply_score": round(reply_score, 4),
        "spam_score": round(spam_score, 4),
        "newsletter_score": round(newsletter_score, 4),
        "penalty": round(penalty, 4),
        "breakdown": {
            "correct_labels": label_correct,
            "urgent_flagged": f"{flagged_correctly}/{len(urgent_emails)}",
            "spam_deleted": f"{spam_deleted}/{len(spam_emails)}",
            "newsletters_unsubbed": f"{unsub_count}/{len(newsletters)}",
            "wrong_deletes": wrong_deletes,
        },
    }
