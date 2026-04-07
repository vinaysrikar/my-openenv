"""
tests/test_environment.py — Unit tests for EmailTriageEnv.

Run with: pytest tests/ -v
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env import EmailTriageEnv
from env.models import Action
from env import graders


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def easy_env():
    env = EmailTriageEnv()
    env.reset("task_easy")
    return env

@pytest.fixture
def medium_env():
    env = EmailTriageEnv()
    env.reset("task_medium")
    return env

@pytest.fixture
def hard_env():
    env = EmailTriageEnv()
    env.reset("task_hard")
    return env


# ---------------------------------------------------------------------------
# reset() tests
# ---------------------------------------------------------------------------

def test_reset_returns_observation(easy_env):
    env = EmailTriageEnv()
    obs = env.reset("task_easy")
    assert obs.task_id == "task_easy"
    assert len(obs.inbox) == 5
    assert obs.step_count == 0

def test_reset_unknown_task():
    env = EmailTriageEnv()
    with pytest.raises(ValueError, match="Unknown task_id"):
        env.reset("task_nonexistent")

def test_reset_medium_has_7_emails(medium_env):
    env = EmailTriageEnv()
    obs = env.reset("task_medium")
    assert len(obs.inbox) == 7

def test_reset_hard_has_10_emails(hard_env):
    env = EmailTriageEnv()
    obs = env.reset("task_hard")
    assert len(obs.inbox) == 10


# ---------------------------------------------------------------------------
# state() tests
# ---------------------------------------------------------------------------

def test_state_serialisable(easy_env):
    import json
    s = easy_env.state()
    # Must be JSON-serialisable
    json.dumps(s)
    assert s["task_id"] == "task_easy"
    assert s["step_count"] == 0
    assert s["done"] is False


# ---------------------------------------------------------------------------
# step() — action tests
# ---------------------------------------------------------------------------

def test_open_email(easy_env):
    result = easy_env.step(Action(action_type="open_email", email_id="e1"))
    assert result.done is False
    assert result.observation.current_email is not None
    assert result.observation.current_email.id == "e1"
    assert result.reward.value == 0.0

def test_correct_label_reward(easy_env):
    result = easy_env.step(Action(action_type="label_email", email_id="e1", label="work"))
    assert result.reward.value == pytest.approx(0.1)

def test_wrong_label_penalty(easy_env):
    result = easy_env.step(Action(action_type="label_email", email_id="e1", label="spam"))
    assert result.reward.value == pytest.approx(-0.05)

def test_delete_spam_reward(easy_env):
    result = easy_env.step(Action(action_type="delete_email", email_id="e3"))  # e3 is spam
    assert result.reward.value == pytest.approx(0.08)
    # Deleted email should not appear in next observation
    assert all(e.id != "e3" for e in result.observation.inbox)

def test_delete_nonspam_penalty(easy_env):
    result = easy_env.step(Action(action_type="delete_email", email_id="e1"))  # e1 is work
    assert result.reward.value == pytest.approx(-0.15)

def test_flag_urgent_correct(easy_env):
    result = easy_env.step(Action(action_type="flag_urgent", email_id="e5"))  # e5 is urgent
    assert result.reward.value == pytest.approx(0.12)

def test_flag_urgent_wrong(easy_env):
    result = easy_env.step(Action(action_type="flag_urgent", email_id="e1"))  # e1 is work
    assert result.reward.value == pytest.approx(-0.05)

def test_unsubscribe_newsletter(easy_env):
    result = easy_env.step(Action(action_type="unsubscribe", email_id="e4"))  # e4 is newsletter
    assert result.reward.value == pytest.approx(0.05)

def test_unsubscribe_non_newsletter(easy_env):
    result = easy_env.step(Action(action_type="unsubscribe", email_id="e1"))
    assert result.reward.value < 0

def test_reply_too_short(medium_env):
    result = medium_env.step(Action(action_type="reply_email", email_id="m1", body="OK thanks"))
    assert result.reward.value < 0

def test_reply_quality(medium_env):
    body = (
        "Dear Alex, thank you for reaching out. The bank details on Invoice #4821 are correct. "
        "Please proceed with the payment. Let me know if you need anything else. Best regards."
    )
    result = medium_env.step(Action(action_type="reply_email", email_id="m1", body=body))
    assert result.reward.value > 0

def test_done_action_ends_episode(easy_env):
    result = easy_env.step(Action(action_type="done"))
    assert result.done is True

def test_step_after_done(easy_env):
    easy_env.step(Action(action_type="done"))
    result = easy_env.step(Action(action_type="open_email", email_id="e1"))
    assert result.done is True
    assert "already finished" in result.observation.message.lower()

def test_unknown_email_id_penalty(easy_env):
    result = easy_env.step(Action(action_type="open_email", email_id="NONEXISTENT"))
    assert result.reward.value < 0

def test_max_steps_terminates():
    env = EmailTriageEnv()
    env.reset("task_easy")
    # Exceed MAX_STEPS (60) by doing many no-ops
    for _ in range(61):
        r = env.step(Action(action_type="open_email", email_id="e1"))
        if r.done:
            assert r.reward.value < 0  # penalty
            return
    pytest.fail("Environment did not terminate after max steps")

def test_cumulative_reward_accumulates(easy_env):
    easy_env.step(Action(action_type="label_email", email_id="e1", label="work"))   # +0.10
    easy_env.step(Action(action_type="label_email", email_id="e3", label="spam"))   # +0.10
    result = easy_env.step(Action(action_type="label_email", email_id="e2", label="personal"))  # +0.10
    assert result.reward.cumulative == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# Grader tests
# ---------------------------------------------------------------------------

def test_easy_grader_perfect_score():
    from env.fixtures import EASY_INBOX
    import copy
    emails = copy.deepcopy(EASY_INBOX)
    for e in emails:
        e.label = e.true_label
    result = graders.grade_easy(emails)
    assert result["score"] == pytest.approx(1.0)

def test_easy_grader_zero_score():
    from env.fixtures import EASY_INBOX
    import copy
    emails = copy.deepcopy(EASY_INBOX)
    for e in emails:
        e.label = "work"  # all wrong (except the actual work email)
    result = graders.grade_easy(emails)
    assert result["score"] < 0.5

def test_hard_grader_returns_all_fields():
    from env.fixtures import HARD_INBOX
    result = graders.grade_hard(HARD_INBOX)
    for key in ["score", "label_score", "urgent_score", "reply_score", "spam_score", "newsletter_score"]:
        assert key in result

def test_grader_score_in_range():
    from env.fixtures import EASY_INBOX, MEDIUM_INBOX, HARD_INBOX
    for emails, grader in [
        (EASY_INBOX, graders.grade_easy),
        (MEDIUM_INBOX, graders.grade_medium),
        (HARD_INBOX, graders.grade_hard),
    ]:
        result = grader(emails)
        assert 0.0 <= result["score"] <= 1.0
