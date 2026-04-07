#!/usr/bin/env python3
"""
baseline.py — Runs an LLM agent against all three EmailTriageEnv tasks
using the OpenAI API client (compatible with any OpenAI-compatible endpoint).

Usage:
    export OPENAI_API_KEY=sk-...
    export OPENAI_BASE_URL=https://api.openai.com/v1   # optional
    export OPENAI_MODEL=gpt-4o-mini                    # optional, default gpt-4o-mini

    python baseline.py
"""
from __future__ import annotations
import json
import os
import sys
import textwrap
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

# Append project root so imports work regardless of cwd
sys.path.insert(0, os.path.dirname(__file__))
from env import EmailTriageEnv
from env.models import Action


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY   = os.environ.get("OPENAI_API_KEY", "")
BASE_URL  = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL     = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
MAX_TURNS = 40

TASKS = ["task_easy", "task_medium", "task_hard"]


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert email triage assistant. Your job is to process an inbox efficiently.

At each step you must output a single JSON action object and nothing else.

Available action types:
  {"action_type": "open_email",    "email_id": "<id>"}
  {"action_type": "label_email",   "email_id": "<id>", "label": "<work|personal|spam|newsletter|urgent>"}
  {"action_type": "reply_email",   "email_id": "<id>", "body": "<reply text>"}
  {"action_type": "delete_email",  "email_id": "<id>"}
  {"action_type": "unsubscribe",   "email_id": "<id>"}
  {"action_type": "flag_urgent",   "email_id": "<id>"}
  {"action_type": "archive_email", "email_id": "<id>"}
  {"action_type": "done"}

Strategy:
1. First open and read each email carefully.
2. Label every email.
3. Flag emails that are CRITICAL/URGENT.
4. Delete spam emails.
5. Unsubscribe from newsletters.
6. Reply to emails that need a response (professional, >20 words).
7. Archive remaining non-urgent emails.
8. When the inbox is fully processed, call {"action_type": "done"}.

Output ONLY valid JSON. No explanation, no markdown fences.
""").strip()


def obs_to_text(obs_dict: dict) -> str:
    """Convert observation dict to a text description for the LLM."""
    inbox = obs_dict.get("inbox", [])
    lines = [f"STEP {obs_dict['step_count']} | Task: {obs_dict['task_id']}"]
    if obs_dict.get("message"):
        lines.append(f"System: {obs_dict['message']}")
    lines.append(f"\nINBOX ({len(inbox)} emails):")
    for e in inbox:
        flags = []
        if e.get("label", "unlabeled") != "unlabeled":
            flags.append(f"label={e['label']}")
        if e.get("is_flagged_urgent"):
            flags.append("FLAGGED_URGENT")
        if e.get("is_unsubscribed"):
            flags.append("UNSUBBED")
        if e.get("reply_body"):
            flags.append("REPLIED")
        if e.get("is_archived"):
            flags.append("ARCHIVED")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        lines.append(f"  [{e['id']}] From: {e['sender']} | Subject: {e['subject']}{flag_str}")
        if e.get("is_open"):
            body_preview = e.get("body", "")[:300]
            lines.append(f"       BODY: {body_preview}")
    if obs_dict.get("current_email"):
        ce = obs_dict["current_email"]
        lines.append(f"\nCURRENTLY OPEN: [{ce['id']}] {ce['subject']}")
        lines.append(f"  FROM: {ce['sender']}")
        lines.append(f"  BODY: {ce.get('body', '')}")
    return "\n".join(lines)


def parse_action(text: str) -> Action | None:
    """Parse LLM output into an Action. Returns None on failure."""
    text = text.strip()
    # Strip markdown fences if the model disobeys
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith("```")
        )
    try:
        data = json.loads(text)
        return Action(**data)
    except Exception as e:
        print(f"    [parse error] {e} | raw: {text[:120]}")
        return None


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task_id: str) -> dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"  TASK: {task_id}")
    print(f"{'='*60}")

    env = EmailTriageEnv()
    obs = env.reset(task_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    cumulative_reward = 0.0
    step = 0

    for _ in range(MAX_TURNS):
        obs_text = obs_to_text(obs.model_dump())
        messages.append({"role": "user", "content": obs_text})

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
            )
        except Exception as e:
            print(f"  [API error] {e}")
            break

        raw = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": raw})

        action = parse_action(raw)
        if action is None:
            # Inject a parse-failure nudge and retry once
            messages.append({"role": "user", "content": "Output was not valid JSON. Respond with ONLY a JSON action."})
            continue

        result = env.step(action)
        obs = result.observation
        cumulative_reward = result.reward.cumulative
        step += 1

        print(f"  step {step:02d} | action={action.action_type} email={action.email_id or ''} "
              f"| reward={result.reward.value:+.3f} ({result.reward.reason[:60]})")

        if result.done:
            break

    grading = env.grade()
    final_score = grading.get("score", 0.0)
    print(f"\n  GRADER SCORE : {final_score:.4f}")
    print(f"  CUM. REWARD  : {cumulative_reward:.4f}")
    print(f"  GRADING      : {json.dumps(grading, indent=2)}")

    return {
        "task_id": task_id,
        "grader_score": final_score,
        "cumulative_reward": cumulative_reward,
        "steps": step,
        "grading_breakdown": grading,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    print(f"Model     : {MODEL}")
    print(f"Base URL  : {BASE_URL}")
    print(f"Max turns : {MAX_TURNS}")

    results = []
    for task_id in TASKS:
        r = run_task(client, task_id)
        results.append(r)

    print(f"\n{'='*60}")
    print("  BASELINE SUMMARY")
    print(f"{'='*60}")
    total = 0.0
    for r in results:
        score = r["grader_score"]
        total += score
        print(f"  {r['task_id']:<20} grader_score={score:.4f}  steps={r['steps']}")
    avg = total / len(results)
    print(f"\n  AVERAGE SCORE : {avg:.4f}")
    print(f"{'='*60}\n")

    # Write results to file for CI / HF Space display
    with open("baseline_results.json", "w") as f:
        json.dump({"results": results, "average_score": round(avg, 4)}, f, indent=2)
    print("Results written to baseline_results.json")


if __name__ == "__main__":
    main()
