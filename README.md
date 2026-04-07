---
title: EmailTriageEnv — OpenEnv
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
  - environment
  - email
  - agent
  - benchmark
license: mit
---

# 📧 EmailTriageEnv

**A realistic email triage benchmark environment implementing the [OpenEnv](https://openenv.dev) interface.**

Agents must process an inbox the way a human knowledge-worker would — labelling messages, replying professionally, deleting spam, unsubscribing from newsletters, and flagging emergencies.

---

## Why this environment?

Email triage is one of the most universal, high-frequency real-world tasks. It requires:

- **Reading comprehension** — understanding tone, urgency, and sender context
- **Classification** — work vs personal vs spam vs newsletter vs urgent
- **Prioritisation** — identifying what needs immediate action
- **Generation** — drafting professional, context-aware replies
- **Sequential reasoning** — efficiently processing a queue without thrashing

This makes it an excellent benchmark for general-purpose language agents.

---

## Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `open_email` | `email_id` | Read the full body of an email |
| `label_email` | `email_id`, `label` | Assign one of: `work`, `personal`, `spam`, `newsletter`, `urgent` |
| `reply_email` | `email_id`, `body` | Draft a reply (only for emails requiring response) |
| `delete_email` | `email_id` | Permanently delete (use only for spam) |
| `unsubscribe` | `email_id` | Unsubscribe from a newsletter |
| `flag_urgent` | `email_id` | Mark as requiring immediate attention |
| `archive_email` | `email_id` | Archive a processed email |
| `done` | — | Signal episode completion |

## Observation Space

Each observation contains:
- `inbox`: list of visible `EmailMessage` objects (id, sender, subject, body, label, flags)
- `current_email`: the currently open email (if any)
- `labels`: valid label choices
- `step_count`: steps taken so far
- `task_id`: which task is running
- `message`: human-readable system feedback

---

## Tasks

### Task 1 — Basic Triage (`task_easy`) 🟢

> **5 emails | Score: label accuracy only**

Label all 5 emails correctly. One of each category: work, personal, spam, newsletter, urgent.

- **Grader**: `correct_labels / total_emails`
- **Expected difficulty**: Easy — labels are clear from subject/sender
- **Baseline score**: ~0.80

---

### Task 2 — Triage + Reply (`task_medium`) 🟡

> **7 emails | Score: 60% labels + 40% reply quality**

Label all emails AND draft professional replies to the 2 emails that require a response.

- **Grader**: weighted label accuracy + heuristic reply quality (length, tone, professionalism)
- **Penalty**: −0.10 per non-spam email deleted
- **Expected difficulty**: Medium — requires reading comprehension and generation
- **Baseline score**: ~0.65

---

### Task 3 — Full Inbox Zero (`task_hard`) 🔴

> **10 emails | Score: 5-component composite**

Full processing of a realistic mixed inbox:
- Label all 10 emails (25%)
- Flag 2 urgent alerts (20%)
- Reply to 5 emails requiring response (25%)
- Delete 2 spam emails (15%)
- Unsubscribe from 3 newsletters (15%)

- **Grader**: weighted composite, penalties for missed urgents and wrong deletes
- **Expected difficulty**: Hard — requires multi-step planning and generation quality
- **Baseline score**: ~0.45

---

## Reward Function

Rewards are **shaped** — the agent receives signal at every step, not just at the end:

| Action | Reward |
|--------|--------|
| Correct label | +0.10 |
| Wrong label | −0.05 |
| Quality reply (good) | up to +0.15 |
| Reply too short | −0.05 |
| Delete spam | +0.08 |
| Delete non-spam | **−0.15** |
| Flag urgent (correct) | +0.12 |
| Flag urgent (wrong) | −0.05 |
| Unsubscribe newsletter | +0.05 |
| Exceed max steps | **−0.50** |
| Episode done (bonus) | up to +0.20 |

---

## REST API

The environment runs as a REST API. All endpoints accept/return JSON.

### Reset

```
POST /reset
{"task_id": "task_easy"}
→ {"session_id": "...", "observation": {...}}
```

### Step

```
POST /step
{"session_id": "...", "action": {"action_type": "open_email", "email_id": "e1"}}
→ {"observation": {...}, "reward": {...}, "done": false, "info": {...}}
```

### State / Grade

```
GET /state/{session_id}
GET /grade/{session_id}
```

Interactive docs available at `/docs`.

---

## Setup & Usage

### Local Python

```bash
git clone https://huggingface.co/spaces/<your-handle>/email-triage-env
cd email-triage-env
pip install -r requirements.txt

# Start the server
uvicorn app:app --reload --port 7860

# Or run the baseline agent
export OPENAI_API_KEY=sk-...
python baseline.py
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... email-triage-env
```

### Baseline Agent

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini       # default
python baseline.py
```

Results are written to `baseline_results.json`.

---

## Baseline Scores (gpt-4o-mini, temperature=0)

| Task | Grader Score | Steps |
|------|-------------|-------|
| task_easy | 0.80 | 12 |
| task_medium | 0.64 | 19 |
| task_hard | 0.47 | 38 |
| **Average** | **0.64** | — |

*Scores are reproducible with `OPENAI_MODEL=gpt-4o-mini` and `temperature=0`.*

---

## OpenEnv Compliance

This environment implements the full OpenEnv interface:

- ✅ Typed `Observation`, `Action`, `Reward` Pydantic models
- ✅ `step(action)` → `(observation, reward, done, info)`
- ✅ `reset(task_id)` → initial `Observation`
- ✅ `state()` → serialisable current state
- ✅ `openenv.yaml` metadata
- ✅ 3 tasks with programmatic graders (0.0 – 1.0)
- ✅ Shaped rewards throughout trajectory
- ✅ Dockerfile + HF Space deployment
- ✅ Baseline script using OpenAI API client

---

## License

MIT
