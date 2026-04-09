import os
import textwrap
from typing import List, Optional

from openai import OpenAI
from env.environment import EmailTriageEnv
from env.models import Action

# Environment variables
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TASK_NAME = os.getenv("EMAIL_TRIAGE_TASK", "task_easy")
BENCHMARK = "email_triage"
MAX_STEPS = 60
TEMPERATURE = 0.7
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.1

SYSTEM_PROMPT = textwrap.dedent("""
    You are an email triage assistant managing an inbox.
    At each step, you will see a list of emails and must choose ONE action.

    Respond with ONLY a JSON object in one of these formats:

    {"action_type": "open_email", "email_id": "<id>"}
    {"action_type": "label_email", "email_id": "<id>", "label": "<work|personal|spam|newsletter|urgent>"}
    {"action_type": "reply_email", "email_id": "<id>", "body": "<reply text at least 5 words>"}
    {"action_type": "delete_email", "email_id": "<id>"}
    {"action_type": "unsubscribe", "email_id": "<id>"}
    {"action_type": "flag_urgent", "email_id": "<id>"}
    {"action_type": "archive_email", "email_id": "<id>"}
    {"action_type": "done"}

    No explanation. Just the JSON object.
""").strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def get_model_action(client: OpenAI, observation: str) -> dict:
    import json
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": observation},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return json.loads(text)
    except Exception as exc:
        print(f"[DEBUG] Model request failed or bad JSON: {exc}", flush=True)
        return {"action_type": "done"}


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = EmailTriageEnv()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(TASK_NAME)
        observation_str = str(obs.model_dump())
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action_dict = get_model_action(client, observation_str)
            action = Action(**action_dict)

            result = env.step(action)

            reward = result.reward.value
            done = result.done
            error = result.info.get("error", None) if isinstance(result.info, dict) else None
            observation_str = str(result.observation.model_dump())

            rewards.append(float(reward))
            steps_taken = step

            log_step(step=step, action=str(action_dict), reward=float(reward), done=done, error=error)

            if done:
                break

        score = sum(rewards) / max(len(rewards), 1)
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Exception during run: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
