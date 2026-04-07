"""
app.py — FastAPI server exposing the EmailTriageEnv as a REST API.
"""

from __future__ import annotations
import uuid
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import EmailTriageEnv
from env.models import Action, StepResult

app = FastAPI(
    title="EmailTriageEnv — OpenEnv",
    description="Email triage environment",
    version="1.0.1",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session store
_sessions: Dict[str, EmailTriageEnv] = {}

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "task_easy"
    session_id: str | None = None


class StepRequest(BaseModel):
    session_id: str
    action: Action


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# ✅ FIXED RESET (IMPORTANT)
@app.post("/reset")
def reset(req: ResetRequest):
    try:
        session_id = req.session_id or str(uuid.uuid4())

        env = EmailTriageEnv()
        obs = env.reset(req.task_id)

        _sessions[session_id] = env

        return {
            "session_id": session_id,
            "observation": obs.model_dump(),
            "message": "Environment reset successful"
        }

    except Exception as e:
        return {
            "error": str(e)
        }


@app.post("/step")
def step(req: StepRequest) -> Dict[str, Any]:
    env = _sessions.get(req.session_id)

    if env is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Call /reset first."
        )

    result: StepResult = env.step(req.action)
    return result.model_dump()


@app.get("/state/{session_id}")
def state(session_id: str) -> Dict[str, Any]:
    env = _sessions.get(session_id)

    if env is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    return env.state()


@app.get("/grade/{session_id}")
def grade(session_id: str) -> Dict[str, Any]:
    env = _sessions.get(session_id)

    if env is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    return env.grade()


@app.get("/")
def root():
    return {
        "name": "EmailTriageEnv",
        "version": "1.0.1",
        "tasks": ["task_easy", "task_medium", "task_hard"]
    }


@app.get("/openenv.yaml")
def openenv_yaml():
    import pathlib
    return {
        "content": pathlib.Path("openenv.yaml").read_text()
    }
