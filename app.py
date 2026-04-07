"""
app.py — FastAPI server exposing the EmailTriageEnv as a REST API.
Runs as a Hugging Face Space (Gradio SDK = docker) or standalone Docker container.

Endpoints:
  POST /reset         body: {"task_id": "task_easy"}
  POST /step          body: Action JSON
  GET  /state
  GET  /grade
  GET  /health
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
    description="A realistic email triage benchmark environment implementing the OpenEnv interface.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (single-instance HF Space)
_sessions: Dict[str, EmailTriageEnv] = {}


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "task_easy"
    session_id: str | None = None


class ResetResponse(BaseModel):
    session_id: str
    observation: dict
    message: str


class StepRequest(BaseModel):
    session_id: str
    action: Action


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "EmailTriageEnv"}


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest):
    session_id = req.session_id or str(uuid.uuid4())
    env = EmailTriageEnv()
    obs = env.reset(req.task_id)
    _sessions[session_id] = env
    return ResetResponse(
        session_id=session_id,
        observation=obs.model_dump(),
        message=f"Session '{session_id}' started for task '{req.task_id}'.",
    )


@app.post("/step")
def step(req: StepRequest) -> Dict[str, Any]:
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{req.session_id}' not found. Call /reset first.")
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
        "version": "1.0.0",
        "tasks": ["task_easy", "task_medium", "task_hard"],
        "docs": "/docs",
        "openenv_spec": "/openenv.yaml",
    }


@app.get("/openenv.yaml")
def openenv_yaml():
    import yaml, pathlib  # noqa: E401
    content = pathlib.Path("openenv.yaml").read_text()
    return {"content": content}
