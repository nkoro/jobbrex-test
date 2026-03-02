# backend_api.py — EvalOps FastAPI backend for Railway
"""
Endpoints:
  GET  /health          — liveness probe
  POST /route           — KA2L routing decision only (no LLM)
  POST /agent           — full agent pipeline (precheck → KA2L → Groq)
  POST /evals           — trigger eval loop in background
  GET  /evals/status    — poll eval loop status
  GET  /results         — return latest eval_results.json
  GET  /policy          — return current policy text
  POST /policy          — append a new rule to policy
"""

import json
import os
import uuid
from datetime import datetime

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

app = FastAPI(
    title="EvalOps API",
    description="Flight agent evaluation backend — JobRex EvalOps Lab",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Replit origin will be added via CORS_ORIGINS env var in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

POLICY_FILE  = os.path.join(os.path.dirname(__file__), "flight_policy.md")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "eval_results.json")


# ── Request / Response models ─────────────────────────────────────────────────

class PromptRequest(BaseModel):
    prompt: str

class PolicyRequest(BaseModel):
    rule: str
    author: str = "analyst"


# ── Background eval state ─────────────────────────────────────────────────────

_eval_status: dict = {
    "running": False,
    "run_id": None,
    "last_completed": None,
    "last_error": None,
}


def _run_evals_bg(run_id: str) -> None:
    """Background task: runs the full eval pipeline."""
    try:
        from run_evals import run_eval_loop
        run_eval_loop()
        _eval_status["last_completed"] = datetime.now().isoformat()
        _eval_status["last_error"] = None
    except Exception as e:
        _eval_status["last_error"] = str(e)
    finally:
        _eval_status["running"] = False


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["infra"])
def health():
    return {"status": "ok", "time": datetime.now().isoformat()}


@app.post("/route", tags=["agent"])
def route_prompt(req: PromptRequest):
    """Run only the KA2L routing decision — no LLM call."""
    try:
        from ka2l_router import route as ka2l_route
        return ka2l_route(req.prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent", tags=["agent"])
def run_agent(req: PromptRequest):
    """Full pipeline: policy pre-check → KA2L routing → Groq agent."""
    try:
        from flight_agent_crew import run_flight_crew
        result = run_flight_crew(req.prompt)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evals", tags=["evals"])
def trigger_evals(background_tasks: BackgroundTasks):
    """Kick off the eval loop asynchronously. Poll /evals/status for progress."""
    if _eval_status["running"]:
        return {"status": "already_running", "run_id": _eval_status["run_id"]}
    run_id = str(uuid.uuid4())[:8]
    _eval_status["running"] = True
    _eval_status["run_id"] = run_id
    _eval_status["last_error"] = None
    background_tasks.add_task(_run_evals_bg, run_id)
    return {"status": "started", "run_id": run_id}


@app.get("/evals/status", tags=["evals"])
def evals_status():
    return _eval_status


@app.get("/results", tags=["evals"])
def get_results():
    try:
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/policy", tags=["policy"])
def get_policy():
    try:
        with open(POLICY_FILE, "r") as f:
            return {"policy": f.read()}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Policy file not found")


@app.post("/policy", tags=["policy"])
def append_policy(req: PolicyRequest):
    """Append a new rule to flight_policy.md with timestamp + author."""
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        # Backup current policy
        backup_path = POLICY_FILE.replace(".md", f"_{ts.replace(':', '-').replace(' ', '_')}.bak")
        with open(POLICY_FILE, "r") as f:
            current = f.read()
        with open(backup_path, "w") as f:
            f.write(current)
        # Append rule
        with open(POLICY_FILE, "a") as f:
            f.write(f"\n- **[{ts}] [{req.author}]:** {req.rule}")
        return {"status": "ok", "appended": req.rule, "backup": backup_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Dev server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend_api:app", host="0.0.0.0", port=port, reload=True)
