from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import jinsie_agent_platform as platform
from jinsie_agent_platform.runner import workflow_runner

app = FastAPI(title="Jinsie RAG Support System")


class RagRunRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


def _platform_prompt_path() -> str:
    """
    Resolve the platform prompt path regardless of current working directory.
    platform.__file__ -> .../jinsie-ai-agent-platform/jinsie_agent_platform/__init__.py
    """
    repo_root = Path(platform.__file__).resolve().parents[1]
    prompt = repo_root / "app" / "prompts" / "system" / "agent_system.md"
    return str(prompt)


@app.post("/v1/rag/run")
def rag_run(req: RagRunRequest):
    try:
        result = workflow_runner(
            req.query,
            debug=False,
            prompt_path=_platform_prompt_path(),
        )
        return {"answer": result}
    except Exception as e:
        # 先让错误可见，方便你对账；后面我们再换成更“产品化”的错误结构
        raise HTTPException(status_code=500, detail=str(e))
