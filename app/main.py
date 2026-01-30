from fastapi import FastAPI
from pydantic import BaseModel

from jinsie_agent_platform.runner import workflow_runner

app = FastAPI(title="Jinsie RAG Support System")


class RagRunRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/rag/run")
def rag_run(req: RagRunRequest):
    # Phase-2 wiring: call platform runner (system repo -> platform repo)
    result = workflow_runner(
        req.query,
        debug=False,
        expected_steps=None,
        strict_degraded=False,
    )
    return {"answer": result}
