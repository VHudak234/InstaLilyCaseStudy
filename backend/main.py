"""FastAPI entry point. Single /chat endpoint the React frontend POSTs to."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import run_agent

app = FastAPI(title="PartSelect Agent")

# Allow the CRA dev server to call us from a different port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    result = run_agent([m.model_dump() for m in req.messages])
    return result
