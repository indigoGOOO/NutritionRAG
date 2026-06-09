"""FastAPI application entrypoint."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from src.api.agent_api import router as agent_router
from src.api.chat_page import CHAT_PAGE

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Nutrition RAG Agent API",
    version="0.1.0",
    description="FastAPI backend and lightweight chat UI for the nutrition RAG agent.",
)
app.include_router(agent_router)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return CHAT_PAGE
