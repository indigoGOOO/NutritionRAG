import sys
from pathlib import Path

import asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api import agent_api
from src.api.agent_api import ChatRequest, SaveUserContentRequest, chat, save_user_content
from src.user_content.service import SavedUserContent


class FakeContentService:
    def save_text(self, **kwargs):
        return SavedUserContent(
            saved=True,
            content_type=kwargs.get("content_type") or "recipe",
            title=kwargs.get("title") or "Saved recipe",
            source_doc_id="doc1",
            storage={"chunks": 1, "kv_pairs": 1, "triples": 1},
            classification={"content_type": kwargs.get("content_type") or "recipe"},
            errors=[],
        )


def test_save_user_content_endpoint(monkeypatch):
    async def fake_get_content_service():
        return FakeContentService()

    monkeypatch.setattr(agent_api, "get_content_service", fake_get_content_service)

    response = asyncio.run(save_user_content(SaveUserContentRequest(
        user_id="u1",
        session_id="s1",
        content_type="recipe",
        title="番茄炒蛋",
        content="食材：番茄、鸡蛋。步骤1：炒。",
    )))

    assert response.saved is True
    assert response.content_type == "recipe"
    assert response.storage["chunks"] == 1


def test_chat_intercepts_explicit_save_request(monkeypatch):
    async def fake_get_content_service():
        return FakeContentService()

    async def fail_get_agent():
        raise AssertionError("chat save should not call agent")

    monkeypatch.setattr(agent_api, "get_content_service", fake_get_content_service)
    monkeypatch.setattr(agent_api, "get_agent", fail_get_agent)

    response = asyncio.run(chat(ChatRequest(
        user_id="u1",
        session_id="s1",
        message="保存这个菜谱：食材鸡蛋，步骤1 打散。",
    )))

    assert response.intent == "content_save"
    assert response.saved_content["saved"] is True
    assert response.saved_content["content_type"] == "recipe"
