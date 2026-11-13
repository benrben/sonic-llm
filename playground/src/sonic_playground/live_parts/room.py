"""How many agents a test can run at once on the loaded model and this Mac's memory (the Max button of the page)."""

from __future__ import annotations

from .agents import doc_need
from .config import MAX_AGENTS
from .scenarios import BURST_OUT, SCENARIOS, TURNS_NEED
from .session import Session

BURST_PROMPT = 128  # tokens, more than the built-in burst prompts take
NO_NEED = 1024  # tokens an agent is given when there are no documents to size it by (the engine's smallest context)
OFFERS_AGENTS = {x["id"] for x in SCENARIOS if x.get("agents")}


def agent_room(s: Session, scenario: str, files: list[str]) -> dict:
    """{max, tokens_per_agent}: `max` agents at once for `scenario` reading `files` (empty: the biggest documents of the library),
    each holding `tokens_per_agent` tokens (its document, its question and room for the answer). Tests without agents give max 1."""
    if scenario not in OFFERS_AGENTS:
        return {"max": 1, "tokens_per_agent": None}
    docs = files if scenario != "followup" else [f for f, _ in (s.readers or {"docs": []})["docs"]]
    need = BURST_OUT + BURST_PROMPT if scenario == "burst" else TURNS_NEED if scenario == "turns" else doc_need(docs[:1] if scenario == "shared" else docs, MAX_AGENTS, None)
    return {"max": s.max_agents("swarm" if scenario == "followup" else scenario, need or NO_NEED), "tokens_per_agent": need}
