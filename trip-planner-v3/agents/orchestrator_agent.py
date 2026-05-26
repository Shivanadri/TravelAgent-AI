import os
import httpx
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

_http = httpx.Client(verify=False)
_http_async = httpx.AsyncClient(verify=False)
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model="openai/gpt-oss-120b:free",
            temperature=0,
            max_tokens=2048,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            http_client=_http,
            http_async_client=_http_async,
        )
    return _llm


class OrchestratorDecision(BaseModel):
    approved:     bool       = Field(description="True to proceed, False to retry")
    retry_agents: list[str]  = Field(default_factory=list, description="Agents to retry: itinerary|budget|places")
    reason:       str        = Field(description="Short explanation of the decision")
    force_stop:   bool       = Field(default=False, description="True if max retries reached — approve as-is")


ORCHESTRATOR_SYSTEM = """
You are the orchestrator of a multi-agent Indian trip planning system.
Make the final quality call: approve the itinerary or send it back for a targeted revision.

────────────────────────────────────────────────
APPROVAL RULES
────────────────────────────────────────────────
  APPROVE if ALL of:
    • review_passed = True  (score ≥ 6)
    • Total estimate is within budget OR surplus_deficit ≥ –5% of total_budget
    • No critical issues listed

  RETRY if ANY of (and retry_count < 3):
    • review_passed = False (score < 6)
    • Total estimate exceeds budget by more than 10%
    • Critical issues present that a specific agent can fix

  FORCE STOP (approve as-is) if retry_count ≥ 3

────────────────────────────────────────────────
RETRY TARGET SELECTION
────────────────────────────────────────────────
  Budget issues          → retry "budget"
  Places / variety       → retry "places"
  Pacing / flow          → retry "itinerary"
  Multiple issues        → retry only the most impactful agent — do not retry all

────────────────────────────────────────────────
OUTPUT RULES
────────────────────────────────────────────────
  • approved     : true or false
  • retry_agents : list with exactly one agent name when retrying (e.g. ["itinerary"])
  • reason       : one clear sentence explaining the decision
  • force_stop   : true only when retry_count ≥ 3
"""


def run_orchestrator_agent(state: dict) -> dict:
    review      = state.get("review_status", {})
    eval_scores = state.get("eval_scores", {})
    budget_sum  = state.get("budget_summary", {})
    retry_count = state.get("retry_count", 0)

    print("\n" + "=" * 55)
    print("   TRIP PLANNER v3 — Agent 10: Orchestrator")
    print("=" * 55)

    if retry_count >= 3:
        print(f"\n  Max retries reached ({retry_count}/3). Approving as-is.\n")
        return {
            "orchestrator_decision": {
                "approved":    True,
                "retry_agents": [],
                "reason":      f"Max retries ({retry_count}) reached — proceeding with best available plan",
                "force_stop":  True,
            },
            "last_completed_node": "orchestrator_agent",
        }

    llm = _get_llm()
    orch_llm = llm.with_structured_output(OrchestratorDecision, method="function_calling")

    prompt = f"""
Review score    : {review.get('score', 0)}/10
Review passed   : {review.get('passed')}
Issues          : {'; '.join(review.get('issues', [])) or 'none'}
Warnings        : {'; '.join(review.get('warnings', [])) or 'none'}
Pacing score    : {eval_scores.get('pacing', '?')}/10
Variety score   : {eval_scores.get('variety', '?')}/10
Within budget   : {budget_sum.get('within_budget')}
Surplus/deficit : Rs.{budget_sum.get('surplus_deficit', 0):,}
Retry count     : {retry_count}/3
Suggested retry : {', '.join(review.get('retry_agents', [])) or 'none'}

Decide: approve the itinerary OR retry specific agents.
"""

    result: OrchestratorDecision = None
    for attempt in range(3):
        result = orch_llm.invoke([
            SystemMessage(content=ORCHESTRATOR_SYSTEM),
            HumanMessage(content=prompt),
        ])
        if result is not None:
            break
        print(f"  [orchestrator] structured output returned None, retrying ({attempt + 1}/3)...")
    if result is None:
        result = OrchestratorDecision(
            approved=True,
            retry_agents=[],
            reason="Orchestrator could not assess — approving as-is to avoid blocking.",
            force_stop=True,
        )

    action = "APPROVED" if result.approved else f"RETRY ({', '.join(result.retry_agents)})"
    print(f"\n  Decision: {action}")
    print(f"  Reason  : {result.reason}")

    if not result.approved:
        print(f"  Retries : {retry_count + 1}/3")

    print("=" * 55)

    new_retry = retry_count if result.approved else retry_count + 1

    return {
        "orchestrator_decision": {
            "approved":     result.approved,
            "retry_agents": result.retry_agents,
            "reason":       result.reason,
            "force_stop":   result.force_stop,
        },
        "retry_count":         new_retry,
        "last_completed_node": "orchestrator_agent",
    }
