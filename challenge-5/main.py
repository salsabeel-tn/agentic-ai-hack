from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
from datetime import timedelta
from azure.identity.aio import DefaultAzureCredential

from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.contents import ChatMessageContent
from semantic_kernel.agents.open_ai.run_polling_options import RunPollingOptions
from azure.ai.agents.models import AzureAISearchQueryType, AzureAISearchTool, ListSortOrder, MessageRole
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings, AzureAIAgentThread, Agent, ChatCompletionAgent, GroupChatOrchestration, RoundRobinGroupChatManager
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from azure.identity import AzureCliCredential  # async credential
from typing import Annotated
import asyncio
import os
import time
import asyncio
import json
import re

from agents.tools import CosmosDBPlugin
from dotenv import load_dotenv

load_dotenv(override=True)  


class ClaimRequest(BaseModel):
    claimId: str
    policyNumber: str


app = FastAPI(title="Claim API", version="0.1.0")

async def get_specialized_agents() -> list[Agent]:
    ...
    async with DefaultAzureCredential() as creds:
        client = AzureAIAgent.create_client(credential=creds, endpoint=endpoint)
        print("✅ Connected to AI Foundry endpoint.")

        # Helper to safely ensure tools list
        async def ensure_tools(definition):
            if not getattr(definition, "tools", None):
                definition.tools = []
            return definition

        # Claim Reviewer Agent
        print("🔍 Getting Claim Reviewer Agent...")
        claim_reviewer_definition = await client.agents.get_agent(agent_id=os.environ.get("CLAIM_REV_AGENT_ID"))
        claim_reviewer_definition = await ensure_tools(claim_reviewer_definition)
        claim_reviewer_agent = AzureAIAgent(
            client=client,
            definition=claim_reviewer_definition,
            description="Agent that reviews insurance claims and retrieves claim details.",
            plugins=[cosmos_plugin_claims],
        )

        # Risk Analyzer Agent
        print("⚠️ Getting Risk Analyzer Agent...")
        risk_analyzer_definition = await client.agents.get_agent(agent_id=os.environ.get("RISK_ANALYZER_AGENT_ID"))
        risk_analyzer_definition = await ensure_tools(risk_analyzer_definition)
        risk_analyzer_agent = AzureAIAgent(
            client=client,
            definition=risk_analyzer_definition,
            description="Agent that analyzes the risk associated with the claim.",
            plugins=[cosmos_plugin_risk],
        )

        # Policy Checker Agent
        print("✅ Getting Policy Checker Agent...")
        policy_checker_definition = await client.agents.get_agent(agent_id=os.environ.get("POLICY_CHECKER_AGENT_ID"))
        policy_checker_definition = await ensure_tools(policy_checker_definition)
        policy_checker_agent = AzureAIAgent(
            client=client,
            definition=policy_checker_definition,
            description="Agent that checks if the policy covers the claim.",
        )
async def agent_response_callback(message: ChatMessageContent) -> None:
    print(f"# {message.name}\n{message.content}")
    
async def run_insurance_claim_orchestration(claim_id: str, policy_number: str):
    """Orchestrate multiple agents to process an insurance claim concurrently using only the claim ID."""

    print(f"🚀 Starting Concurrent Insurance Claim Processing Orchestration for claim ID: {claim_id} and policy number: {policy_number}")
    print(f"{'='*80}")
    
    # Create our specialized agents
    agents = await get_specialized_agents()
    
    group_chat_orchestration = GroupChatOrchestration(
        members=agents,
        manager=RoundRobinGroupChatManager(max_rounds=4), 
        agent_response_callback=agent_response_callback,
    )

    # Create and start runtime
    runtime = InProcessRuntime()
    runtime.start()
    
    try:        
        # Create task that instructs agents to retrieve claim details first
        task = f"""Analyze the insurance claim with ID: {claim_id} and  policy number {policy_number} and come back with a decision on whether to approve or deny the claim."""
        # Invoke concurrent orchestration
        orchestration_result = await group_chat_orchestration.invoke(
            task=task,
            runtime=runtime
        )
        
        # Get result
        result = await orchestration_result.get(timeout=300)  # 5 minute timeout

        print(f"\n✅ Insurance Claim Orchestration Complete!")
        # print result
        print(result)
        return result
        
    except Exception as e:
        print(f"❌ Error during orchestration: {str(e)}")
        raise
        
    finally:
        await runtime.stop_when_idle()
        print(f"\n🧹 Orchestration cleanup complete.")

def _normalize_orchestration_result(result: Any) -> Dict[str, Any]:
    """Normalize whatever the orchestration returns into a simple dict.

    The orchestration may return: a dict, a JSON string, a ChatMessageContent-like
    object with a .content attribute, or a list/tuple containing one of the above.
    Printing the object may show the JSON payload, but returning the raw object
    lets FastAPI serialize the object's full structure. This function extracts
    the JSON payload (with 'decision' and 'justification' where possible) or
    falls back to a {'response': str(result)} dict.
    """

    # If it's already a dict with desired keys, return it
    if isinstance(result, dict):
        if "decision" in result and "justification" in result:
            return result
        # try to find nested dict that has the keys
        for v in result.values():
            if isinstance(v, dict) and "decision" in v and "justification" in v:
                return v

    # If it's a list/tuple, try to extract from the first meaningful element
    if isinstance(result, (list, tuple)) and result:
        for item in result:
            normalized = _normalize_orchestration_result(item)
            if "decision" in normalized:
                return normalized

    # If it's a string, try to parse JSON from it
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            return _normalize_orchestration_result(parsed)
        except json.JSONDecodeError:
            # Attempt to extract a JSON object substring that contains 'decision'
            m = re.search(r"(\{[\s\S]*?\})", result)
            if m:
                try:
                    parsed = json.loads(m.group(1))
                    return _normalize_orchestration_result(parsed)
                except Exception:
                    pass
            return {"response": result}

    # If it has a 'content' attribute (e.g. ChatMessageContent-like), try that
    if hasattr(result, "content"):
        try:
            return _normalize_orchestration_result(result.content)
        except Exception:
            return {"response": str(result.content)}

    # If it has a 'message' attribute, inspect it
    if hasattr(result, "message"):
        try:
            return _normalize_orchestration_result(result.message)
        except Exception:
            return {"response": str(result.message)}

    # Fallback: return a simple response with a string representation
    return {"response": str(result)}


@app.post("/process-claim")
async def process_claim(req: ClaimRequest):
    """Process a claim request.

    Expects JSON body with 'claimId' and 'policyNumber' (both strings).
    Returns a JSON object with a single 'response' field (string).
    """

    # Run the orchestration to process the claim
    analysis_report = await run_insurance_claim_orchestration(req.claimId, req.policyNumber)

    print("Analysis Report:", analysis_report)

    # Normalize the orchestration output into a plain serializable dict
    normalized = _normalize_orchestration_result(analysis_report)

    print("Normalized response to return:", normalized)

    return normalized
