import asyncio as _asyncio

import time as _time
from observability.observability_wrapper import (
    trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
)
from config import settings as _obs_settings

import logging as _obs_startup_log
from contextlib import asynccontextmanager
from observability.instrumentation import initialize_tracer

_obs_startup_logger = _obs_startup_log.getLogger(__name__)

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': False,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': False
}

import logging
import json
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
import openai

from config import Config

# Constants from USER PROMPT TEMPLATE and RAG requirements
SYSTEM_PROMPT = (
    "You are a professional general domain knowledge assistant. Your task is to answer user questions by retrieving relevant information from the provided knowledge base documents using Azure AI Search. Follow these instructions:\n\n"
    "- Always base your answers strictly on the retrieved content from the knowledge base.\n\n"
    "- Clearly cite or reference the source document when possible.\n\n"
    "- If the answer cannot be found in the retrieved content, politely inform the user that no relevant information is available.\n\n"
    "- Maintain a formal, concise, and professional tone in all responses.\n\n"
    "- Do not speculate or provide information not present in the knowledge base.\n\n"
    "Output format: Provide a clear, direct answer. If applicable, include a brief reference to the source document (e.g., \"Source: Healthcare.pdf\").\n\n"
    "Fallback: \"If no relevant information is found, respond with: \\\"I'm sorry, I could not find relevant information in the available documents.\\\"\""
)
OUTPUT_FORMAT = (
    "- Direct, concise answer to the user's question\n"
    "- Reference to the source document when applicable (e.g., \"Source: Healthcare.pdf\")\n"
    "- Professional and formal language"
)
FALLBACK_RESPONSE = "I'm sorry, I could not find relevant information in the available documents."

VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(Path(__file__).parent / "validation_config.json")

SELECTED_DOCUMENT_TITLES = ["Healthcare.pdf"]

ENRICHED_FIELDS = ["entities", "keyphrases", "relationships"]

# Observability lifespan function
@asynccontextmanager
async def _obs_lifespan(application):
    """Initialise observability on startup, clean up on shutdown."""
    try:
        _obs_startup_logger.info('')
        _obs_startup_logger.info('========== Agent Configuration Summary ==========')
        _obs_startup_logger.info(f'Environment: {getattr(Config, "ENVIRONMENT", "N/A")}')
        _obs_startup_logger.info(f'Agent: {getattr(Config, "AGENT_NAME", "N/A")}')
        _obs_startup_logger.info(f'Project: {getattr(Config, "PROJECT_NAME", "N/A")}')
        _obs_startup_logger.info(f'LLM Provider: {getattr(Config, "MODEL_PROVIDER", "N/A")}')
        _obs_startup_logger.info(f'LLM Model: {getattr(Config, "LLM_MODEL", "N/A")}')
        _cs_endpoint = getattr(Config, 'AZURE_CONTENT_SAFETY_ENDPOINT', None)
        _cs_key = getattr(Config, 'AZURE_CONTENT_SAFETY_KEY', None)
        if _cs_endpoint and _cs_key:
            _obs_startup_logger.info('Content Safety: Enabled (Azure Content Safety)')
            _obs_startup_logger.info(f'Content Safety Endpoint: {_cs_endpoint}')
        else:
            _obs_startup_logger.info('Content Safety: Not Configured')
        _obs_startup_logger.info('Observability Database: Azure SQL')
        _obs_startup_logger.info(f'Database Server: {getattr(Config, "OBS_AZURE_SQL_SERVER", "N/A")}')
        _obs_startup_logger.info(f'Database Name: {getattr(Config, "OBS_AZURE_SQL_DATABASE", "N/A")}')
        _obs_startup_logger.info('===============================================')
        _obs_startup_logger.info('')
    except Exception as _e:
        _obs_startup_logger.warning('Config summary failed: %s', _e)

    _obs_startup_logger.info('')
    _obs_startup_logger.info('========== Content Safety & Guardrails ==========')
    if GUARDRAILS_CONFIG.get('content_safety_enabled'):
        _obs_startup_logger.info('Content Safety: Enabled')
        _obs_startup_logger.info(f'  - Severity Threshold: {GUARDRAILS_CONFIG.get("content_safety_severity_threshold", "N/A")}')
        _obs_startup_logger.info(f'  - Check Toxicity: {GUARDRAILS_CONFIG.get("check_toxicity", False)}')
        _obs_startup_logger.info(f'  - Check Jailbreak: {GUARDRAILS_CONFIG.get("check_jailbreak", False)}')
        _obs_startup_logger.info(f'  - Check PII Input: {GUARDRAILS_CONFIG.get("check_pii_input", False)}')
        _obs_startup_logger.info(f'  - Check Credentials Output: {GUARDRAILS_CONFIG.get("check_credentials_output", False)}')
    else:
        _obs_startup_logger.info('Content Safety: Disabled')
    _obs_startup_logger.info('===============================================')
    _obs_startup_logger.info('')

    _obs_startup_logger.info('========== Initializing Agent Services ==========')
    # 1. Observability DB schema (imports are inside function — only needed at startup)
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
        _obs_startup_logger.info('✓ Observability database connected')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Observability database connection failed (metrics will not be saved)')
    # 2. OpenTelemetry tracer (initialize_tracer is pre-injected at top level)
    try:
        _t = initialize_tracer()
        if _t is not None:
            _obs_startup_logger.info('✓ Telemetry monitoring enabled')
        else:
            _obs_startup_logger.warning('✗ Telemetry monitoring disabled')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Telemetry monitoring failed to initialize')
    _obs_startup_logger.info('=================================================')
    _obs_startup_logger.info('')
    yield

app = FastAPI(lifespan=_obs_lifespan,

    title="General Domain Knowledge Answering Agent",
    description="Answers user questions from knowledge base documents using Azure AI Search and GPT-4.1. Strict source grounding, document filtering, and professional tone.",
    version=Config.SERVICE_VERSION if hasattr(Config, "SERVICE_VERSION") else "1.0.0",
    # SYNTAX-FIX: lifespan=_obs_lifespan
)

_logger = logging.getLogger("agent")
_enriched_available = None  # None = not yet checked, True/False after first search

class QueryRequest(BaseModel):
    query: str = Field(..., description="User question (max 50,000 chars)")

    @model_validator(mode="after")
    def validate_content(self):
        if not self.query or not self.query.strip():
            raise ValueError("Query must be non-empty.")
        if len(self.query.strip()) > 50000:
            raise ValueError("Query exceeds maximum length.")
        self.query = self.query.strip()
        return self

class QueryResponse(BaseModel):
    success: bool = Field(..., description="Whether the query was processed successfully")
    answer: str = Field(..., description="Agent's answer")
    sources: Optional[List[str]] = Field(None, description="List of source document titles referenced")
    tool_calls_made: Optional[List[str]] = Field(None, description="List of tool calls made (empty for this agent)")
    error: Optional[str] = Field(None, description="Error message if any")

# LLM Output Sanitizer
import re as _re

_FENCE_RE = _re.compile(r"```(?:\w+)?\s*\n(.*?)```", _re.DOTALL)
_LONE_FENCE_START_RE = _re.compile(r"^```\w*$")
_WRAPPER_RE = _re.compile(
    r"^(?:"
    r"Here(?:'s| is)(?: the)? (?:the |your |a )?(?:code|solution|implementation|result|explanation|answer)[^:]*:\s*"
    r"|Sure[!,.]?\s*"
    r"|Certainly[!,.]?\s*"
    r"|Below is [^:]*:\s*"
    r")",
    _re.IGNORECASE,
)
_SIGNOFF_RE = _re.compile(
    r"^(?:Let me know|Feel free|Hope this|This code|Note:|Happy coding|If you)",
    _re.IGNORECASE,
)
_BLANK_COLLAPSE_RE = _re.compile(r"\n{3,}")

def _strip_fences(text: str, content_type: str) -> str:
    """Extract content from Markdown code fences."""
    fence_matches = _FENCE_RE.findall(text)
    if fence_matches:
        if content_type == "code":
            return "\n\n".join(block.strip() for block in fence_matches)
        for match in fence_matches:
            fenced_block = _FENCE_RE.search(text)
            if fenced_block:
                text = text[:fenced_block.start()] + match.strip() + text[fenced_block.end():]
        return text
    lines = text.splitlines()
    if lines and _LONE_FENCE_START_RE.match(lines[0].strip()):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def _strip_trailing_signoffs(text: str) -> str:
    """Remove conversational sign-off lines from the end of code output."""
    lines = text.splitlines()
    while lines and _SIGNOFF_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).rstrip()

@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_llm_output(raw: str, content_type: str = "code") -> str:
    """
    Generic post-processor that cleans common LLM output artefacts.
    Args:
        raw: Raw text returned by the LLM.
        content_type: 'code' | 'text' | 'markdown'.
    Returns:
        Cleaned string ready for validation, formatting, or direct return.
    """
    if not raw:
        return ""
    text = _strip_fences(raw.strip(), content_type)
    text = _WRAPPER_RE.sub("", text, count=1).strip()
    if content_type == "code":
        text = _strip_trailing_signoffs(text)
    return _BLANK_COLLAPSE_RE.sub("\n\n", text).strip()

# Embedding Service
class EmbeddingService:
    """Generates embeddings for user queries using Azure OpenAI embedding model."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            api_key = Config.AZURE_OPENAI_API_KEY
            if not api_key:
                raise ValueError("AZURE_OPENAI_API_KEY not configured")
            self._client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            )
        return self._client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for input text."""
        _t0 = _time.time()
        client = self._get_client()
        try:
            resp = await client.embeddings.create(
                input=text,
                model=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "text-embedding-ada-002"
            )
            embedding = resp.data[0].embedding
            try:
                trace_tool_call(
                    tool_name="openai_client.embeddings.create",
                    latency_ms=int((_time.time() - _t0) * 1000),
                    output=str(embedding)[:200],
                    status="success",
                )
            except Exception:
                pass
            return embedding
        except Exception as e:
            _logger.error("Embedding generation failed: %s", e)
            try:
                trace_tool_call(
                    tool_name="openai_client.embeddings.create",
                    latency_ms=int((_time.time() - _t0) * 1000),
                    output=None,
                    status="error",
                    error=e,
                )
            except Exception:
                pass
            raise

# Chunk Retriever
class ChunkRetriever:
    """Retrieves relevant chunks from Azure AI Search using vector + keyword search."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            endpoint = Config.AZURE_SEARCH_ENDPOINT
            api_key = Config.AZURE_SEARCH_API_KEY
            index_name = Config.AZURE_SEARCH_INDEX_NAME
            if not endpoint or not api_key or not index_name:
                raise ValueError("Azure Search endpoint, API key, or index name not configured")
            self._client = SearchClient(
                endpoint=endpoint,
                index_name=index_name,
                credential=AzureKeyCredential(api_key),
            )
        return self._client

    async def _search_with_fallback(self, query: str, embedding: List[float], selected_titles: List[str], top_k: int) -> List[Dict[str, Any]]:
        """Try search with enriched fields; fall back to base fields if index lacks them."""
        global _enriched_available
        from azure.core.exceptions import HttpResponseError

        vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=top_k, fields="vector")
        base_fields = ["chunk", "title"]

        # If we already know enriched fields are not available, skip them
        if _enriched_available is False:
            select_fields = base_fields
        else:
            select_fields = base_fields + ENRICHED_FIELDS

        search_kwargs = {
            "search_text": query,
            "vector_queries": [vector_query],
            "top": top_k,
            "select": select_fields,
        }
        if selected_titles:
            odata_parts = [f"title eq '{t}'" for t in selected_titles]
            search_kwargs["filter"] = " or ".join(odata_parts)

        client = self._get_client()
        _t0 = _time.time()
        try:
            results = list(client.search(**search_kwargs))
            if _enriched_available is None:
                _enriched_available = True
                _logger.info("Enriched index fields are AVAILABLE — using: %s", ENRICHED_FIELDS)
            try:
                trace_tool_call(
                    tool_name="search_client.search",
                    latency_ms=int((_time.time() - _t0) * 1000),
                    output=str(results)[:200],
                    status="success",
                )
            except Exception:
                pass
            return results
        except HttpResponseError as e:
            if "Could not find a property named" in str(e) and _enriched_available is not False:
                _enriched_available = False
                _logger.warning("Enriched index fields NOT available in this index — falling back to base fields: %s", base_fields)
                search_kwargs["select"] = base_fields
                results = list(client.search(**search_kwargs))
                try:
                    trace_tool_call(
                        tool_name="search_client.search",
                        latency_ms=int((_time.time() - _t0) * 1000),
                        output=str(results)[:200],
                        status="success",
                    )
                except Exception:
                    pass
                return results
            try:
                trace_tool_call(
                    tool_name="search_client.search",
                    latency_ms=int((_time.time() - _t0) * 1000),
                    output=None,
                    status="error",
                    error=e,
                )
            except Exception:
                pass
            raise

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def retrieve_chunks(self, query: str, document_titles: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks from Azure AI Search."""
        embedding_service = EmbeddingService()
        embedding = await embedding_service.embed_text(query)
        results = await self._search_with_fallback(query, embedding, document_titles, top_k)
        return results

# LLM Service
class LLMService:
    """Calls Azure OpenAI GPT-4.1 with system prompt, user query, retrieved chunks."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            api_key = Config.AZURE_OPENAI_API_KEY
            if not api_key:
                raise ValueError("AZURE_OPENAI_API_KEY not configured")
            self._client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            )
        return self._client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def generate_answer(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Generate answer using LLM with system prompt, user query, and retrieved chunks."""
        # Build context for LLM
        context_parts = []
        sources = set()
        for r in chunks:
            part = r.get("chunk", "")
            title = r.get("title", "")
            if title:
                sources.add(title)
            if _enriched_available:
                for field in ENRICHED_FIELDS:
                    value = r.get(field)
                    if value:
                        part += f"\n{field}: {json.dumps(value) if isinstance(value, (list, dict)) else value}"
            context_parts.append(part)
        context = "\n\n".join(context_parts)
        # Compose system message
        system_message = SYSTEM_PROMPT + "\n\nOutput Format: " + OUTPUT_FORMAT
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
            {"role": "assistant", "content": context}
        ]
        client = self._get_client()
        _llm_kwargs = Config.get_llm_kwargs()
        _t0 = _time.time()
        try:
            response = await client.chat.completions.create(
                model=Config.LLM_MODEL or "gpt-4.1",
                messages=messages,
                **_llm_kwargs
            )
            content = response.choices[0].message.content
            try:
                trace_model_call(
                    provider="azure",
                    model_name=Config.LLM_MODEL or "gpt-4.1",
                    prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                    latency_ms=int((_time.time() - _t0) * 1000),
                    response_summary=content[:200] if content else "",
                )
            except Exception:
                pass
            return content
        except Exception as e:
            _logger.error("LLM call failed: %s", e)
            try:
                trace_model_call(
                    provider="azure",
                    model_name=Config.LLM_MODEL or "gpt-4.1",
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=int((_time.time() - _t0) * 1000),
                    response_summary=str(e),
                    status="error",
                    error=e,
                )
            except Exception:
                pass
            return FALLBACK_RESPONSE

# Response Formatter
class ResponseFormatter:
    """Formats LLM output according to output instructions; adds source attribution."""

    def format_response(self, llm_output: str, sources: List[str]) -> str:
        """Format LLM output, add source attribution, apply fallback if needed."""
        answer = sanitize_llm_output(llm_output, content_type="text")
        if not answer or answer.strip() == "":
            return FALLBACK_RESPONSE
        # Check for fallback phrase
        if FALLBACK_RESPONSE.lower() in answer.lower():
            return FALLBACK_RESPONSE
        # Add source attribution if possible
        if sources:
            # If not already referenced, append source
            if not any(s in answer for s in sources):
                answer = f"{answer}\nSource: {', '.join(sources)}"
        return answer

# Audit Logger
class AuditLogger:
    """Logs all user queries, retrievals, LLM calls, responses, and errors."""

    def log_interaction(self, interaction: Dict[str, Any]) -> None:
        try:
            _logger.info("Audit log: %s", json.dumps(interaction, default=str))
        except Exception as e:
            _logger.error("Audit logging failed: %s", e)

# Main Agent Class
class GeneralDomainKnowledgeAgent:
    """Orchestrates input processing, retrieval, LLM calls, response formatting, and audit logging."""

    def __init__(self):
        self.chunk_retriever = ChunkRetriever()
        self.llm_service = LLMService()
        self.response_formatter = ResponseFormatter()
        self.audit_logger = AuditLogger()

    @trace_agent(agent_name=_obs_settings.AGENT_NAME, project_name=_obs_settings.PROJECT_NAME)
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process(self, query: str) -> Dict[str, Any]:
        """Main agent entry point: receives query, retrieves chunks, generates answer, formats response, logs interaction."""
        async with trace_step(
            "parse_input", step_type="parse",
            decision_summary="Validate and parse user query",
            output_fn=lambda r: f"query={r[:100]}" if isinstance(r, str) else str(r)
        ) as step:
            parsed_query = query.strip()
            step.capture(parsed_query)

        async with trace_step(
            "retrieve_chunks", step_type="process",
            decision_summary="Retrieve relevant chunks from Azure AI Search",
            output_fn=lambda r: f"chunks={len(r)}"
        ) as step:
            try:
                chunks = await self.chunk_retriever.retrieve_chunks(parsed_query, SELECTED_DOCUMENT_TITLES, top_k=5)
                step.capture(chunks)
            except Exception as e:
                self.audit_logger.log_interaction({
                    "query": parsed_query,
                    "error": str(e),
                    "stage": "retrieval",
                    "success": False
                })
                return {
                    "success": False,
                    "answer": FALLBACK_RESPONSE,
                    "sources": [],
                    "tool_calls_made": [],
                    "error": f"Retrieval failed: {e}"
                }

        async with trace_step(
            "llm_call", step_type="llm_call",
            decision_summary="Generate answer using LLM",
            output_fn=lambda r: f"answer={r[:100]}" if isinstance(r, str) else str(r)
        ) as step:
            try:
                llm_output = await self.llm_service.generate_answer(parsed_query, chunks)
                step.capture(llm_output)
            except Exception as e:
                self.audit_logger.log_interaction({
                    "query": parsed_query,
                    "error": str(e),
                    "stage": "llm_call",
                    "success": False
                })
                return {
                    "success": False,
                    "answer": FALLBACK_RESPONSE,
                    "sources": [],
                    "tool_calls_made": [],
                    "error": f"LLM call failed: {e}"
                }

        async with trace_step(
            "format_response", step_type="format",
            decision_summary="Format LLM output and add source attribution",
            output_fn=lambda r: f"formatted={r[:100]}" if isinstance(r, str) else str(r)
        ) as step:
            sources = [r.get("title", "") for r in chunks if r.get("title")]
            answer = self.response_formatter.format_response(llm_output, sources)
            step.capture(answer)

        self.audit_logger.log_interaction({
            "query": parsed_query,
            "answer": answer,
            "sources": sources,
            "success": True
        })

        return {
            "success": True,
            "answer": answer,
            "sources": sources,
            "tool_calls_made": [],
            "error": None
        }

# FastAPI Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle generic errors and malformed JSON."""
    _logger.error("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "answer": FALLBACK_RESPONSE,
            "sources": [],
            "tool_calls_made": [],
            "error": f"Internal error: {exc}. Please check your input and try again."
        }
    )

@app.exception_handler(ValueError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle input validation errors."""
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "answer": FALLBACK_RESPONSE,
            "sources": [],
            "tool_calls_made": [],
            "error": f"Input error: {exc}. Please check your input and try again."
        }
    )

@app.post("/query", response_model=QueryResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def query_endpoint(req: QueryRequest):
    """Main query endpoint."""
    agent = GeneralDomainKnowledgeAgent()
    try:
        result = await agent.process(req.query)
        return QueryResponse(
            success=result.get("success", False),
            answer=result.get("answer", FALLBACK_RESPONSE),
            sources=result.get("sources", []),
            tool_calls_made=result.get("tool_calls_made", []),
            error=result.get("error")
        )
    except Exception as e:
        _logger.error("Agent processing failed: %s", e)
        return QueryResponse(
            success=False,
            answer=FALLBACK_RESPONSE,
            sources=[],
            tool_calls_made=[],
            error=f"Agent error: {e}"
        )

async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""
    import uvicorn

    # Unified logging config — routes uvicorn, agent, and observability through
    # the same handler so all telemetry appears in a single consistent stream.
    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    _asyncio.run(_run_agent())