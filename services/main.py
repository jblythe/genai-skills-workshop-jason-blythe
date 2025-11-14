"""FastAPI service for the Alaska Department of Snow virtual assistant.

The service exposes a `/chat` endpoint that orchestrates retrieval-augmented generation
(RAG) using Vertex AI Gemini with safety gates provided by custom heuristics and
Google Model Armor.
- Retrieval is backed by Vertex AI Search (Dialogflow Data Store).
- Prompt/response sanitisation leverages Model Armor templates when configured.
- Every major step is logged to Cloud Logging for observability and auditability.
"""

from dataclasses import dataclass
from datetime import datetime
import logging
import os
import re
from typing import Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import google.auth
import requests
from google.auth.transport.requests import AuthorizedSession, Request

from google.cloud import discoveryengine_v1
from google.cloud import logging_v2
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("VERTEXAI_LOCATION", "us-central1")
SEARCH_SERVING_CONFIG = os.environ.get("VERTEX_SEARCH_SERVING_CONFIG")
MODEL_NAME = os.environ.get("GENAI_MODEL", "gemini-2.5-flash-lite")
MODEL_ARMOR_PROMPT_TEMPLATE = os.environ.get("MODEL_ARMOR_PROMPT_TEMPLATE")
MODEL_ARMOR_RESPONSE_TEMPLATE = os.environ.get(
    "MODEL_ARMOR_RESPONSE_TEMPLATE", MODEL_ARMOR_PROMPT_TEMPLATE
)

MODEL_ARMOR_API_BASE = "https://modelarmor.us-central1.rep.googleapis.com/v1"
_MODEL_ARMOR_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)

SENSITIVE_PATTERNS = {
    "pii_email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", re.IGNORECASE),
    "pii_phone": re.compile(r"(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"),
}

DEFAULT_CONTEXT = (
    "You are the official Alaska Department of Snow (ADS) virtual assistant. "
    "Provide concise, policy-compliant answers to residents about snow removal operations, "
    "parking rules during snow emergencies, street maintenance status, and ADS contact channels. "
    "Only answer when confident. If the question cannot be answered with ADS guidance, admit the gap "
    "and recommend contacting ADS directly."
)

ADS_KEYWORDS = {
    "snow",
    "plow",
    "plowing",
    "parking",
    "permit",
    "storm",
    "winter",
    "ice",
    "sidewalk",
    "driveway",
    "ads",
    "alaska",
    "department of snow",
    "road",
    "closure",
    "schedule",
    "emergency",
}


def _resolve_serving_config() -> str:
    """Return the fully-qualified Vertex Search serving config path."""
    if not SEARCH_SERVING_CONFIG:
        raise RuntimeError(
            "Vertex Search serving config not configured. Set VERTEX_SEARCH_SERVING_CONFIG to the path "
            "returned by initialize_dialogflow_datastore()."
        )
    if "/" in SEARCH_SERVING_CONFIG:
        return SEARCH_SERVING_CONFIG
    if not PROJECT_ID or not LOCATION:
        raise RuntimeError(
            "Cannot resolve Vertex Search serving config without GOOGLE_CLOUD_PROJECT and VERTEXAI_LOCATION."
        )
    return (
        f"projects/{PROJECT_ID}/locations/global/collections/default_collection/"
        f"dataStores/{SEARCH_SERVING_CONFIG}/servingConfigs/default_serving_config"
    )


SERVING_CONFIG_NAME = _resolve_serving_config()

# ---------------------------------------------------------------------------
# FastAPI setup & logging
# ---------------------------------------------------------------------------
app = FastAPI(title="ADS Snow Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging_client = logging_v2.Client(project=PROJECT_ID)
logger = logging_client.logger("ads_snow_agent")


def log_event(event: str, session_id: str, severity: str = "INFO", **fields) -> None:
    """Emit a structured log entry for the supplied event.

    Parameters
    ----------
    event
        A short event identifier (e.g. "retrieval_completed").
    session_id
        Unique identifier for the active chat session.
    severity
        Cloud Logging severity level (default "INFO").
    **fields
        Extra key/value pairs captured in the log payload.
    """

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event": event,
        "session_id": session_id,
        **fields,
    }
    logger.log_struct(payload, severity=severity)


# ---------------------------------------------------------------------------
# Agent configuration helpers (adapted from lab_1_baking_agent workflow)
# ---------------------------------------------------------------------------


@dataclass
class ModelArmorTemplateConfig:
    """Model Armor template identifiers used for sanitisation."""

    prompt_template: str
    response_template: Optional[str] = None


@dataclass
class AgentSettings:
    """Settings required to initialise the ADS virtual assistant."""

    vertex_project_id: str
    vertex_location: str
    model_name: str
    base_context: str = DEFAULT_CONTEXT
    model_armor_prompt_template: Optional[str] = None
    model_armor_response_template: Optional[str] = None


def validate_user_question(question: str) -> Tuple[bool, Optional[str]]:
    """Return whether the user question is in-scope for the ADS assistant."""

    normalised = question.lower()
    if any(keyword in normalised for keyword in ADS_KEYWORDS):
        return True, None
    return (
        False,
        "I can help with Alaska Department of Snow policies, operations, and resident guidance.",
    )


def build_context(additional_context: str = "") -> str:
    """Combine the base ADS context with optional additional instructions."""

    segments = [DEFAULT_CONTEXT]
    if additional_context.strip():
        segments.append(additional_context.strip())
    return "\n\n".join(segments)


def augment_user_query(question: str, context: str) -> str:
    """Construct the prompt sent to the model with baked-in instructions."""

    return f"{context}\n\nResident question: {question.strip()}"


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------
search_client = discoveryengine_v1.SearchServiceClient()


def retrieve_context(query: str, session_id: str) -> Tuple[list[str], list[str]]:
    """Query Vertex AI Search for supporting context."""

    search_request = discoveryengine_v1.SearchRequest(
        serving_config=SERVING_CONFIG_NAME,
        query=query,
        page_size=6,
        query_expansion_spec=discoveryengine_v1.SearchRequest.QueryExpansionSpec(
            condition=discoveryengine_v1.SearchRequest.QueryExpansionSpec.Condition.DISABLED
        ),
    )
    search_results = search_client.search(request=search_request)

    context_blocks: list[str] = []
    sources: list[str] = []
    for result in search_results:
        document = result.document
        text = (document.content or "").strip()
        if not text and document.struct_data:
            text = "\n".join(f"{key}: {value}" for key, value in document.struct_data.items())
        if not text and document.derived_struct_data:
            text = "\n".join(
                f"{key}: {value}" for key, value in document.derived_struct_data.items()
            )
        source = document.content_uri or document.name
        if text:
            context_blocks.append(f"Source: {source}\n{text}")
        sources.append(source)

    log_event("retrieval_completed", session_id, neighbors=len(context_blocks), sources=sources)
    return context_blocks, sources


# ---------------------------------------------------------------------------
# Vertex chat client (adapted from lab_1_baking_agent)
# ---------------------------------------------------------------------------


class VertexChatClient:
    """Wrapper around Vertex AI GenerativeModel."""

    def __init__(self, project: str, location: str, model_name: str):
        try:
            vertexai.init(project=project, location=location)
        except Exception as exc:  # pragma: no cover - env specific
            raise RuntimeError("Vertex AI initialization failed") from exc
        self.model = GenerativeModel(model_name)
        self.generation_config = GenerationConfig(temperature=0.2, max_output_tokens=512)

    def generate(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt, generation_config=self.generation_config)
        except Exception as exc:  # pragma: no cover - upstream SDK errors vary
            raise RuntimeError("Vertex chat generation failed") from exc

        if not response or not getattr(response, "text", None):
            raise RuntimeError("Vertex chat returned an empty response")

        return response.text.strip()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class PromptSafetyError(Exception):
    """Raised when heuristic safety rules reject user prompts or model output."""


class SafetyViolationError(Exception):
    """Raised when Model Armor blocks content."""

class SafetyCheckError(Exception):
    """Raised when Model Armor cannot perform the requested check."""


# ---------------------------------------------------------------------------
# Model Armor integration (aligned with lab_1_baking_agent)
# ---------------------------------------------------------------------------
_AUTHORIZED_SESSION: Optional[AuthorizedSession] = None


def get_authorized_session() -> AuthorizedSession:
    """Return a cached AuthorizedSession for Model Armor API calls."""

    global _AUTHORIZED_SESSION
    if _AUTHORIZED_SESSION is None:
        credentials, _ = google.auth.default(scopes=_MODEL_ARMOR_SCOPES)
        credentials.refresh(Request())
        _AUTHORIZED_SESSION = AuthorizedSession(credentials)
    return _AUTHORIZED_SESSION


def _is_blocked(model_armor_result: dict) -> bool:
    """Inspect Model Armor response payload and determine if it blocked content.

    Parameters
    ----------
    model_armor_result
        Raw JSON payload returned from a Model Armor sanitisation request.
    """

    verdict = (
        model_armor_result.get("verdict")
        or model_armor_result.get("decision")
        or model_armor_result.get("overallVerdict")
        or model_armor_result.get("outcome")
    )
    if isinstance(verdict, str) and verdict.upper() in {"BLOCK", "REJECT", "DENY"}:
        return True

    findings = (
        model_armor_result.get("findings")
        or model_armor_result.get("issues")
        or model_armor_result.get("scans")
        or []
    )
    if isinstance(findings, list):
        for finding in findings:
            if isinstance(finding, dict):
                severity = finding.get("severity") or finding.get("level")
                action = finding.get("action") or finding.get("recommendation")
                if (severity and str(severity).upper() in {"HIGH", "BLOCK", "CRITICAL"}) or (
                    action and str(action).upper() in {"BLOCK", "REDACT", "QUARANTINE"}
                ):
                    return True
    return False


class ModelArmorClient:
    """Wrapper around Model Armor REST endpoints using template resources."""

    def __init__(self, config: ModelArmorTemplateConfig):
        self.config = config
        self.session = get_authorized_session()

    def sanitize_prompt(self, prompt: str, session_id: Optional[str] = None) -> str:
        endpoint = f"{MODEL_ARMOR_API_BASE}/{self.config.prompt_template}:sanitizeUserPrompt"
        payload = {"userPromptData": {"text": prompt}}
        try:
            response = self.session.post(endpoint, json=payload, timeout=20)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise SafetyCheckError("Model Armor prompt sanitization failed") from exc

        result = response.json()
        if _is_blocked(result):
            raise SafetyViolationError("Model Armor blocked the prompt")

        sanitized = result.get("sanitizationResult", {}).get("sanitizedText") or prompt
        if session_id:
            log_event(
                "prompt_sanitized",
                session_id,
                sanitized=bool(sanitized and sanitized != prompt),
            )
        return sanitized

    def sanitize_response(self, response_text: str, session_id: Optional[str] = None) -> str:
        endpoint = f"{MODEL_ARMOR_API_BASE}/{self.config.response_template}:sanitizeModelResponse"
        payload = {"modelResponseData": {"text": response_text}}
        try:
            response = self.session.post(endpoint, json=payload, timeout=20)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise SafetyCheckError("Model Armor response sanitization failed") from exc

        result = response.json()
        if _is_blocked(result):
            raise SafetyViolationError("Model Armor blocked the response")

        sanitized = result.get("sanitizationResult", {}).get("sanitizedText") or response_text
        if session_id:
            log_event(
                "response_sanitized",
                session_id,
                sanitized=bool(sanitized and sanitized != response_text),
            )
        return sanitized


armor_client: Optional[ModelArmorClient] = None
if MODEL_ARMOR_PROMPT_TEMPLATE:
    try:
        armor_config = ModelArmorTemplateConfig(
            prompt_template=MODEL_ARMOR_PROMPT_TEMPLATE,
            response_template=MODEL_ARMOR_RESPONSE_TEMPLATE,
        )
        armor_client = ModelArmorClient(config=armor_config)
    except Exception as exc:  # pragma: no cover - allow service to run without armor
        logging.getLogger("ads_snow_agent").warning("Model Armor init failed: %s", exc)
        armor_client = None


agent_settings = AgentSettings(
    vertex_project_id=PROJECT_ID,
    vertex_location=LOCATION,
    model_name=MODEL_NAME,
    base_context=DEFAULT_CONTEXT,
    model_armor_prompt_template=MODEL_ARMOR_PROMPT_TEMPLATE,
    model_armor_response_template=MODEL_ARMOR_RESPONSE_TEMPLATE,
)

chat_client = VertexChatClient(
    project=agent_settings.vertex_project_id,
    location=agent_settings.vertex_location,
    model_name=agent_settings.model_name,
)


# ---------------------------------------------------------------------------
# Heuristic validation helpers
# ---------------------------------------------------------------------------
def validate_response(answer: str) -> None:
    """Perform post-generation checks to ensure the answer is policy-compliant.

    Parameters
    ----------
    answer
        Model-generated text ready to return to the user.
    """

    for label, pattern in SENSITIVE_PATTERNS.items():
        if pattern.search(answer):
            raise PromptSafetyError(f"ADS safety policy violation detected: {label}")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """Payload submitted by the frontend chat widget.

    Attributes
    ----------
    session_id
        Unique identifier for the user conversation.
    message
        Latest user message to forward to the assistant.
    """

    session_id: str
    message: str


class ChatResponse(BaseModel):
    """Response payload returned to the frontend chat widget.

    Attributes
    ----------
    answer
        Assistant response text.
    sources
        List of source identifiers referenced in the answer.
    """

    answer: str
    sources: list[str]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health_check():
    """Liveness probe endpoint."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat_request(request: ChatRequest):
    """Handle an interactive chat turn from the frontend.

    Parameters
    ----------
    request
        Pydantic model containing the user session identifier and message.
    """

    session_id = request.session_id
    user_message = request.message
    log_event("chat_received", session_id, prompt=user_message)

    try:
        is_valid, violation_message = validate_user_question(user_message)
        if not is_valid:
            raise PromptSafetyError(violation_message or "Out-of-domain request")
        log_event("prompt_validated", session_id)

        base_context = build_context(agent_settings.base_context)
        augmented_prompt = augment_user_query(user_message, base_context)

        context_blocks, sources = retrieve_context(user_message, session_id)
        if context_blocks:
            augmented_prompt = (
                f"{augmented_prompt}\n\nADS reference material:\n" + "\n\n".join(context_blocks)
            )

        safe_prompt = augmented_prompt
        if armor_client:
            safe_prompt = armor_client.sanitize_prompt(augmented_prompt, session_id=session_id)

        answer = chat_client.generate(safe_prompt)
        log_event("generation_completed", session_id)

        if armor_client:
            answer = armor_client.sanitize_response(answer, session_id=session_id)

        validate_response(answer)
        log_event("safety_checks_passed", session_id, sources=sources)

        log_event("chat_completed", session_id, answer=answer, sources=sources)
        return ChatResponse(answer=answer, sources=sources)

    except SafetyViolationError as unsafe:
        log_event("model_armor_block", session_id, severity="WARNING", detail=str(unsafe))
        raise HTTPException(status_code=422, detail="Prompt blocked by safety system") from unsafe
    except SafetyCheckError as armor_error:
        log_event("model_armor_error", session_id, severity="ERROR", detail=str(armor_error))
        raise HTTPException(status_code=500, detail="Safety system error") from armor_error
    except PromptSafetyError as unsafe:
        log_event("prompt_rejected", session_id, severity="WARNING", detail=str(unsafe))
        raise HTTPException(status_code=422, detail=str(unsafe)) from unsafe
    except Exception as exc:  # pragma: no cover - defensive, ensures logging for unexpected errors
        log_event("chat_error", session_id, severity="ERROR", detail=str(exc))
        raise HTTPException(status_code=500, detail="Agent service error") from exc
