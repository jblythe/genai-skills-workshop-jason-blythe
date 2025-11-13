"""FastAPI service for the Alaska Department of Snow virtual assistant.

The service exposes a `/chat` endpoint that orchestrates retrieval-augmented generation
(RAG) using Vertex AI Gemini with safety gates provided by custom heuristics and
Google Model Armor.
- Retrieval is backed by a Matching Engine index (Vertex AI Vector Search).
- Prompt/response sanitisation leverages Model Armor templates when configured.
- Every major step is logged to Cloud Logging for observability and auditability.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import os
import re
from datetime import datetime
from typing import Optional

import google.auth
import requests
from google.auth.transport.requests import AuthorizedSession, Request

from google.cloud import logging_v2
from vertexai.generative_models import GenerativeModel, GenerationConfig
from google.cloud import aiplatform

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("VERTEXAI_LOCATION", "us-central1")
INDEX_ENDPOINT_ID = os.environ.get("VERTEX_MATCHING_ENGINE_ENDPOINT_ID")
INDEX_ENDPOINT_NAME_ENV = os.environ.get("VERTEX_MATCHING_ENGINE_ENDPOINT")
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
ALLOWED_TOPICS = ["snow", "plow", "parking", "permit"]


def _resolve_matching_engine_endpoint() -> str:
    """Return the fully-qualified Matching Engine endpoint path.

    Prefers the bare endpoint ID (`VERTEX_MATCHING_ENGINE_ENDPOINT_ID`) when set,
    falling back to a legacy fully-qualified value (`VERTEX_MATCHING_ENGINE_ENDPOINT`).
    """

    candidate = INDEX_ENDPOINT_ID or INDEX_ENDPOINT_NAME_ENV
    if not candidate:
        raise RuntimeError(
            "Matching Engine endpoint not configured. Set VERTEX_MATCHING_ENGINE_ENDPOINT_ID"
            " or VERTEX_MATCHING_ENGINE_ENDPOINT."
        )
    if "/" in candidate:
        return candidate
    if not PROJECT_ID or not LOCATION:
        raise RuntimeError(
            "Cannot resolve Matching Engine endpoint ID without GOOGLE_CLOUD_PROJECT and VERTEXAI_LOCATION."
        )
    return f"projects/{PROJECT_ID}/locations/{LOCATION}/indexEndpoints/{candidate}"


MATCHING_ENGINE_ENDPOINT_NAME = _resolve_matching_engine_endpoint()

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
# External clients (Vertex AI + Matching Engine)
# ---------------------------------------------------------------------------
aiplatform.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel(MODEL_NAME)
endpoint = aiplatform.MatchingEngineIndexEndpoint(MATCHING_ENGINE_ENDPOINT_NAME)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class PromptSafetyError(Exception):
    """Raised when heuristic safety rules reject user prompts or model output."""


class ModelArmorError(Exception):
    """Raised when Model Armor encounters an operational issue."""


class SafetyViolationError(Exception):
    """Raised when Model Armor blocks content."""


# ---------------------------------------------------------------------------
# Model Armor integration
# ---------------------------------------------------------------------------
def _create_authorized_session() -> AuthorizedSession:
    """Return an authorised session used for Model Armor REST calls."""
    credentials, _ = google.auth.default(scopes=_MODEL_ARMOR_SCOPES)
    credentials.refresh(Request())
    return AuthorizedSession(credentials)


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
    """Convenience wrapper around Model Armor prompt/response sanitisation APIs."""

    def __init__(self, prompt_template: str, response_template: Optional[str] = None):
        """Initialise the client.

        Parameters
        ----------
        prompt_template
            Fully-qualified Model Armor template resource used to sanitise prompts.
        response_template
            Optional template resource for sanitising model responses. Defaults to
            `prompt_template` when omitted.
        """

        if not prompt_template:
            raise ValueError("Model Armor prompt template is required")
        self.prompt_template = prompt_template
        self.response_template = response_template or prompt_template
        self.session = _create_authorized_session()

    def sanitize_prompt(self, prompt: str, session_id: str) -> str:
        """Sanitise the user prompt using Model Armor templates.

        Parameters
        ----------
        prompt
            User prompt (or constructed prompt including context) to sanitise.
        session_id
            Identifier for the current chat session, used to annotate logs.
        """

        endpoint = f"{MODEL_ARMOR_API_BASE}/{self.prompt_template}:sanitizeUserPrompt"
        payload = {"userPromptData": {"text": prompt}}
        try:
            response = self.session.post(endpoint, json=payload, timeout=20)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ModelArmorError("Model Armor prompt sanitization failed") from exc

        result = response.json()
        if _is_blocked(result):
            raise SafetyViolationError("Model Armor blocked the prompt")

        sanitized = (
            result.get("sanitizationResult", {}).get("sanitizedText")
            or result.get("sanitizedPrompt", {}).get("text")
            if isinstance(result.get("sanitizedPrompt"), dict)
            else None
        )
        log_event("prompt_sanitized", session_id, sanitized=bool(sanitized))
        return sanitized or prompt

    def sanitize_response(self, response_text: str, session_id: str) -> str:
        """Sanitise the model response prior to returning it to the caller.

        Parameters
        ----------
        response_text
            The generated answer from the LLM.
        session_id
            Identifier for the current chat session, used to annotate logs.
        """

        endpoint = f"{MODEL_ARMOR_API_BASE}/{self.response_template}:sanitizeModelResponse"
        payload = {"modelResponseData": {"text": response_text}}
        try:
            response = self.session.post(endpoint, json=payload, timeout=20)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ModelArmorError("Model Armor response sanitization failed") from exc

        result = response.json()
        if _is_blocked(result):
            raise SafetyViolationError("Model Armor blocked the response")

        sanitized = (
            result.get("sanitizationResult", {}).get("sanitizedText")
            or result.get("sanitizedResponse", {}).get("text")
            if isinstance(result.get("sanitizedResponse"), dict)
            else None
        )
        log_event("response_sanitized", session_id, sanitized=bool(sanitized))
        return sanitized or response_text


armor_client: Optional[ModelArmorClient] = None
if MODEL_ARMOR_PROMPT_TEMPLATE:
    try:
        armor_client = ModelArmorClient(
            prompt_template=MODEL_ARMOR_PROMPT_TEMPLATE,
            response_template=MODEL_ARMOR_RESPONSE_TEMPLATE,
        )
    except Exception as exc:  # pragma: no cover - fallback allows service to run without armor
        logging.getLogger("ads_snow_agent").warning("Model Armor initialization failed: %s", exc)
        armor_client = None


# ---------------------------------------------------------------------------
# Heuristic validation helpers
# ---------------------------------------------------------------------------
def validate_prompt(message: str) -> None:
    """Reject prompts containing disallowed topics or obvious sensitive data.

    Parameters
    ----------
    message
        User-supplied prompt text.
    """

    lower_msg = message.lower()
    if not any(topic in lower_msg for topic in ALLOWED_TOPICS):
        raise PromptSafetyError("Unsupported topic for ADS agent")
    for label, pattern in SENSITIVE_PATTERNS.items():
        if pattern.search(message):
            raise PromptSafetyError(f"Sensitive data detected: {label}")


def validate_response(answer: str) -> None:
    """Perform post-generation checks to ensure the answer is policy-compliant.

    Parameters
    ----------
    answer
        Model-generated text ready to return to the user.
    """

    for label, pattern in SENSITIVE_PATTERNS.items():
        if pattern.search(answer):
            raise PromptSafetyError(f"Sensitive data detected in response: {label}")


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
@app.get("/healthz")
def healthz():
    """Liveness probe endpoint."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Handle an interactive chat turn from the frontend.

    Parameters
    ----------
    request
        Pydantic model containing the user session identifier and message.
    """

    if not INDEX_ENDPOINT_NAME_ENV:
        raise HTTPException(status_code=500, detail="Matching Engine endpoint not configured")

    session_id = request.session_id
    user_message = request.message
    log_event("chat_received", session_id, prompt=user_message)

    try:
        # Validate and optionally sanitise the user prompt
        validate_prompt(user_message)
        log_event("prompt_validated", session_id)

        sanitized_prompt = user_message
        if armor_client:
            sanitized_prompt = armor_client.sanitize_prompt(user_message, session_id=session_id)

        # Retrieve relevant neighbours from Matching Engine
        neighbors = endpoint.match(
            deployed_index_id=endpoint.deployed_indexes[0].id,
            queries=[sanitized_prompt],
            num_neighbors=6,
        )[0]
        context_blocks = []
        sources = []
        for neighbor in neighbors:
            meta = neighbor.metadata or {}
            text = meta.get("text", "")
            source = meta.get("source", "unknown")
            context_blocks.append(f"Source: {source}\n{text}")
            sources.append(source)
        log_event("retrieval_completed", session_id, neighbors=len(neighbors), sources=sources)

        # Assemble prompt and generate
        prompt = sanitized_prompt + "\n\nContext:\n" + "\n\n".join(context_blocks)
        if armor_client:
            prompt = armor_client.sanitize_prompt(prompt, session_id=session_id)

        response = model.generate_content(
            [prompt],
            generation_config=GenerationConfig(temperature=0.2, max_output_tokens=512),
        )
        answer = response.text
        log_event("generation_completed", session_id)

        # Post-generation safety checks
        if armor_client:
            answer = armor_client.sanitize_response(answer, session_id=session_id)
        validate_response(answer)
        log_event("safety_checks_passed", session_id)

        log_event("chat_completed", session_id, answer=answer, sources=sources)
        return ChatResponse(answer=answer, sources=sources)

    except SafetyViolationError as unsafe:
        log_event("model_armor_block", session_id, severity="WARNING", detail=str(unsafe))
        raise HTTPException(status_code=422, detail="Prompt blocked by safety system") from unsafe
    except ModelArmorError as armor_error:
        log_event("model_armor_error", session_id, severity="ERROR", detail=str(armor_error))
        raise HTTPException(status_code=500, detail="Safety system error") from armor_error
    except PromptSafetyError as unsafe:
        log_event("prompt_rejected", session_id, severity="WARNING", detail=str(unsafe))
        raise HTTPException(status_code=422, detail=str(unsafe)) from unsafe
    except Exception as exc:  # pragma: no cover - defensive, ensures logging for unexpected errors
        log_event("chat_error", session_id, severity="ERROR", detail=str(exc))
        raise HTTPException(status_code=500, detail="Agent service error") from exc
