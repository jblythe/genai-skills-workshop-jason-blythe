#!/usr/bin/env bash
# Deploy the ADS Snow Agent backend to Cloud Run.
#
# Requirements:
#   - gcloud CLI authenticated for the target project
#   - Vertex AI Search data store (Dialogflow data store) provisioned via the notebook
#   - (Optional) Model Armor templates created via the notebook
#
# Usage:
#   PROJECT_ID=... REGION=... VERTEX_SEARCH_SERVING_CONFIG=... ./deploy.sh
#
# Optional overrides:
#   SERVICE_NAME (default: ads-snow-agent-api)
#   IMAGE (default: gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest)
#   MODEL_ARMOR_PROMPT_TEMPLATE (default: unset)
#   MODEL_ARMOR_RESPONSE_TEMPLATE (default: value of MODEL_ARMOR_PROMPT_TEMPLATE)
set -euo pipefail

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud CLI is required" >&2
  exit 1
fi

PROJECT_ID=${PROJECT_ID:-}
REGION=${REGION:-us-central1}
SERVICE_NAME=${SERVICE_NAME:-ads-snow-agent-api}
IMAGE=${IMAGE:-gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest}
VERTEX_SEARCH_SERVING_CONFIG=${VERTEX_SEARCH_SERVING_CONFIG:-}
MODEL_ARMOR_PROMPT_TEMPLATE=${MODEL_ARMOR_PROMPT_TEMPLATE:-}
MODEL_ARMOR_RESPONSE_TEMPLATE=${MODEL_ARMOR_RESPONSE_TEMPLATE:-${MODEL_ARMOR_PROMPT_TEMPLATE}}

if [[ -z "${PROJECT_ID}" ]]; then
  echo "PROJECT_ID is required" >&2
  exit 1
fi

if [[ -z "${VERTEX_SEARCH_SERVING_CONFIG}" ]]; then
  echo "VERTEX_SEARCH_SERVING_CONFIG is required" >&2
  exit 1
fi

echo "Building backend container: ${IMAGE}" >&2
gcloud builds submit --project "${PROJECT_ID}" --tag "${IMAGE}" .

env_args=(
  "GOOGLE_CLOUD_PROJECT=${PROJECT_ID}"
  "VERTEXAI_LOCATION=${REGION}"
  "VERTEX_SEARCH_SERVING_CONFIG=${VERTEX_SEARCH_SERVING_CONFIG}"
)

if [[ -n "${MODEL_ARMOR_PROMPT_TEMPLATE}" ]]; then
  env_args+=("MODEL_ARMOR_PROMPT_TEMPLATE=${MODEL_ARMOR_PROMPT_TEMPLATE}")
fi
if [[ -n "${MODEL_ARMOR_RESPONSE_TEMPLATE}" ]]; then
  env_args+=("MODEL_ARMOR_RESPONSE_TEMPLATE=${MODEL_ARMOR_RESPONSE_TEMPLATE}")
fi

echo "Deploying Cloud Run service: ${SERVICE_NAME}" >&2

gcloud run deploy "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars "$(IFS=,; echo "${env_args[*]}")"
