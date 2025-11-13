#!/usr/bin/env bash
# Deploy the ADS Snow Agent frontend to Cloud Run.
#
# Requirements:
#   - gcloud CLI authenticated for the target project
#   - Backend Cloud Run URL available for VITE_API_URL
#
# Usage:
#   PROJECT_ID=... REGION=... VITE_API_URL=... ./deploy.sh
#
# Optional overrides:
#   SERVICE_NAME (default: ads-agent-web)
#   IMAGE (default: gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest)
set -euo pipefail

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud CLI is required" >&2
  exit 1
fi

PROJECT_ID=${PROJECT_ID:-}
REGION=${REGION:-us-central1}
SERVICE_NAME=${SERVICE_NAME:-ads-agent-web}
IMAGE=${IMAGE:-gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest}
VITE_API_URL=${VITE_API_URL:-}

if [[ -z "${PROJECT_ID}" ]]; then
  echo "PROJECT_ID is required" >&2
  exit 1
fi

if [[ -z "${VITE_API_URL}" ]]; then
  echo "VITE_API_URL is required" >&2
  exit 1
fi

echo "Building frontend container: ${IMAGE}" >&2
gcloud builds submit --project "${PROJECT_ID}" \
  --tag "${IMAGE}" \
  --substitutions _VITE_API_URL="${VITE_API_URL}" .

echo "Deploying Cloud Run service: ${SERVICE_NAME}" >&2

gcloud run deploy "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated
