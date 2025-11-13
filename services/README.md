# ADS Snow Agent API

FastAPI backend that powers the Alaska Department of Snow virtual assistant. Provides the `/chat` endpoint with prompt sanitization, RAG retrieval, and Gemini generation.

## Prerequisites

- Python 3.11 (for local development)
- Vertex AI project + Matching Engine endpoint
- Model Armor prompt template (optional but recommended)

## Environment Variables

- `GOOGLE_CLOUD_PROJECT`: GCP project ID.
- `VERTEXAI_LOCATION`: Vertex AI region (e.g. `us-central1`).
- `VERTEX_MATCHING_ENGINE_ENDPOINT_ID`: Matching Engine endpoint ID returned by the notebook (for example, `ads-faq-endpoint`). The service derives the full resource name using the project and region.
- `MODEL_ARMOR_PROMPT_TEMPLATE`: Prompt template resource created in the notebook. The default notebook flow produces `projects/<project-id>/locations/<region>/templates/ads-snow-prompt-template`.
- `MODEL_ARMOR_RESPONSE_TEMPLATE`: Response template resource created in the notebook (defaults to the prompt template if you do not configure a separate template). The default notebook flow produces `projects/<project-id>/locations/<region>/templates/ads-snow-response-template`.

> **Tip:** After running the notebook cells that create the Matching Engine index and Model Armor templates, export the values for reuse:
>
> ```bash
> MATCHING_ENGINE_ENDPOINT_ID="ads-faq-endpoint"   # printed by initialize_dialogflow_datastore()
> MODEL_ARMOR_PROMPT_TEMPLATE="projects/$PROJECT_ID/locations/$REGION/templates/ads-snow-prompt-template"
> MODEL_ARMOR_RESPONSE_TEMPLATE="projects/$PROJECT_ID/locations/$REGION/templates/ads-snow-response-template"
> ```

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

## Cloud Run Deployment

```bash
chmod +x deploy.sh
PROJECT_ID=qwiklabs-gcp-04-ee8165cd97c8 \
REGION=us-central1 \
MATCHING_ENGINE_ENDPOINT_ID="ads-faq-endpoint" \
MODEL_ARMOR_PROMPT_TEMPLATE="projects/$PROJECT_ID/locations/$REGION/templates/ads-snow-prompt-template" \
MODEL_ARMOR_RESPONSE_TEMPLATE="projects/$PROJECT_ID/locations/$REGION/templates/ads-snow-response-template" \
./deploy.sh
```

Or run the commands manually:

```bash
PROJECT_ID=qwiklabs-gcp-04-ee8165cd97c8
REGION=us-central1
SERVICE=ads-snow-agent-api
IMAGE=gcr.io/$PROJECT_ID/$SERVICE:latest
MATCHING_ENGINE_ENDPOINT_ID="ads-faq-endpoint"   # from notebook output
MODEL_ARMOR_PROMPT_TEMPLATE="projects/$PROJECT_ID/locations/$REGION/templates/ads-snow-prompt-template"
MODEL_ARMOR_RESPONSE_TEMPLATE="projects/$PROJECT_ID/locations/$REGION/templates/ads-snow-response-template"

# Build container (Mac/Apple Silicon users should retain the linux/amd64 target)
docker buildx build --platform=linux/amd64 -t $IMAGE services
# or use Cloud Build:
# gcloud builds submit services --tag $IMAGE

# Deploy container
gcloud run deploy $SERVICE \
  --image $IMAGE \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars \
      GOOGLE_CLOUD_PROJECT=$PROJECT_ID,\
      VERTEXAI_LOCATION=$REGION,\
      VERTEX_MATCHING_ENGINE_ENDPOINT_ID=$MATCHING_ENGINE_ENDPOINT_ID,\
      MODEL_ARMOR_PROMPT_TEMPLATE=$MODEL_ARMOR_PROMPT_TEMPLATE,\
      MODEL_ARMOR_RESPONSE_TEMPLATE=$MODEL_ARMOR_RESPONSE_TEMPLATE
```

> Remove the Model Armor variables if you are not using the service. Ensure IAM for Cloud Run service account allows access to Vertex AI and Matching Engine.
