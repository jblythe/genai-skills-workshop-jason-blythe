# ADS Snow Agent API

FastAPI backend that powers the Alaska Department of Snow virtual assistant. Provides the `/chat` endpoint with prompt sanitization, RAG retrieval, and Gemini generation.

## Prerequisites

- Python 3.11 (for local development)
- Vertex AI project + Dialogflow Data Store (Vertex AI Search) created via the notebook
- Model Armor prompt/response templates created via the notebook

## Environment Variables

- `GOOGLE_CLOUD_PROJECT`: GCP project ID.
- `VERTEXAI_LOCATION`: Vertex AI region (e.g. `us-central1`).
- `VERTEX_SEARCH_SERVING_CONFIG`: Fully-qualified Vertex Search serving config returned by `initialize_dialogflow_datastore()` (for example, `projects/<project>/locations/global/collections/default_collection/dataStores/ads-faq-unstructured/servingConfigs/default_serving_config`).
- `MODEL_ARMOR_PROMPT_TEMPLATE`: Prompt template resource created in the notebook (`projects/<project>/locations/<region>/templates/ads-snow-prompt-template`).
- `MODEL_ARMOR_RESPONSE_TEMPLATE`: Response template resource created in the notebook (`projects/<project>/locations/<region>/templates/ads-snow-response-template`).

> **Tip:** After running the notebook provisioning cells, export the values for reuse:
>
> ```bash
> VERTEX_SEARCH_SERVING_CONFIG="projects/$PROJECT_ID/locations/global/collections/default_collection/dataStores/ads-faq-unstructured/servingConfigs/default_serving_config"
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
export PROJECT_ID=qwiklabs-gcp-04-ee8165cd97c8
export REGION=us-central1
export VERTEX_SEARCH_SERVING_CONFIG="projects/$PROJECT_ID/locations/global/collections/default_collection/dataStores/ads-faq-unstructured/servingConfigs/default_serving_config"
export MODEL_ARMOR_PROMPT_TEMPLATE="projects/$PROJECT_ID/locations/$REGION/templates/ads-snow-prompt-template"
export MODEL_ARMOR_RESPONSE_TEMPLATE="projects/$PROJECT_ID/locations/$REGION/templates/ads-snow-response-template"
./deploy.sh
```

Or run the commands manually:

```bash
export PROJECT_ID=qwiklabs-gcp-04-ee8165cd97c8
export REGION=us-central1
export SERVICE=ads-snow-agent-api
export IMAGE=gcr.io/$PROJECT_ID/$SERVICE:latest
export VERTEX_SEARCH_SERVING_CONFIG="projects/$PROJECT_ID/locations/global/collections/default_collection/dataStores/ads-faq-unstructured/servingConfigs/default_serving_config"
export MODEL_ARMOR_PROMPT_TEMPLATE="projects/$PROJECT_ID/locations/$REGION/templates/ads-snow-prompt-template"
export MODEL_ARMOR_RESPONSE_TEMPLATE="projects/$PROJECT_ID/locations/$REGION/templates/ads-snow-response-template"

docker buildx build --platform=linux/amd64 -t $IMAGE services

gcloud run deploy $SERVICE \
  --image $IMAGE \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars \
      GOOGLE_CLOUD_PROJECT=$PROJECT_ID,\
      VERTEXAI_LOCATION=$REGION,\
      VERTEX_SEARCH_SERVING_CONFIG=$VERTEX_SEARCH_SERVING_CONFIG,\
      MODEL_ARMOR_PROMPT_TEMPLATE=$MODEL_ARMOR_PROMPT_TEMPLATE,\
      MODEL_ARMOR_RESPONSE_TEMPLATE=$MODEL_ARMOR_RESPONSE_TEMPLATE
```

> Remove the Model Armor variables if you are not using the service. Ensure the Cloud Run service account has permissions for Vertex AI Search and Model Armor.
