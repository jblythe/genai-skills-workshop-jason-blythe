# ADS Agent Web

React frontend for the Alaska Department of Snow virtual assistant. The app renders a chat experience that calls the Cloud Run FastAPI backend.

## Prerequisites

- Node.js 20+
- npm (or pnpm / bun)
- A deployed backend API endpoint

## Local Development

```bash
npm install
cp env.example .env.local # edit VITE_API_URL to target your backend
npm run dev
```

## Production Build

```bash
npm install
npm run build
```

## Cloud Run Deployment

```bash
chmod +x deploy.sh
PROJECT_ID=qwiklabs-gcp-04-ee8165cd97c8 \
REGION=us-central1 \
VITE_API_URL=https://ads-snow-agent-api-<your-suffix>-uc.a.run.app \
./deploy.sh
```

Or run the commands manually:

```bash
PROJECT_ID=qwiklabs-gcp-04-ee8165cd97c8
REGION=us-central1
SERVICE=ads-agent-web
IMAGE=gcr.io/$PROJECT_ID/$SERVICE:latest
API_URL=https://ads-snow-agent-api-<your-suffix>-uc.a.run.app

# Build container (Mac/Apple Silicon users should target linux/amd64)
docker buildx build --platform=linux/amd64 \
  --build-arg VITE_API_URL=$API_URL \
  -t $IMAGE web
# or use Cloud Build:
# gcloud builds submit web --tag $IMAGE --build-arg VITE_API_URL=$API_URL

# Deploy container
gcloud run deploy $SERVICE \
  --image $IMAGE \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated
```
