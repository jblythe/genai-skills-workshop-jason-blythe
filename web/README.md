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
export PROJECT_ID=qwiklabs-gcp-04-ee8165cd97c8
export REGION=us-central1
export VITE_API_URL=https://ads-snow-agent-api-i2pzc23xkq-uc.a.run.app
./deploy.sh  # builds with VITE_API_URL and sets it as a runtime env var
```

Or run the commands manually:

```bash
export PROJECT_ID=qwiklabs-gcp-04-ee8165cd97c8
export REGION=us-central1
export SERVICE=ads-agent-web
export IMAGE=gcr.io/$PROJECT_ID/$SERVICE:latest
export API_URL=https://ads-snow-agent-api-i2pzc23xkq-uc.a.run.app

gcloud builds submit web \
  --config web/cloudbuild.yaml \
  --substitutions _IMAGE=$IMAGE,_VITE_API_URL=$API_URL

gcloud run deploy $SERVICE \
  --image $IMAGE \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars VITE_API_URL=$API_URL
```
