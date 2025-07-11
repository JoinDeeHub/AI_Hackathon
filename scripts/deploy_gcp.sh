#!/bin/bash

# Set environment variables
export GCP_PROJECT_ID="your-project-id"
export SERVICE_NAME="restyle-ai"
export REGION="us-central1"

# Build Docker image
docker build -t gcr.io/$GCP_PROJECT_ID/$SERVICE_NAME .

# Push to Google Container Registry
gcloud auth configure-docker
docker push gcr.io/$GCP_PROJECT_ID/$SERVICE_NAME

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$GCP_PROJECT_ID/$SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars FIREBASE_CONFIG="$FIREBASE_CONFIG" \
  --port 8000

# Output service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')
echo "Service deployed to: $SERVICE_URL"