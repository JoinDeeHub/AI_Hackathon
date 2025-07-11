import os
import json
import numpy as np
import torch
import clip
import cv2
from PIL import Image
import streamlit as st
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
import firebase_admin
from firebase_admin import credentials, firestore
import logging
import docker
import subprocess
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Dict, Any
import tempfile

# --- Firebase Initialization ---
def init_firebase():
    try:
        # Use a service account (for production, use environment variables)
        if not firebase_admin._apps:
            cred = credentials.Certificate({
                "type": "service_account",
                "project_id": os.getenv("FIREBASE_PROJECT_ID", "restyle-ai"),
                "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
                "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace('\\n', '\n'),
                "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
                "client_id": os.getenv("FIREBASE_CLIENT_ID"),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": os.getenv("FIREBASE_CERT_URL")
            })
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        logging.error(f"Firebase initialization failed: {str(e)}")
        return None

# --- ClothingMatcher Class (Core AI Engine) ---
class ClothingMatcher:
    def __init__(self, data_dir="data/images", ann_dir="data/annotations"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.dataset_embeddings = None
        self.metadata = []
        self.categories = {}
        self.data_dir = data_dir
        self.ann_dir = ann_dir
        self.db = init_firebase()
        
    def opencv_preprocess(self, image_path: str) -> np.ndarray:
        """Enhanced image preprocessing with OpenCV"""
        try:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Advanced preprocessing pipeline
            img = cv2.resize(img, (224, 224))
            img = cv2.GaussianBlur(img, (5, 5), 0)
            img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            return img
        except Exception as e:
            logging.error(f"OpenCV preprocessing failed: {str(e)}")
            return None

    def load_dataset(self):
        """Load dataset with Firebase fallback"""
        try:
            # Try loading from Firebase first
            if self.db:
                docs = self.db.collection('fashion_items').stream()
                self.metadata = []
                embeddings = []
                
                for doc in docs:
                    item = doc.to_dict()
                    self.metadata.append({
                        'path': item['image_url'],
                        'category': item['category'],
                        'item_id': doc.id
                    })
                    embeddings.append(np.array(item['embedding']))
                
                if embeddings:
                    self.dataset_embeddings = np.array(embeddings)
                    logging.info(f"Loaded {len(embeddings)} items from Firebase")
                    return
                
            # Fallback to local dataset
            logging.info("Loading from local dataset")
            with open(os.path.join(self.ann_dir, "category.json")) as f:
                self.categories = json.load(f)
                
            with open(os.path.join(self.ann_dir, "images.json")) as f:
                image_annotations = json.load(f)
            
            image_paths = [os.path.join(self.data_dir, ann["file_name"]) 
                          for ann in image_annotations]
            
            embeddings = []
            for path, ann in zip(image_paths, image_annotations):
                try:
                    image = Image.open(path).convert("RGB")
                    preprocessed = self.preprocess(image).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        embedding = self.model.encode_image(preprocessed).cpu().numpy()
                    
                    embeddings.append(embedding.flatten())
                    
                    self.metadata.append({
                        'path': path,
                        'category': self.categories.get(ann["category_id"], "unknown"),
                        'item_id': ann["id"]
                    })
                except Exception as e:
                    logging.warning(f"Error processing {path}: {str(e)}")
            
            if embeddings:
                self.dataset_embeddings = np.array(embeddings)
                logging.info(f"Loaded {len(embeddings)} fashion items")
            else:
                raise ValueError("No valid images found")
                
            # Cache to Firebase if available
            if self.db:
                batch = self.db.batch()
                for i, (meta, emb) in enumerate(zip(self.metadata, embeddings)):
                    doc_ref = self.db.collection('fashion_items').document(f"item_{i}")
                    batch.set(doc_ref, {
                        'category': meta['category'],
                        'image_url': meta['path'],
                        'embedding': emb.tolist()
                    })
                batch.commit()
                logging.info("Dataset cached to Firebase")
                
        except Exception as e:
            logging.error(f"Dataset loading failed: {str(e)}")
            st.error(f"Dataset loading failed: {str(e)}")

    def get_embedding(self, image) -> np.ndarray:
        """Get CLIP embedding for an image"""
        try:
            if isinstance(image, str):  # Path
                img = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):  # OpenCV image
                img = Image.fromarray(image)
            else:  # PIL Image
                img = image
                
            preprocessed = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return self.model.encode_image(preprocessed).cpu().numpy().flatten()
        except Exception as e:
            logging.error(f"Embedding generation failed: {str(e)}")
            return None

    def find_matches(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find top similar items using cosine similarity"""
        try:
            if self.dataset_embeddings is None:
                self.load_dataset()
                
            similarities = cosine_similarity([query_embedding], self.dataset_embeddings)[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            return [
                {
                    'path': self.metadata[i]['path'],
                    'similarity': float(similarities[i]),
                    'category': self.metadata[i]['category'],
                    'item_id': self.metadata[i]['item_id']
                }
                for i in top_indices
            ]
        except Exception as e:
            logging.error(f"Matching failed: {str(e)}")
            return []

    def calculate_impact(self, material: str, weight: float = 0.5) -> Dict[str, float]:
        """Calculate environmental impact with enhanced accuracy"""
        # Comprehensive impact factors (kg CO2 per kg of material)
        factors = {
            'cotton': {'co2': 8.1, 'water': 10200},
            'polyester': {'co2': 5.5, 'water': 200},
            'wool': {'co2': 5.4, 'water': 150000},
            'silk': {'co2': 20.0, 'water': 125000},
            'linen': {'co2': 2.1, 'water': 2500},
            'nylon': {'co2': 7.2, 'water': 500},
            'viscose': {'co2': 3.0, 'water': 3000},
            'leather': {'co2': 17.0, 'water': 17000},
            'denim': {'co2': 8.0, 'water': 10000},
            'default': {'co2': 5.0, 'water': 1000}
        }
        
        material_data = factors.get(material.lower(), factors['default'])
        return {
            'co2_saved': material_data['co2'] * weight,
            'water_saved': material_data['water'] * weight
        }

# --- Streamlit Frontend ---
def streamlit_app():
    st.set_page_config(
        page_title="ReStyleAI - Circular Fashion Platform",
        page_icon="♻️",
        layout="wide"
    )
    
    st.title("♻️ ReStyleAI: AI-Powered Circular Fashion Platform")
    st.caption("Sustainable Fashion Matching with Advanced AI")
    
    # Initialize matcher
    if 'matcher' not in st.session_state:
        st.session_state.matcher = ClothingMatcher()
        with st.spinner("Initializing AI system..."):
            st.session_state.matcher.load_dataset()
    
    # Dataset preview
    st.subheader("Fashion Database Preview")
    if st.session_state.matcher.metadata:
        sample_items = st.session_state.matcher.metadata[:6]
        cols = st.columns(3)
        for i, item in enumerate(sample_items):
            with cols[i % 3]:
                try:
                    if item['path'].startswith('http'):
                        st.image(item['path'], caption=f"{item['category']} (ID: {item['item_id']})", use_container_width=True)
                    else:
                        img = Image.open(item['path'])
                        st.image(img, caption=f"{item['category']} (ID: {item['item_id']})", use_container_width=True)
                except:
                    st.warning(f"Couldn't load image: {item['path']}")
    
    # Main interface
    st.divider()
    st.subheader("Style Matching Engine")
    
    # Upload section
    uploaded_file = st.file_uploader("Upload clothing item", type=["jpg", "png", "jpeg"])
    material = st.selectbox("Select material", 
                          ["Cotton", "Polyester", "Wool", "Silk", "Linen", "Nylon", "Leather", "Denim"])
    
    if uploaded_file:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Your Item", width=300)
            
            # OpenCV preprocessing for demo
            st.caption("Enhanced Image Processing")
            opencv_img = st.session_state.matcher.opencv_preprocess(tmp_path)
            if opencv_img is not None:
                st.image(opencv_img, caption="Processed with OpenCV", width=300)
            
            # Calculate impact
            impact = st.session_state.matcher.calculate_impact(material)
            st.metric("🚫 CO₂ Prevented", f"{impact['co2_saved']:.1f} kg")
            st.metric("💧 Water Saved", f"{impact['water_saved']:.0f} liters")
            
            # Sustainability facts
            with st.expander("Environmental Impact Details"):
                st.info(f"Recycling this {material.lower()} item saves:")
                st.write(f"- 🌍 Equivalent to driving {impact['co2_saved']*0.4:.1f} km in a car")
                st.write(f"- 💧 Enough water for {impact['water_saved']/10:.0f} days of drinking")
                st.write(f"- 🌳 Equivalent to {impact['co2_saved']*0.2:.1f} tree seedlings grown for 10 years")
        
        with col2:
            st.subheader("Top Sustainable Matches")
            # Get embedding
            embedding = st.session_state.matcher.get_embedding(tmp_path)
            
            if embedding is not None:
                matches = st.session_state.matcher.find_matches(embedding)
                
                if matches:
                    for match in matches:
                        cols = st.columns([1, 4])
                        with cols[0]:
                            try:
                                st.image(match['path'], width=150)
                            except:
                                st.warning("Image unavailable")
                        with cols[1]:
                            similarity = match['similarity']
                            st.progress(similarity, text=f"{similarity*100:.1f}% match")
                            st.caption(f"**Category**: {match['category']}")
                            st.caption(f"**Item ID**: {match['item_id']}")
                            st.caption(f"**Style similarity**: {similarity*100:.1f}%")
                    
                    # Impact visualization
                    impact_data = {
                        'Metric': ['CO₂ Saved', 'Water Saved'],
                        'Amount': [impact['co2_saved'], impact['water_saved']],
                        'Color': ['#2ecc71', '#3498db']
                    }
                    fig = px.bar(impact_data, x='Metric', y='Amount', color='Color',
                                 text='Amount', color_discrete_map="identity",
                                 title="Environmental Impact per Transaction")
                    fig.update_layout(
                        showlegend=False,
                        yaxis_title="Amount Saved",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No matches found. Try another image.")
            else:
                st.error("Failed to process image")

# --- FastAPI Backend ---
app = FastAPI(
    title="ReStyleAI API",
    description="Backend service for sustainable fashion matching",
    version="1.0.0"
)

matcher = ClothingMatcher()
matcher.load_dataset()

@app.post("/match")
async def match_item(
    image: UploadFile = File(...),
    material: str = Form("Cotton")
):
    """Endpoint for matching clothing items"""
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            content = await image.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Process image
        opencv_img = matcher.opencv_preprocess(tmp_path)
        embedding = matcher.get_embedding(tmp_path)
        
        if embedding is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Image processing failed"}
            )
        
        # Find matches
        matches = matcher.find_matches(embedding)
        
        # Calculate impact
        impact = matcher.calculate_impact(material)
        
        # Cleanup
        os.unlink(tmp_path)
        
        return {
            "matches": matches,
            "impact": impact,
            "status": "success"
        }
    except Exception as e:
        logging.error(f"API error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": matcher.model is not None}

# --- Deployment Script ---
def deploy_to_gcp():
    """Deploy to Google Cloud Platform using Docker"""
    try:
        # Build Docker image
        docker_client = docker.from_env()
        image, build_log = docker_client.images.build(
            path=".",
            tag="restyle-ai:latest"
        )
        
        # Push to GCP Container Registry
        subprocess.run([
            "gcloud", "auth", "configure-docker"
        ], check=True)
        
        subprocess.run([
            "docker", "tag", "restyle-ai:latest", 
            f"gcr.io/{os.getenv('GCP_PROJECT_ID')}/restyle-ai:latest"
        ], check=True)
        
        subprocess.run([
            "docker", "push", 
            f"gcr.io/{os.getenv('GCP_PROJECT_ID')}/restyle-ai:latest"
        ], check=True)
        
        # Deploy to Cloud Run
        deploy_cmd = [
            "gcloud", "run", "deploy", "restyle-ai",
            "--image", f"gcr.io/{os.getenv('GCP_PROJECT_ID')}/restyle-ai:latest",
            "--platform", "managed",
            "--region", "us-central1",
            "--allow-unauthenticated"
        ]
        
        if os.getenv("GCP_SERVICE_ACCOUNT"):
            deploy_cmd.extend(["--service-account", os.getenv("GCP_SERVICE_ACCOUNT")])
            
        subprocess.run(deploy_cmd, check=True)
        
        return True
    except Exception as e:
        logging.error(f"Deployment failed: {str(e)}")
        return False

# --- Main Execution ---
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ReStyleAI System")
    parser.add_argument("--mode", choices=["streamlit", "api", "deploy"], default="streamlit", 
                       help="Run mode: streamlit (frontend), api (backend), deploy (to GCP)")
    args = parser.parse_args()
    
    if args.mode == "streamlit":
        streamlit_app()
    elif args.mode == "api":
        uvicorn.run(app, host="0.0.0.0", port=8000)
    elif args.mode == "deploy":
        if deploy_to_gcp():
            print("Deployment successful!")
        else:
            print("Deployment failed. Check logs.")