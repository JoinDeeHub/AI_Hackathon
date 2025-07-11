from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from .ai_engine import ClothingMatcher
from .firebase_service import save_match_to_firebase
import numpy as np
import tempfile
import os
import logging
import traceback
import time
import uvicorn
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ReStyleAI-Backend")

app = FastAPI(
    title="ReStyleAI API",
    description="Backend service for sustainable fashion matching",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI engine
matcher = ClothingMatcher()
health_status = {
    "status": "starting",
    "model_loaded": False,
    "dataset_loaded": False,
    "items_loaded": 0,
    "services": {
        "firebase": False,
        "clip_model": False,
        "dataset": False
    },
    "message": "Initializing..."
}

@app.on_event("startup")
async def startup_event():
    """Load dataset on startup with health status tracking"""
    try:
        # Update health status
        health_status["status"] = "loading_model"
        health_status["message"] = "Loading AI model..."
        logger.info("Loading AI model...")
        
        # Initialize model
        health_status["services"]["clip_model"] = matcher.model is not None
        health_status["message"] = "AI model loaded"
        logger.info("AI model loaded")
        
        # Update health status
        health_status["status"] = "loading_dataset"
        health_status["message"] = "Loading fashion dataset..."
        logger.info("Loading fashion dataset...")
        
        # Load dataset
        matcher.load_dataset()
        
        # Update health status
        health_status["dataset_loaded"] = matcher.dataset_embeddings is not None
        health_status["items_loaded"] = len(matcher.metadata) if matcher.metadata else 0
        health_status["services"]["dataset"] = health_status["dataset_loaded"]
        
        # Check Firebase status
        health_status["services"]["firebase"] = matcher.db is not None
        
        if health_status["dataset_loaded"]:
            health_status["status"] = "healthy"
            health_status["message"] = f"Loaded {health_status['items_loaded']} fashion items"
            logger.info(f"Successfully loaded {health_status['items_loaded']} fashion items")
        else:
            health_status["status"] = "degraded"
            health_status["message"] = "Dataset failed to load"
            logger.error("Dataset failed to load")
            
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["message"] = f"Startup failed: {str(e)}"
        logger.error(f"Startup failed: {str(e)}")
        logger.error(traceback.format_exc())
# Serve static files (images)
DATA_PATH = os.getenv("DATA_PATH", "/app/data")
app.mount("/static", StaticFiles(directory=os.path.join(DATA_PATH, "images")), name="static")

@app.post("/match")
async def match_item(
    image: UploadFile = File(...),
    material: str = Form("Cotton"),
    weight: float = Form(0.5),
    save_to_firebase: bool = Form(False)
):
    for match in matches:
        if not match['path'].startswith(('http://', 'https://')):
            try:
                with open(match['path'], "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    match['image_data'] = f"data:image/jpeg;base64,{encoded_string}"
            except Exception as e:
                logger.error(f"Image encoding failed: {str(e)}")
            
    """Endpoint for matching clothing items and calculating impact"""
    # Check health status before processing
    if health_status["status"] != "healthy":
        raise HTTPException(status_code=503, detail="Service unavailable: " + health_status["message"])
    
    try:
        logger.info(f"Received match request for {material} item (weight: {weight}kg)")
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            content = await image.read()
            tmp.write(content)
            tmp_path = tmp.name
            logger.info(f"Saved uploaded file to: {tmp_path}")
        
        # Process image
        logger.info("Processing image...")
        embedding = matcher.get_embedding(tmp_path)
        
        if embedding is None:
            logger.error("Image processing failed")
            raise HTTPException(status_code=400, detail="Image processing failed")
        
        # Find matches
        logger.info("Finding matches...")
        matches = matcher.find_matches(embedding)
        logger.info(f"Found {len(matches)} matches")
        
        # Calculate impact
        impact = matcher.calculate_impact(material, weight)
        logger.info(f"Calculated impact: CO2={impact['co2_saved']}kg, Water={impact['water_saved']}L")
        
        # Generate real-world equivalents
        equivalents = matcher.generate_real_world_equivalents(impact)
        
        # Calculate circular economy score
        circular_score = matcher.calculate_circular_score(impact)
        logger.info(f"Circular score: {circular_score}")
        
        # Prepare response
        response = {
            "matches": matches,
            "impact": impact,
            "equivalents": equivalents,
            "circular_score": circular_score,
            "status": "success"
        }
        
        # Save to Firebase if requested
        if save_to_firebase and health_status["services"]["firebase"]:
            try:
                logger.info("Saving to Firebase...")
                match_data = {
                    "query_image": tmp_path,
                    "material": material,
                    "matches": matches,
                    "impact": impact,
                    "circular_score": circular_score
                }
                save_match_to_firebase(match_data)
                logger.info("Firebase save successful")
            except Exception as e:
                logger.error(f"Firebase save failed: {str(e)}")
        
        # Cleanup
        os.unlink(tmp_path)
        logger.info("Temporary file removed")
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
def health_check():
    """Comprehensive health check endpoint"""
    # Add performance metrics
    health_status["memory_usage"] = "N/A"  # Placeholder for memory metrics
    
    # Add detailed service status
    health_status["services"] = {
        "clip_model": matcher.model is not None,
        "dataset": matcher.dataset_embeddings is not None,
        "firebase": matcher.db is not None,
        "matching": len(matcher.metadata) > 0 if matcher.metadata else False
    }
    
    # Update overall status based on critical services
    if not health_status["services"]["clip_model"]:
        health_status["status"] = "unhealthy"
        health_status["message"] = "AI model not loaded"
    elif not health_status["services"]["dataset"]:
        health_status["status"] = "degraded"
        health_status["message"] = "Dataset not loaded"
    elif health_status["items_loaded"] == 0:
        health_status["status"] = "degraded"
        health_status["message"] = "No fashion items loaded"
    else:
        health_status["status"] = "healthy"
        health_status["message"] = "Ready to process requests"
    
    return health_status

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "()": "uvicorn.logging.DefaultFormatter",
                    "fmt": "%(levelprefix)s %(asctime)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
            },
            "loggers": {
                "uvicorn": {"handlers": ["default"], "level": "INFO"},
                "ReStyleAI-Backend": {"handlers": ["default"], "level": "DEBUG"},
            },
        }
    )