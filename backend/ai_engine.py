import os
import json
import numpy as np
import torch
import clip
import cv2
from PIL import Image
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ReStyleAI")

class ClothingMatcher:
    def __init__(self, data_dir="data/images", ann_dir="data/annotations"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.dataset_embeddings = None
        self.metadata = []
        self.categories = {}
        self.data_dir = data_dir
        self.ann_dir = ann_dir
        
    def opencv_preprocess(self, image_path: str) -> np.ndarray:
        """Enhanced image preprocessing with OpenCV"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image at {image_path}")
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = cv2.GaussianBlur(img, (5, 5), 0)
            return img
        except Exception as e:
            logger.error(f"OpenCV preprocessing failed: {str(e)}")
            return None

    def load_dataset(self):
        """Load dataset from local files"""
        try:
            logger.info("Loading dataset...")
            
            # Load category mapping
            category_path = os.path.join(self.ann_dir, "category.json")
            if not os.path.exists(category_path):
                logger.error(f"category.json not found at {category_path}")
                return
                
            with open(category_path) as f:
                self.categories = json.load(f)
                
            # Load image annotations
            images_path = os.path.join(self.ann_dir, "images.json")
            if not os.path.exists(images_path):
                logger.error(f"images.json not found at {images_path}")
                return
                
            with open(images_path) as f:
                image_annotations = json.load(f)
            
            # Get image paths
            image_paths = []
            for ann in image_annotations:
                img_path = os.path.join(self.data_dir, ann["file_name"])
                if not os.path.exists(img_path):
                    logger.warning(f"Image not found: {img_path}")
                    continue
                image_paths.append(img_path)
            
            if not image_paths:
                logger.error("No valid images found in dataset directory")
                return
                
            # Precompute embeddings
            embeddings = []
            valid_metadata = []
            
            for path, ann in zip(image_paths, image_annotations):
                try:
                    image = Image.open(path).convert("RGB")
                    preprocessed = self.preprocess(image).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        embedding = self.model.encode_image(preprocessed).cpu().numpy()
                    
                    embeddings.append(embedding.flatten())
                    
                    valid_metadata.append({
                        'path': path,
                        'category': self.categories.get(ann["category_id"], "unknown"),
                        'item_id': ann["id"]
                    })
                except Exception as e:
                    logger.warning(f"Error processing {path}: {str(e)}")
            
            if embeddings:
                self.dataset_embeddings = np.array(embeddings)
                self.metadata = valid_metadata
                logger.info(f"Successfully loaded {len(embeddings)} fashion items")
            else:
                logger.error("No valid embeddings created for dataset")
                
        except Exception as e:
            logger.error(f"Dataset loading failed: {str(e)}")

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
            logger.error(f"Embedding generation failed: {str(e)}")
            return None

    def find_matches(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find top similar items using cosine similarity"""
        try:
            if self.dataset_embeddings is None or len(self.dataset_embeddings) == 0:
                self.load_dataset()
                
            # Calculate cosine similarities
            norm_dataset = self.dataset_embeddings / np.linalg.norm(self.dataset_embeddings, axis=1, keepdims=True)
            norm_query = query_embedding / np.linalg.norm(query_embedding)
            similarities = np.dot(norm_dataset, norm_query)
            
            # Get top matches
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
            logger.error(f"Matching failed: {str(e)}")
            return []

    def calculate_impact(self, material: str, weight: float = 0.5) -> Dict[str, float]:
        """Calculate comprehensive environmental impact"""
        factors = {
            'cotton': {'co2': 8.1, 'water': 10200, 'energy': 55, 'land_use': 3.4},
            'polyester': {'co2': 5.5, 'water': 200, 'energy': 45, 'land_use': 0.2},
            'wool': {'co2': 5.4, 'water': 150000, 'energy': 60, 'land_use': 8.7},
            'silk': {'co2': 20.0, 'water': 125000, 'energy': 75, 'land_use': 10.2},
            'linen': {'co2': 2.1, 'water': 2500, 'energy': 30, 'land_use': 1.8},
            'nylon': {'co2': 7.2, 'water': 500, 'energy': 50, 'land_use': 0.5},
            'leather': {'co2': 17.0, 'water': 17000, 'energy': 65, 'land_use': 15.0},
            'denim': {'co2': 8.0, 'water': 10000, 'energy': 58, 'land_use': 4.5},
            'default': {'co2': 5.0, 'water': 1000, 'energy': 40, 'land_use': 2.0}
        }
        
        material_data = factors.get(material.lower(), factors['default'])
        return {
            'co2_saved': material_data['co2'] * weight,
            'water_saved': material_data['water'] * weight,
            'energy_saved': material_data['energy'] * weight,
            'land_saved': material_data['land_use'] * weight
        }

    def generate_real_world_equivalents(self, impact: dict) -> list:
        """Generate real-world equivalents for impact metrics"""
        equivalents = []
        
        # CO₂ equivalents
        if impact['co2_saved'] > 0:
            equivalents.append({
                'icon': '🚗',
                'label': f"Equivalent to driving {impact['co2_saved'] * 0.4:.1f} km in a car"
            })
            equivalents.append({
                'icon': '🌳',
                'label': f"Equal to {impact['co2_saved'] * 0.2:.1f} tree seedlings grown for 10 years"
            })
        
        # Water equivalents
        if impact['water_saved'] > 0:
            equivalents.append({
                'icon': '🚿',
                'label': f"Enough for {impact['water_saved'] / 10:.0f} days of drinking water"
            })
        
        # Land equivalents
        if impact['land_saved'] > 0:
            equivalents.append({
                'icon': '⚽',
                'label': f"Land area equal to {impact['land_saved'] * 1000:.0f} soccer fields"
            })
        
        return equivalents

    def calculate_circular_score(self, impact: dict, item_age: int = 2) -> float:
        """Calculate circular economy impact score (0-100)"""
        weights = {
            'co2_saved': 0.3,
            'water_saved': 0.25,
            'energy_saved': 0.2,
            'land_saved': 0.15
        }
        
        # Base score from environmental impact
        base_score = sum(impact[k] * weights[k] for k in weights) * 10
        
        # Bonus for extending item lifespan
        lifespan_bonus = min(10, item_age * 0.5)
        
        return min(100, base_score + lifespan_bonus)