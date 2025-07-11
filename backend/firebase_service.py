import firebase_admin
from firebase_admin import credentials, firestore
import os
import logging

logger = logging.getLogger("FirebaseService")

def init_firebase():
    try:
        # Check if Firebase app already exists
        if not firebase_admin._apps:
            # Get Firebase config from environment
            firebase_config = os.getenv("FIREBASE_CONFIG")
            if not firebase_config:
                logger.warning("FIREBASE_CONFIG environment variable not set")
                return None
                
            config_dict = json.loads(firebase_config)
            cred = credentials.Certificate(config_dict)
            firebase_admin.initialize_app(cred)
            
        return firestore.client()
    except Exception as e:
        logger.error(f"Firebase initialization failed: {str(e)}")
        return None

def save_match_to_firebase(match_data: dict):
    try:
        db = init_firebase()
        if not db:
            return False
            
        doc_ref = db.collection('matches').document()
        doc_ref.set({
            'timestamp': firestore.SERVER_TIMESTAMP,
            'query_image': match_data.get('query_image', ''),
            'material': match_data.get('material', ''),
            'matches': match_data.get('matches', []),
            'impact': match_data.get('impact', {}),
            'circular_score': match_data.get('circular_score', 0)
        })
        return True
    except Exception as e:
        logger.error(f"Failed to save to Firebase: {str(e)}")
        return False