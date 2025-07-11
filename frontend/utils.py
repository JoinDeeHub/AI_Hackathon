import requests
import streamlit as st
import tempfile
import os
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Determine environment and set API URL
if os.path.exists('/.dockerenv'):
    API_URL = "http://backend:8000"
else:
    API_URL = "http://localhost:8000"

def create_session():
    """Create a requests session with retry logic"""
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def call_match_api(uploaded_file, material, weight=0.5):
    """Call the match API endpoint with enhanced error handling"""
    try:
        # Create session with retry logic
        session = create_session()
        
        # Save uploaded file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        # Prepare files and data
        files = {"image": (uploaded_file.name, open(tmp_path, "rb"), "image/jpeg")}
        data = {
            "material": material,
            "weight": weight,
            "save_to_firebase": True
        }
        
        # Make API call
        st.info(f"🌐 Connecting to API")
        start_time = time.time()
        response = session.post(
            f"{API_URL}/match", 
            files=files, 
            data=data,
            timeout=30
        )
        
        # Cleanup
        os.unlink(tmp_path)
        
        # Log response time
        response_time = time.time() - start_time
        st.info(f"⏱️ API response received in {response_time:.2f} seconds")
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"❌ API Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"🔥 API call failed: {str(e)}")
        return None