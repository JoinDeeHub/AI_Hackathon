import streamlit as st
from utils import call_match_api
import tempfile
import os
from PIL import Image
import time

# Configure Streamlit
st.set_page_config(
    page_title="ReStyleAI - Circular Fashion Platform",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
:root {
    --primary: #2ecc71;
    --secondary: #3498db;
    --accent: #9b59b6;
    --dark: #2c3e50;
    --light: #f8f9fa;
    --danger: #e74c3c;
    --warning: #f39c12;
}

[data-testid="stHeader"] {
    background-color: var(--dark);
    color: white;
}

[data-testid="stSidebar"] {
    background-color: var(--dark) !important;
    color: white !important;
}

[data-testid="stSidebar"] .sidebar-content {
    color: white !important;
}

.main-content {
    background-color: var(--light);
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.upload-container {
    border: 2px dashed #ccc;
    border-radius: 5px;
    padding: 40px 20px;
    text-align: center;
    margin-bottom: 20px;
    background-color: white;
    transition: all 0.3s;
}

.upload-container:hover {
    border-color: var(--primary);
    background-color: #f0fdf4;
}

.item-card {
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    transition: transform 0.3s;
    background: white;
    height: 100%;
}

.item-card:hover {
    transform: translateY(-5px);
}

.item-card img {
    height: 200px;
    object-fit: cover;
    width: 100%;
}

.item-card-content {
    padding: 15px;
}

.item-card h4 {
    margin-top: 0;
    margin-bottom: 5px;
    font-size: 1.1em;
    text-align: center;
}

.match-card {
    background: white;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    margin-bottom: 15px;
}

.impact-badge {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: bold;
    margin-right: 5px;
    margin-bottom: 5px;
    background-color: var(--primary);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Initialize ClothingMatcher in session state
if 'matcher' not in st.session_state:
    st.session_state.matcher = None
    st.session_state.dataset_loaded = False

# Sidebar with company info
with st.sidebar:
    st.image("ReStyleAI_Logo.png", use_container_width=True)
    st.subheader("Circular Fashion Platform")
    st.markdown("""
    **Transform your wardrobe sustainably:**
    - ♻️ Reduce textile waste
    - 💧 Save precious resources
    - 🌱 Support ethical fashion
    """)
    st.divider()
    st.markdown("""
    ### How It Works
    1. Upload your clothing item
    2. Select material type
    3. Discover sustainable matches
    4. See your environmental impact
    """)
    st.divider()
    st.divider()
    st.caption("© 2025 ReStyleAI | Driving Circular Economy in Fashion")

# Main content
st.title("♻️ ReStyleAI: AI-Powered Circular Fashion Platform")
st.caption("Sustainable Fashion Matching with Advanced AI")

# Initialize matcher if not loaded
if not st.session_state.dataset_loaded:
    with st.spinner("Initializing AI system and loading fashion database..."):
        # Simulate loading time
        time.sleep(2)
        
        # Create mock metadata for preview
        st.session_state.matcher = type('', (), {})()
        st.session_state.matcher.metadata = [
            {'path': "https://images.unsplash.com/photo-1525507119028-ed4c629a60a3", 'category': "Dress", 'item_id': "DR-001"},
            {'path': "https://images.unsplash.com/photo-1591047139829-d91aecb6caea", 'category': "Top", 'item_id': "TP-045"},
            {'path': "https://images.unsplash.com/photo-1551232864-3f0890e580d9", 'category': "Skirt", 'item_id': "SK-102"},
            {'path': "https://images.unsplash.com/photo-1618354691373-d851c5c3a990", 'category': "Activewear", 'item_id': "AW-112"},
            {'path': "https://images.unsplash.com/photo-1534030347209-467a5b0ad3e6", 'category': "Jacket", 'item_id': "JK-091"}
        ]
        st.session_state.dataset_loaded = True

# Fashion Database Preview
st.subheader("Fashion Database Preview")
st.caption("Explore our curated collection of sustainable fashion items")

# Display database preview
if st.session_state.matcher and st.session_state.matcher.metadata:
    sample_items = st.session_state.matcher.metadata
    cols = st.columns(3)
    for i, item in enumerate(sample_items[:6]):
        with cols[i % 3]:
            try:
                st.markdown(f"""
                <div class="item-card">
                    <img src="{item['path']}" alt="{item['category']}">
                    <div class="item-card-content">
                        <h4>{item['category']}</h4>
                        <p style="text-align: center; font-size: 0.9em;">ID: {item['item_id']}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Couldn't load image: {item['path']}")

# Divider
st.divider()

# Style Matching Engine
st.subheader("Style Matching Engine")

# Main content container
with st.container():
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Upload clothing item")
        
        # Upload section with drag-and-drop style
        with st.container():
            st.markdown("""
            <div class="upload-container">
                <p>Drag and drop file here</p>
                <p>Limit 200MB per file • JPG, PNG, JPEG</p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "", 
                type=["jpg", "png", "jpeg"], 
                label_visibility="collapsed"
            )
        
        # Material selection
        material = st.selectbox(
            "Select material", 
            ["Cotton", "Polyester", "Wool", "Silk", "Linen", "Nylon", "Leather", "Denim"],
            index=0
        )
        
        # Weight slider
        weight = st.slider("Item weight (kg)", 0.1, 5.0, 0.5, 0.1)
        
        # Process button
        if st.button("Find Sustainable Matches", type="primary", use_container_width=True):
            if uploaded_file:
                with st.spinner("Analyzing item and calculating environmental impact..."):
                    response = call_match_api(uploaded_file, material, weight)
                    
                if response:
                    st.session_state.api_response = response
                    st.success("Found sustainable matches!")
            else:
                st.warning("Please upload an image first")

    # Display results if available
    if 'api_response' in st.session_state:
        response = st.session_state.api_response
        
        with col2:
            st.markdown("### Top Sustainable Matches")
            
            # Display matches
            for i, match in enumerate(response['matches'][:3]):
                with st.container():
                    st.markdown(f"""
                        <div class="match-card">
                            <h4>{match['category']}</h4>
                                <div style="margin: 10px 0;">
                                    <span class="impact-badge">Match: {match['similarity']*100:.1f}%</span>
                                    <span class="impact-badge">♻️ Sustainable</span>
                                </div>
                        </div>                
                    """, unsafe_allow_html=True)
            
            # Impact summary
            impact = response['impact']
            st.markdown("### Environmental Impact")
            col_a, col_b = st.columns(2)
            col_a.metric("CO₂ Saved", f"{impact['co2_saved']:.1f} kg")
            col_b.metric("Water Saved", f"{impact['water_saved']:.0f} L")
            
            # Circular economy score
            st.markdown(f"### Circular Economy Score: `{response['circular_score']:.0f}/100`")
            st.progress(response['circular_score']/100, text=f"{response['circular_score']:.0f}% circular")
            
            # Real-world equivalents
            with st.expander("Real-world Impact Equivalents"):
                for eq in response['equivalents']:
                    st.write(f"{eq['icon']} {eq['label']}")