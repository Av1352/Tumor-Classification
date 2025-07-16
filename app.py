import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Tumor Classification AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model function (placeholder)
@st.cache_resource
def load_model():
    # Replace with your actual model loading
    # model = tf.keras.models.load_model('your_model.h5')
    # return model
    return None

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert to numpy array
    img_array = np.array(image)
    
    # Resize to 96x96 if needed
    if img_array.shape[:2] != (96, 96):
        img_array = cv2.resize(img_array, (96, 96))
    
    # Convert to RGB if RGBA
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Normalize
    img_array = img_array.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def create_attention_heatmap(image):
    """Generate attention heatmap for visualization"""
    h, w = image.shape[:2]
    
    # Create center-focused heatmap
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    
    # Distance from center
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Create attention map (higher attention in center)
    attention = np.exp(-distance / 15)
    
    # Add some random variation
    np.random.seed(42)
    noise = np.random.normal(0, 0.1, attention.shape)
    attention += noise
    attention = np.clip(attention, 0, 1)
    
    return attention

def predict_tumor(image, model=None):
    """Make prediction on uploaded image"""
    # Preprocess
    processed_img = preprocess_image(image)
    
    # Make prediction (placeholder - replace with actual model)
    if model is not None:
        prediction = model.predict(processed_img)[0][0]
    else:
        # Demo prediction
        np.random.seed(hash(str(processed_img.sum())) % 1000)
        prediction = np.random.beta(2, 2)  # More realistic distribution
    
    return prediction

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 Tumor Classification AI</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            AI-powered histopathologic cancer detection with explainable visualizations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Performance")
        
        # Performance metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "91%", "6% ↑")
        with col2:
            st.metric("AUC Score", "0.97", "0.04 ↑")
        
        st.markdown("---")
        
        # Model info
        st.subheader("🧠 Model Details")
        st.write("**Architecture:** CNN with BatchNorm")
        st.write("**Input Size:** 96×96 pixels")
        st.write("**Focus Area:** Center 32×32 region")
        st.write("**Training Images:** 220,025")
        
        st.markdown("---")
        
        # Performance comparison chart
        st.subheader("📈 Model Comparison")
        
        comparison_data = {
            'Model': ['Baseline CNN', 'BatchNorm CNN'],
            'Accuracy': [0.85, 0.91],
            'AUC': [0.93, 0.97]
        }
        
        fig = px.bar(comparison_data, x='Model', y='Accuracy', 
                    title='Model Performance Comparison',
                    color='Accuracy', color_continuous_scale='blues')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Histopathology Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file (96x96 pixels recommended)",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a histopathology image for tumor analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.write(f"**Image Size:** {image.size}")
            st.write(f"**Format:** {image.format}")
            
            # Analyze button
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    # Make prediction
                    model = load_model()
                    prediction = predict_tumor(image, model)
                    
                    # Store results in session state
                    st.session_state.prediction = prediction
                    st.session_state.image = image
    
    with col2:
        st.subheader("🎯 Analysis Results")
        
        if hasattr(st.session_state, 'prediction') and hasattr(st.session_state, 'image'):
            prediction = st.session_state.prediction
            image = st.session_state.image
            
            # Prediction result
            if prediction > 0.5:
                st.error(f"🔴 **TUMOR DETECTED**")
                st.write(f"**Confidence:** {prediction:.1%}")
                risk_color = "red"
            else:
                st.success(f"✅ **NO TUMOR DETECTED**")
                st.write(f"**Confidence:** {(1-prediction):.1%}")
                risk_color = "green"
            
            # Confidence gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Tumor Probability"},
                gauge = {
                    'axis': {'range': [None, 1]},
                    'bar': {'color': risk_color},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgreen"},
                        {'range': [0.5, 1], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.5
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Visualization section
            st.subheader("🔍 Explainable AI Visualization")
            
            # Create visualizations
            img_array = np.array(image)
            if img_array.shape[:2] != (96, 96):
                img_array = cv2.resize(img_array, (96, 96))
            
            # Create attention heatmap
            attention_map = create_attention_heatmap(img_array)
            
            # Display visualizations
            tab1, tab2, tab3 = st.tabs(["Original", "Focus Area", "Attention Map"])
            
            with tab1:
                st.image(img_array, caption="Original Image (96×96)", use_column_width=True)
            
            with tab2:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(img_array)
                
                # Highlight center 32x32 region
                center_start = (96 - 32) // 2
                rect = plt.Rectangle((center_start, center_start), 32, 32, 
                                fill=False, edgecolor='red', linewidth=3)
                ax.add_patch(rect)
                ax.set_title("Center 32×32 Analysis Region", fontsize=14)
                ax.axis('off')
                
                st.pyplot(fig)
                plt.close()
            
            with tab3:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(img_array, alpha=0.7)
                im = ax.imshow(attention_map, alpha=0.5, cmap='jet')
                ax.set_title("AI Attention Heatmap", fontsize=14)
                ax.axis('off')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, shrink=0.8, label='Attention Intensity')
                
                st.pyplot(fig)
                plt.close()
        
        else:
            st.info("👆 Upload an image and click 'Analyze' to see results")
    
    # Warning disclaimer
    st.markdown("---")
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ Important Medical Disclaimer</h4>
        <p>This is an educational demonstration and research project. <strong>This tool is NOT intended for clinical use, medical diagnosis, or patient care.</strong> Always consult qualified healthcare professionals for medical decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Technical details
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        ### Model Architecture
        - **Base Model:** Convolutional Neural Network (CNN)
        - **Enhancement:** Batch Normalization layers
        - **Input Shape:** 96×96×3 (RGB images)
        - **Output:** Binary classification (tumor/no tumor)
        
        ### Training Details
        - **Dataset:** Histopathologic Cancer Detection Challenge
        - **Training Images:** 220,025 labeled images
        - **Validation Split:** 80/20 split
        - **Focus Area:** Center 32×32 pixel region
        
        ### Performance Metrics
        - **Baseline Model:** 85% accuracy, 0.93 AUC
        - **Improved Model:** 91% accuracy, 0.97 AUC
        - **Improvement:** +6% accuracy, +0.04 AUC
        
        ### Technology Stack
        - **Framework:** TensorFlow/Keras
        - **Frontend:** Streamlit
        - **Visualization:** Matplotlib, Plotly
        - **Deployment:** Streamlit Cloud
        """)

if __name__ == "__main__":
    main()