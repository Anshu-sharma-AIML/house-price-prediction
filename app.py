import streamlit as st
import pandas as pd
import joblib
import os

# Set page styling
st.set_page_config(
    page_title="Bangalore House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Custom Styling for premium look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #2ecc71;
        color: white;
        font-size: 18px;
        font-weight: bold;
        height: 50px;
        border-radius: 8px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #27ae60;
        box-shadow: 0 4px 15px rgba(46, 204, 113, 0.4);
    }
    .result-box {
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #2ecc71;
        margin-top: 20px;
    }
    </style>
""", unsafe_html=True)

st.title("🏠 Bangalore House Price Predictor")
st.write("Provide details of the house below to estimate its market price based on real Bangalore real estate listings data.")

# Load model files
@st.cache_resource
def load_models():
    best_model = joblib.load('best_model.joblib')
    le = joblib.load('label_encoder.joblib')
    scaler = joblib.load('scaler.joblib')
    return best_model, le, scaler

# Ensure models exist
if not os.path.exists('best_model.joblib') or not os.path.exists('label_encoder.joblib'):
    st.error("❌ Trained model files not found. Please run 'python model.py' locally first to train the model.")
else:
    try:
        best_model, le, scaler = load_models()
        
        # 1. Inputs Section
        st.subheader("🛠️ House Features")
        
        col1, col2 = st.columns(2)
        with col1:
            area = st.number_input("📏 Total Area (sqft)", min_value=200, max_value=50000, value=1200, step=100)
            bhk = st.slider("🛏️ BHK (Bedrooms)", min_value=1, max_value=10, value=2)
            
        with col2:
            locations = sorted(list(le.classes_))
            if 'Other' in locations:
                locations.remove('Other')
                locations.append('Other')
            location = st.selectbox("📍 Location / Area", locations, index=locations.index('Whitefield') if 'Whitefield' in locations else 0)
            
            bathrooms = st.slider("🚿 Bathrooms", min_value=1, max_value=10, value=2)
            
        balcony = st.slider("🏢 Balconies", min_value=0, max_value=5, value=1)
        
        # 2. Prediction Trigger
        if st.button("💰 Predict Price"):
            loc_enc = le.transform([location])[0]
            
            sample = pd.DataFrame([{
                'total_sqft': area,
                'bath': bathrooms,
                'balcony': balcony,
                'bhk': bhk,
                'location_enc': loc_enc
            }])
            
            # Predict
            model_name = type(best_model).__name__
            is_linear = any(name in model_name for name in ['LinearRegression', 'Ridge', 'Lasso'])
            
            if is_linear:
                sample_sc = scaler.transform(sample)
                predicted_lakhs = best_model.predict(sample_sc)[0]
            else:
                predicted_lakhs = best_model.predict(sample)[0]
                
            price_in_rupees = predicted_lakhs * 100000
            
            # Display Prediction
            st.markdown('<div class="result-box">', unsafe_html=True)
            st.subheader("🎉 Prediction Results")
            st.write(f"**📍 Location**: {location} | **📏 Area**: {area} sqft | **🛏️ Type**: {bhk} BHK ({bathrooms} Bath)")
            
            if price_in_rupees >= 10000000:
                crores = price_in_rupees / 10000000
                st.success(f"### Estimated Price: **₹{crores:.2f} Crore** (₹{predicted_lakhs:.2f} Lakhs)")
            else:
                st.success(f"### Estimated Price: **₹{predicted_lakhs:.2f} Lakhs** (₹{price_in_rupees:,.0f})")
                
            st.caption(f"Predicted using the best model: **{model_name}**")
            st.markdown('</div>', unsafe_html=True)
            
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
