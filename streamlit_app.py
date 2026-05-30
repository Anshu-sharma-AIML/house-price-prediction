import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor — Bangalore",
    page_icon="🏠",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background-color: #0f0e0c; }

.title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    background: linear-gradient(135deg, #e8c97a, #c9a84c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0;
}

.subtitle {
    text-align: center;
    color: #8a8270;
    font-size: 0.9rem;
    margin-bottom: 2rem;
}

.result-box {
    background: linear-gradient(135deg, #0d1a0f, #0f1a0d);
    border: 1px solid #2a4a30;
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    margin-top: 1rem;
}

.price-text {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: #4caf7d;
}

.stat-row {
    display: flex;
    justify-content: space-around;
    text-align: center;
    padding: 1rem 0;
    border-bottom: 1px solid #2e2c28;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Train model ────────────────────────────────────
@st.cache_resource
def train_model():
    df = pd.read_csv('House_Data.csv')
    df.drop(columns=['society', 'availability', 'area_type'], inplace=True)
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['size', 'location'], inplace=True)
    df['bhk'] = df['size'].str.extract(r'(\d+)').astype(int)

    def convert_sqft(val):
        val = str(val).strip()
        if '-' in val:
            try:
                parts = val.split('-')
                return (float(parts[0]) + float(parts[1])) / 2
            except:
                return None
        try:
            return float(val)
        except:
            return None

    df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
    df['bath']    = df['bath'].fillna(df['bath'].median())
    df['balcony'] = df['balcony'].fillna(df['balcony'].median())
    df.dropna(inplace=True)

    df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']
    df = df[(df['price_per_sqft'] > 500) & (df['price_per_sqft'] < 100000)]
    df = df[df['bhk'] <= 10]
    df = df[df['bath'] <= df['bhk'] + 2]
    df = df[df['total_sqft'] >= 200]

    loc_counts = df['location'].value_counts()
    top_locs   = loc_counts[loc_counts >= 10].index
    df['location'] = df['location'].apply(lambda x: x if x in top_locs else 'Other')

    le = LabelEncoder()
    df['location_enc'] = le.fit_transform(df['location'])

    X = df[['total_sqft', 'bath', 'balcony', 'bhk', 'location_enc']]
    y = df['price']

    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    locations = sorted(df['location'].unique().tolist())
    return model, le, locations


model, le, locations = train_model()

# ── UI ─────────────────────────────────────────────
st.markdown('<div class="title">🏠 House Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Bangalore Real Estate · Gradient Boosting · 13,000+ listings</div>', unsafe_allow_html=True)

# Stats row
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Training Data", "13,000+", "rows")
with col2:
    st.metric("Best Model R²", "0.617", "Gradient Boosting")
with col3:
    st.metric("Models Tested", "5", "algorithms")

st.divider()

# Input form
st.subheader("Enter Property Details")

location = st.selectbox("📍 Location", locations)

col1, col2 = st.columns(2)
with col1:
    sqft = st.number_input("📐 Total Sqft", min_value=200, max_value=20000, value=1200, step=50)
    bath = st.selectbox("🚿 Bathrooms", [1, 2, 3, 4, 5], index=1)
with col2:
    bhk  = st.selectbox("🛏️ BHK", [1, 2, 3, 4, 5], index=1)
    balcony = st.selectbox("🌿 Balconies", [0, 1, 2, 3], index=1)

st.divider()

# Predict button
if st.button("🔮 Predict Price", use_container_width=True, type="primary"):
    loc_enc = le.transform([location])[0] if location in le.classes_ else le.transform(['Other'])[0]

    sample = pd.DataFrame([{
        'total_sqft'  : sqft,
        'bath'        : bath,
        'balcony'     : balcony,
        'bhk'         : bhk,
        'location_enc': loc_enc
    }])

    price = model.predict(sample)[0]

    st.success(f"### 💰 Estimated Price: ₹ {price:.2f} Lakhs")
    st.markdown(f"""
    **Summary:**
    - 📍 Location: {location}
    - 📐 Area: {sqft} sqft
    - 🛏️ {bhk} BHK | 🚿 {bath} Bath | 🌿 {balcony} Balcony
    - 💵 Price per sqft: ₹ {(price * 100000 / sqft):,.0f}
    """)

st.divider()
st.caption("Built by Anshu Sharma · B.Tech CSE · Shri Shankaracharya Professional University")
