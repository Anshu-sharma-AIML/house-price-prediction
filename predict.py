import os
import sys
import pandas as pd
import joblib

# Clear console
os.system('cls' if os.name == 'nt' else 'clear')

print("="*60)
print(" 🏠 BANGALORE HOUSE PRICE PREDICTOR - INTERACTIVE TOOL")
print("="*60)

# Check if model files exist
if not os.path.exists('best_model.joblib') or not os.path.exists('label_encoder.joblib'):
    print("❌ Error: Trained model files not found.")
    print("Please run 'python model.py' first to train the model on the Bangalore dataset.")
    sys.exit(1)

# Load the saved model, encoder, and scaler
try:
    best_model = joblib.load('best_model.joblib')
    le = joblib.load('label_encoder.joblib')
    scaler = joblib.load('scaler.joblib')
except Exception as e:
    print(f"❌ Error loading model files: {e}")
    sys.exit(1)

def get_int_input(prompt, min_val, max_val, default):
    while True:
        try:
            val = input(f"{prompt} (default {default}): ").strip()
            if not val:
                return default
            val = int(val)
            if min_val <= val <= max_val:
                return val
            else:
                print(f"⚠️ Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("⚠️ Invalid input. Please enter a valid integer.")

def get_location_input(prompt, le, default='Whitefield'):
    # Show some examples of valid locations
    popular = ['Whitefield', 'Sarjapur Road', 'Electronic City', 'Kanakpura Road', 'Yelahanka', 'Uttarahalli', 'Hebbal', 'Marathahalli']
    print(f"💡 Popular Locations: {', '.join(popular)}, etc.")
    
    while True:
        val = input(f"{prompt} [default {default}]: ").strip()
        if not val:
            val = default
        
        # Check if the location is in the label encoder classes
        # case-insensitive search
        matched = [c for c in le.classes_ if c.lower() == val.lower()]
        if matched:
            actual_location = matched[0]
            print(f"✅ Location matched: '{actual_location}'")
            return actual_location
        else:
            print(f"⚠️ Location '{val}' is not in the dataset's high-frequency list.")
            confirm = input("Would you like to fall back to 'Other' location? (Yes/No) [default Yes]: ").strip().lower()
            if not confirm or confirm.startswith('y'):
                return 'Other'

# 1. Get user inputs
print("\nPlease enter details of the house in Bangalore:\n")

area = get_int_input("📏 Enter Total Area (sqft) [200 - 50000]", 200, 50000, 1200)
bhk = get_int_input("🛏️ Enter BHK (Bedrooms) [1 - 10]", 1, 10, 2)
bathrooms = get_int_input("🚿 Enter Number of Bathrooms [1 - 10]", 1, 10, bhk)
balcony = get_int_input("🏢 Enter Number of Balconies [0 - 5]", 0, 5, 1)
location = get_location_input("📍 Enter Location", le, 'Whitefield')

# 2. Encode Location
loc_enc = le.transform([location])[0]

# Build sample data frame matching training feature order
sample = pd.DataFrame([{
    'total_sqft': area,
    'bath': bathrooms,
    'balcony': balcony,
    'bhk': bhk,
    'location_enc': loc_enc
}])

# 3. Scale if the model is a linear model
model_name = type(best_model).__name__
is_linear = any(name in model_name for name in ['LinearRegression', 'Ridge', 'Lasso'])

# Predict Price
try:
    if is_linear:
        # Scale features using the fitted scaler
        sample_sc = scaler.transform(sample)
        predicted_lakhs = best_model.predict(sample_sc)[0]
    else:
        # Tree-based models take raw features
        predicted_lakhs = best_model.predict(sample)[0]
    
    # Prices in the Bangalore dataset are in Lakhs of Rupees
    price_in_rupees = predicted_lakhs * 100000
    
    print("\n" + "="*50)
    print(" 🎉 BANGALORE HOUSE PREDICTION RESULTS")
    print("="*50)
    print(f"  📍 Location:   {location}")
    print(f"  📏 Area:       {area} sqft")
    print(f"  🛏️ Type:       {bhk} BHK ({bhk} Bedrooms)")
    print(f"  🚿 Bathrooms:  {bathrooms}")
    print(f"  🏢 Balconies:  {balcony}")
    print("-"*50)
    
    if price_in_rupees >= 10000000: # 1 Crore or more
        crores = price_in_rupees / 10000000
        print(f"  💰 ESTIMATED PRICE: ₹{crores:.2f} Crore (₹{predicted_lakhs:.2f} Lakhs)")
    else:
        print(f"  💰 ESTIMATED PRICE: ₹{predicted_lakhs:.2f} Lakhs (₹{price_in_rupees:,.0f})")
    
    print(f"  🤖 Model Used:      {model_name}")
    print("="*50 + "\n")
except Exception as e:
    print(f"❌ Error during prediction: {e}")
