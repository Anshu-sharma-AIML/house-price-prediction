import os
import sys
import pandas as pd
import joblib

# Clear console
os.system('cls' if os.name == 'nt' else 'clear')

print("="*60)
print(" 🏠 HOUSE PRICE PREDICTOR - INTERACTIVE TOOL")
print("="*60)

# Check if model files exist
if not os.path.exists('best_model.joblib') or not os.path.exists('label_encoder.joblib'):
    print("❌ Error: Trained model files not found.")
    print("Please run 'python model.py' first to train the model and save it.")
    sys.exit(1)

# Load the saved model and encoder
try:
    best_model = joblib.load('best_model.joblib')
    le = joblib.load('label_encoder.joblib')
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

def get_choice_input(prompt, choices, default):
    choices_str = "/".join(choices)
    while True:
        val = input(f"{prompt} ({choices_str}) [default {default}]: ").strip()
        if not val:
            return default
        # Match case insensitively
        matched = [c for c in choices if c.lower() == val.lower()]
        if matched:
            return matched[0]
        else:
            print(f"⚠️ Invalid choice. Please select from {choices}.")

# 1. Get user inputs
print("\nPlease enter details of the house you want to estimate:\n")

area = get_int_input("📏 Enter Area (sqft) [500 - 10000]", 500, 10000, 1500)
bedrooms = get_int_input("🛏️ Enter Number of Bedrooms (1 - 10)", 1, 10, 3)
bathrooms = get_int_input("🚿 Enter Number of Bathrooms (1 - 10)", 1, 10, 2)
floors = get_int_input("🏢 Enter Number of Floors (1 - 5)", 1, 5, 1)
age = get_int_input("⏳ Enter Age of House (years) [0 - 100]", 0, 100, 5)
garage = get_int_input("🚗 Enter Garage Capacity (0 - 5)", 0, 5, 1)

garden_yn = get_choice_input("🏡 Does it have a Garden?", ["Yes", "No"], "Yes")
garden = 1 if garden_yn == "Yes" else 0

pool_yn = get_choice_input("🏊 Does it have a Pool?", ["Yes", "No"], "No")
pool = 1 if pool_yn == "Yes" else 0

location = get_choice_input("📍 Select Location Tier", ["Prime", "Good", "Average", "Outskirts"], "Good")

# 2. Feature Engineering
location_enc = le.transform([location])[0]
total_rooms = bedrooms + bathrooms
is_new = 1 if age <= 5 else 0
luxury_score = garden + pool + (1 if garage > 1 else 0)

# Build sample data frame matching training feature order
sample = pd.DataFrame([{
    'Area_sqft': area,
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'Floors': floors,
    'Age_years': age,
    'Garage': garage,
    'Has_Garden': garden,
    'Has_Pool': pool,
    'Location_enc': location_enc,
    'Total_rooms': total_rooms,
    'Is_new': is_new,
    'Luxury_score': luxury_score
}])

# 3. Predict Price
try:
    predicted_price = best_model.predict(sample)[0]
    
    print("\n" + "="*50)
    print(" 🎉 PREDICTION RESULTS")
    print("="*50)
    print(f"  📏 Area:          {area} sqft")
    print(f"  🛏️ Rooms:         {bedrooms} BHK ({bedrooms} Bed, {bathrooms} Bath)")
    print(f"  🏢 Floors:        {floors}")
    print(f"  ⏳ Age:           {age} years old ({'New property' if is_new else 'Resale property'})")
    print(f"  🚗 Garage:        {garage} spaces")
    print(f"  🏡 Has Garden:    {garden_yn}")
    print(f"  🏊 Has Pool:      {pool_yn}")
    print(f"  📍 Location:      {location}")
    print("-"*50)
    print(f"  💰 ESTIMATED PRICE: ₹{predicted_price:,.2f}")
    print("="*50 + "\n")
except Exception as e:
    print(f"❌ Error during prediction: {e}")
