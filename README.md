## 🚀 Live Demo
👉 [Click here to try the app](https://house-price-prediction-zcckobfbrbyywxwpv9ow9j.streamlit.app/)

---



# 🏠 House Price Prediction — Bangalore

A machine learning project that predicts house prices in Bangalore using a real dataset of **13,000+ property listings**.

**Tech Stack:** Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn

---

## 📌 Project Overview

This project builds and compares multiple regression models to predict property prices in Bangalore, India. It covers the full ML pipeline — data cleaning, EDA, feature engineering, model training, and evaluation using **R² Score** and **RMSE**.

---

## 📂 Project Structure

```
house-price-prediction/
│
├── House_Data.csv                # Real Bangalore housing dataset (13,000+ rows)
├── model.py                      # Full ML pipeline script
├── house_price_prediction.ipynb  # Jupyter notebook with detailed analysis
├── requirements.txt              # Python dependencies
├── plots/                        # Generated visualisation plots
│   ├── price_distribution.png
│   ├── price_by_bhk.png
│   ├── sqft_vs_price.png
│   ├── correlation_heatmap.png
│   ├── top_locations.png
│   ├── model_comparison.png
│   ├── actual_vs_predicted.png
│   └── feature_importance.png
└── README.md
```

---

## 📊 Dataset

| Property | Detail |
|----------|--------|
| Source | Bangalore real estate listings |
| Rows | 13,320 (raw) → 11,903 (after cleaning) |
| Target | `price` (in Lakhs ₹) |
| Key Features | Location, BHK, Total Sqft, Bathrooms, Balcony |

---

## 🔧 Data Cleaning Steps

- Dropped irrelevant columns (`society`, `availability`, `area_type`)
- Removed 529 duplicate rows
- Extracted BHK count from text field (`"2 BHK"` → `2`)
- Converted range values in `total_sqft` (`"1000-1500"` → `1250.0`)
- Filled missing `bath` and `balcony` with median values
- Removed outliers using price-per-sqft thresholds
- Grouped rare locations (< 10 listings) as `"Other"`

---

## 🤖 Models Trained & Results

| Model | R² Score | RMSE (Lakhs) | CV R² |
|-------|----------|--------------|-------|
| **Gradient Boosting** | **0.617** | **77.8** | **0.583** |
| Random Forest | 0.585 | 81.0 | 0.573 |
| Ridge Regression | 0.487 | 90.1 | 0.472 |
| Lasso Regression | 0.487 | 90.1 | 0.472 |
| Linear Regression | 0.487 | 90.1 | 0.472 |

> **Best Model: Gradient Boosting** — R² = 0.617, RMSE = ₹77.8 Lakhs
>
> Note: Real-world housing data has inherent noise from unobserved factors (interior condition, negotiation, etc.) which limits maximum achievable R².

---

## 📈 Key Findings

- **Location** is the most important predictor of house price
- **Total sqft** and **BHK** are the next strongest features
- Gradient Boosting significantly outperforms linear models
- Top locations by median price: Race Course Road, Sankey Road, Lavelle Road

---

## ▶️ How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python model.py

# Or open the notebook
jupyter notebook house_price_prediction.ipynb
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Pandas** — data manipulation and cleaning
- **NumPy** — numerical operations
- **Scikit-learn** — ML models, preprocessing, evaluation
- **Matplotlib & Seaborn** — data visualisation

---

## 👤 Author

**Anshu Sharma**  
B.Tech CSE — Shri Shankaracharya Professional University, Bhilai  
📧 anshusharma6117@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/AnshuSharma)
