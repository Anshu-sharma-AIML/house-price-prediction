# 🏠 House Price Prediction

A regression-based machine learning project that predicts house prices using features like area, location, number of rooms, and more.

Built with **Python, Pandas, NumPy, Scikit-learn, Matplotlib, and Seaborn**.

---

## 📌 Project Overview

This project explores the house price prediction problem end-to-end:
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Training and comparing 5 regression models
- Evaluating with R² Score and RMSE
- Visualising predictions and feature importances

---

## 📂 Project Structure

```
house-price-prediction/
│
├── house_price_prediction.ipynb  # Full notebook (EDA + models + plots)
├── model.py                      # Standalone Python script
├── house_prices.csv              # Dataset (auto-generated on first run)
├── requirements.txt              # Dependencies
├── plots/                        # Output plots
│   ├── price_distribution.png
│   ├── price_by_location.png
│   ├── correlation_heatmap.png
│   ├── model_comparison.png
│   ├── actual_vs_predicted.png
│   └── feature_importance.png
└── README.md
```

---

## 🔧 Features Used

| Feature | Description |
|---------|-------------|
| `Area_sqft` | Total area of the property |
| `Bedrooms` | Number of bedrooms |
| `Bathrooms` | Number of bathrooms |
| `Floors` | Number of floors |
| `Age_years` | Age of the property |
| `Garage` | Garage capacity |
| `Has_Garden` | Garden availability (0/1) |
| `Has_Pool` | Swimming pool (0/1) |
| `Location` | Neighborhood tier (Prime/Good/Average/Outskirts) |
| `Total_rooms` | Engineered: Bedrooms + Bathrooms |
| `Is_new` | Engineered: Age ≤ 5 years |
| `Luxury_score` | Engineered: Garden + Pool + Garage > 1 |

---

## 🤖 Models Trained

| Model | R² Score | Notes |
|-------|----------|-------|
| Linear Regression | ~0.88 | Baseline |
| Ridge Regression | ~0.88 | L2 regularisation |
| Lasso Regression | ~0.87 | L1 regularisation |
| Random Forest | ~0.95 | ⭐ Best Model |
| Gradient Boosting | ~0.94 | Close second |

> **Best Model: Random Forest** with R² ≈ 0.95 and RMSE ≈ ₹1.5L

---

## 📊 Sample Visualisations

- Price distribution (normal + log-transformed)
- Price vs Location (box plots)
- Area vs Price scatter by location
- Correlation heatmap
- Model R² and RMSE comparison
- Actual vs Predicted plot
- Feature Importance chart

---

## ▶️ How to Run

### Option 1 – Jupyter Notebook (Recommended)
```bash
pip install -r requirements.txt
jupyter notebook house_price_prediction.ipynb
```

### Option 2 – Python Script (Model Training)
```bash
pip install -r requirements.txt
python model.py
```

### Option 3 – Interactive Price Predictor
```bash
python predict.py
```

---

## 📈 Evaluation Metrics

- **R² Score** — Proportion of variance explained by the model
- **RMSE** — Root Mean Squared Error (penalises large errors)
- **MAE** — Mean Absolute Error
- **5-Fold Cross-Validation R²** — Checks model generalisation

---

## 🛠️ Tech Stack

- Python 3.10+
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

---

## 👤 Author

**Anshu Sharma**  
B.Tech CSE — Shri Shankaracharya Professional University, Bhilai  
📧 anshusharma6117@gmail.com
