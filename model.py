"""
House Price Prediction — Bangalore Housing Dataset
Author : Anshu Sharma
Dataset: 13,000+ real Bangalore house listings
Tech   : Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

os.makedirs('plots', exist_ok=True)
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (10, 6)


# ─────────────────────────────────────────────────────
# 1. LOAD & CLEAN DATA
# ─────────────────────────────────────────────────────
def load_and_clean(path='House_Data.csv'):
    print("[1/5] Loading and cleaning data...")
    df = pd.read_csv(path)
    print(f"      Raw data: {df.shape[0]} rows, {df.shape[1]} columns")

    # Drop low-value columns
    df.drop(columns=['society', 'availability', 'area_type'], inplace=True)

    # Drop duplicates and rows missing key fields
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['size', 'location'], inplace=True)

    # Extract BHK count from size (e.g. "2 BHK" → 2)
    df['bhk'] = df['size'].str.extract(r'(\d+)').astype(int)

    # Fix total_sqft — handle ranges like "1000-1500" → take average
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

    # Fill missing numeric values
    df['bath']    = df['bath'].fillna(df['bath'].median())
    df['balcony'] = df['balcony'].fillna(df['balcony'].median())
    df.dropna(inplace=True)

    # Remove outliers using price per sqft
    df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']
    df = df[(df['price_per_sqft'] > 500) & (df['price_per_sqft'] < 100000)]
    df = df[df['bhk'] <= 10]
    df = df[df['bath'] <= df['bhk'] + 2]
    df = df[df['total_sqft'] >= 200]

    # Group rare locations as 'Other' (keep only locations with 10+ listings)
    loc_counts = df['location'].value_counts()
    top_locs   = loc_counts[loc_counts >= 10].index
    df['location'] = df['location'].apply(lambda x: x if x in top_locs else 'Other')

    print(f"      Clean data: {df.shape[0]} rows")
    return df


# ─────────────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────
def run_eda(df):
    print("[2/5] Running EDA and saving plots...")

    # Price distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df['price'], bins=50, color='steelblue', edgecolor='white')
    axes[0].set(title='Price Distribution (Lakhs ₹)', xlabel='Price (Lakhs)', ylabel='Count')
    axes[1].hist(np.log1p(df['price']), bins=50, color='coral', edgecolor='white')
    axes[1].set(title='Log Price Distribution', xlabel='Log(Price)', ylabel='Count')
    plt.tight_layout()
    plt.savefig('plots/price_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # BHK vs Price
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[df['bhk'] <= 6], x='bhk', y='price', palette='Blues')
    plt.title('Price by BHK', fontsize=14, fontweight='bold')
    plt.xlabel('BHK')
    plt.ylabel('Price (Lakhs ₹)')
    plt.tight_layout()
    plt.savefig('plots/price_by_bhk.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Sqft vs Price
    plt.figure(figsize=(10, 6))
    plt.scatter(df['total_sqft'], df['price'], alpha=0.2, color='steelblue', s=10)
    plt.title('Total Sqft vs Price', fontsize=14, fontweight='bold')
    plt.xlabel('Total Sqft')
    plt.ylabel('Price (Lakhs ₹)')
    plt.tight_layout()
    plt.savefig('plots/sqft_vs_price.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    corr = df[['total_sqft', 'bath', 'balcony', 'bhk', 'price']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                square=True, linewidths=0.5)
    plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Top 10 most expensive locations
    top_locs = df.groupby('location')['price'].median().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    top_locs.plot(kind='barh', color='steelblue')
    plt.title('Top 10 Locations by Median Price', fontsize=14, fontweight='bold')
    plt.xlabel('Median Price (Lakhs ₹)')
    plt.tight_layout()
    plt.savefig('plots/top_locations.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("      Plots saved to /plots/")


# ─────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING & PREPROCESSING
# ─────────────────────────────────────────────────────
def preprocess(df):
    print("[3/5] Preprocessing features...")
    le = LabelEncoder()
    df['location_enc'] = le.fit_transform(df['location'])

    features = ['total_sqft', 'bath', 'balcony', 'bhk', 'location_enc']
    X = df[features]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"      Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, le, scaler, features


# ─────────────────────────────────────────────────────
# 4. TRAIN & EVALUATE MODELS
# ─────────────────────────────────────────────────────
def train_and_evaluate(X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, features):
    print("[4/5] Training models...")

    model_configs = [
        ('Linear Regression',  LinearRegression(),                                           X_train_sc, X_test_sc),
        ('Ridge Regression',   Ridge(alpha=10),                                              X_train_sc, X_test_sc),
        ('Lasso Regression',   Lasso(alpha=0.1),                                             X_train_sc, X_test_sc),
        ('Random Forest',      RandomForestRegressor(n_estimators=100, random_state=42),     X_train,    X_test),
        ('Gradient Boosting',  GradientBoostingRegressor(n_estimators=100, random_state=42), X_train,    X_test),
    ]

    results = []
    for name, model, Xtr, Xte in model_configs:
        model.fit(Xtr, y_train)
        yp   = model.predict(Xte)
        r2   = r2_score(y_test, yp)
        rmse = np.sqrt(mean_squared_error(y_test, yp))
        mae  = mean_absolute_error(y_test, yp)
        cv   = cross_val_score(model, Xtr, y_train, cv=5, scoring='r2').mean()
        results.append({'name': name, 'model': model, 'r2': r2,
                        'rmse': rmse, 'mae': mae, 'cv': cv, 'preds': yp})
        print(f"      {name:<22} R²={r2:.4f}  RMSE=₹{rmse:.2f}L  CV={cv:.4f}")

    # Comparison plots
    res_df = pd.DataFrame([{k: v for k, v in r.items()
                            if k not in ('model', 'preds')} for r in results])
    res_df = res_df.sort_values('r2', ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    c1 = ['#2ecc71' if v == res_df['r2'].max() else '#3498db' for v in res_df['r2']]
    axes[0].barh(res_df['name'], res_df['r2'], color=c1)
    axes[0].set(title='R² Score Comparison', xlabel='R²')
    for i, v in enumerate(res_df['r2']):
        axes[0].text(v + 0.005, i, f'{v:.4f}', va='center')

    c2 = ['#2ecc71' if v == res_df['rmse'].min() else '#e74c3c' for v in res_df['rmse']]
    axes[1].barh(res_df['name'], res_df['rmse'], color=c2)
    axes[1].set(title='RMSE Comparison — lower is better', xlabel='RMSE (Lakhs ₹)')
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Best model plots
    best = max(results, key=lambda x: x['r2'])
    yp   = best['preds']

    plt.figure(figsize=(9, 6))
    plt.scatter(y_test, yp, alpha=0.3, color='steelblue', s=15)
    mn, mx = float(y_test.min()), float(y_test.max())
    plt.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect prediction')
    plt.title(f'Actual vs Predicted — {best["name"]}', fontsize=14, fontweight='bold')
    plt.xlabel('Actual Price (Lakhs ₹)')
    plt.ylabel('Predicted Price (Lakhs ₹)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/actual_vs_predicted.png', dpi=150, bbox_inches='tight')
    plt.close()

    if hasattr(best['model'], 'feature_importances_'):
        imp = pd.Series(best['model'].feature_importances_, index=features).sort_values()
        plt.figure(figsize=(8, 5))
        c = ['#2ecc71' if v == imp.max() else '#3498db' for v in imp]
        imp.plot(kind='barh', color=c)
        plt.title(f'Feature Importance — {best["name"]}', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('plots/feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()

    return best['model'], res_df


# ─────────────────────────────────────────────────────
# 5. PREDICT ON SAMPLE
# ─────────────────────────────────────────────────────
def predict_sample(model, le, scaler):
    print("\n[5/5] Sample prediction...")

    # Example: 2 BHK, 1200 sqft, Electronic City Phase II
    location_name = 'Whitefield'
    loc_enc = le.transform([location_name])[0] if location_name in le.classes_ else le.transform(['Other'])[0]

    sample = pd.DataFrame([{
        'total_sqft'  : 1200,
        'bath'        : 2,
        'balcony'     : 1,
        'bhk'         : 2,
        'location_enc': loc_enc
    }])

    predicted = model.predict(sample)[0]
    print(f"\n{'='*45}")
    print("  SAMPLE HOUSE DETAILS")
    print(f"{'='*45}")
    print(f"  Location   : {location_name}")
    print(f"  Area       : 1200 sqft")
    print(f"  BHK        : 2")
    print(f"  Bathrooms  : 2")
    print(f"  💰 Predicted Price: ₹{predicted:.2f} Lakhs")
    print(f"{'='*45}")


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*50)
    print("  HOUSE PRICE PREDICTION — BANGALORE")
    print("="*50 + "\n")

    df = load_and_clean('House_Data.csv')
    run_eda(df)
    X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, le, scaler, features = preprocess(df)
    best_model, results = train_and_evaluate(X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, features)
    predict_sample(best_model, le, scaler)

    print("\n✅ Done! Check /plots/ for all visualisations.")
