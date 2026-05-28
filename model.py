"""
House Price Prediction
Author: Anshu Sharma
Tech Stack: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
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


# ─────────────────────────────────────────
# 1. Generate Dataset
# ─────────────────────────────────────────
def generate_dataset(n=1000, seed=42):
    np.random.seed(seed)
    locations = ['Prime', 'Good', 'Average', 'Outskirts']
    loc_mult  = {'Prime': 1.6, 'Good': 1.2, 'Average': 1.0, 'Outskirts': 0.75}

    area      = np.random.randint(500, 5000, n)
    bedrooms  = np.random.randint(1, 6, n)
    bathrooms = np.clip(bedrooms - np.random.randint(0, 2, n), 1, 5)
    floors    = np.random.randint(1, 4, n)
    age       = np.random.randint(0, 40, n)
    garage    = np.random.randint(0, 3, n)
    location  = np.random.choice(locations, n, p=[0.15, 0.30, 0.35, 0.20])
    garden    = np.random.randint(0, 2, n)
    pool      = np.random.randint(0, 2, n)

    mult  = np.array([loc_mult[l] for l in location])
    price = (
        (area * 45 + bedrooms * 120000 + bathrooms * 80000
         + floors * 50000 - age * 15000 + garage * 90000
         + garden * 70000 + pool * 150000) * mult
        + np.random.normal(0, 150000, n)
    ).clip(300000, None).astype(int)

    df = pd.DataFrame({
        'Area_sqft': area, 'Bedrooms': bedrooms, 'Bathrooms': bathrooms,
        'Floors': floors, 'Age_years': age, 'Garage': garage,
        'Has_Garden': garden, 'Has_Pool': pool,
        'Location': location, 'Price': price
    })
    df.to_csv('house_prices.csv', index=False)
    print(f"[DATA] Dataset created — {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────
# 2. EDA Plots
# ─────────────────────────────────────────
def run_eda(df):
    print("\n[EDA] Running exploratory data analysis...")

    # Price distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df['Price'], bins=40, color='steelblue', edgecolor='white')
    axes[0].set(title='Price Distribution', xlabel='Price (₹)', ylabel='Count')
    axes[1].hist(np.log1p(df['Price']), bins=40, color='coral', edgecolor='white')
    axes[1].set(title='Log Price Distribution', xlabel='Log(Price)', ylabel='Count')
    plt.tight_layout()
    plt.savefig('plots/price_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Price by location
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Location', y='Price',
                order=['Prime', 'Good', 'Average', 'Outskirts'],
                palette=['#e74c3c', '#e67e22', '#3498db', '#95a5a6'])
    plt.title('Price by Location')
    plt.tight_layout()
    plt.savefig('plots/price_by_location.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df.select_dtypes(include=np.number).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                linewidths=0.5, square=True)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('plots/correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("[EDA] Plots saved to /plots/")


# ─────────────────────────────────────────
# 3. Preprocessing & Feature Engineering
# ─────────────────────────────────────────
def preprocess(df):
    df_m = df.copy()
    le = LabelEncoder()
    df_m['Location_enc'] = le.fit_transform(df_m['Location'])
    df_m['Total_rooms']  = df_m['Bedrooms'] + df_m['Bathrooms']
    df_m['Is_new']       = (df_m['Age_years'] <= 5).astype(int)
    df_m['Luxury_score'] = df_m['Has_Garden'] + df_m['Has_Pool'] + (df_m['Garage'] > 1).astype(int)

    features = ['Area_sqft', 'Bedrooms', 'Bathrooms', 'Floors', 'Age_years',
                'Garage', 'Has_Garden', 'Has_Pool', 'Location_enc',
                'Total_rooms', 'Is_new', 'Luxury_score']

    X = df_m[features]
    y = df_m['Price']
    return X, y, le, features


# ─────────────────────────────────────────
# 4. Train & Evaluate Models
# ─────────────────────────────────────────
def evaluate(name, model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    yp   = model.predict(Xte)
    rmse = np.sqrt(mean_squared_error(yte, yp))
    mae  = mean_absolute_error(yte, yp)
    r2   = r2_score(yte, yp)
    cv   = cross_val_score(model, Xtr, ytr, cv=5, scoring='r2').mean()
    print(f"\n{'─'*40}\n{name}")
    print(f"  R²   : {r2:.4f}  |  CV R²: {cv:.4f}")
    print(f"  RMSE : ₹{rmse:>12,.0f}")
    print(f"  MAE  : ₹{mae:>12,.0f}")
    return {'Model': name, 'R2': round(r2, 4), 'RMSE': round(rmse, 0),
            'MAE': round(mae, 0), 'CV_R2': round(cv, 4), 'preds': yp, 'obj': model}


def train_models(X_train, y_train, X_test, y_test, X_train_sc, X_test_sc):
    print("\n[MODELS] Training and evaluating models...")
    results = [
        evaluate('Linear Regression',   LinearRegression(),                                X_train_sc, y_train, X_test_sc, y_test),
        evaluate('Ridge Regression',    Ridge(alpha=10),                                   X_train_sc, y_train, X_test_sc, y_test),
        evaluate('Lasso Regression',    Lasso(alpha=1000),                                 X_train_sc, y_train, X_test_sc, y_test),
        evaluate('Random Forest',       RandomForestRegressor(n_estimators=100, random_state=42), X_train, y_train, X_test, y_test),
        evaluate('Gradient Boosting',   GradientBoostingRegressor(n_estimators=100, random_state=42), X_train, y_train, X_test, y_test),
    ]
    return results


# ─────────────────────────────────────────
# 5. Visualise Results
# ─────────────────────────────────────────
def plot_results(results, y_test, features):
    res_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ('preds', 'obj')} for r in results])
    res_df = res_df.sort_values('R2', ascending=False).reset_index(drop=True)
    print("\n[RESULTS] Model Comparison:")
    print(res_df.to_string(index=False))

    # Comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    c1 = ['#2ecc71' if v == res_df['R2'].max() else '#3498db' for v in res_df['R2']]
    axes[0].barh(res_df['Model'], res_df['R2'], color=c1)
    axes[0].set(title='R² Score Comparison', xlabel='R²')
    for i, v in enumerate(res_df['R2']):
        axes[0].text(v + 0.005, i, f'{v:.4f}', va='center')

    c2 = ['#2ecc71' if v == res_df['RMSE'].min() else '#e74c3c' for v in res_df['RMSE']]
    axes[1].barh(res_df['Model'], res_df['RMSE'], color=c2)
    axes[1].set(title='RMSE Comparison (lower = better)', xlabel='RMSE (₹)')
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Best model diagnostics
    best = max(results, key=lambda r: r['R2'])
    y_pred = best['preds']

    plt.figure(figsize=(9, 6))
    plt.scatter(y_test, y_pred, alpha=0.4, color='steelblue', s=20)
    mn, mx = y_test.min(), y_test.max()
    plt.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect')
    plt.title(f'Actual vs Predicted — {best["Model"]}')
    plt.xlabel('Actual (₹)')
    plt.ylabel('Predicted (₹)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/actual_vs_predicted.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Feature importance (if tree-based)
    model_obj = best['obj']
    if hasattr(model_obj, 'feature_importances_'):
        imp = pd.Series(model_obj.feature_importances_, index=features).sort_values()
        plt.figure(figsize=(9, 6))
        c = ['#2ecc71' if v == imp.max() else '#3498db' for v in imp]
        imp.plot(kind='barh', color=c)
        plt.title(f'Feature Importance — {best["Model"]}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('plots/feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()

    print("[RESULTS] All plots saved to /plots/")
    return best['obj']


# ─────────────────────────────────────────
# 6. Predict on New Sample
# ─────────────────────────────────────────
def predict_sample(model, le):
    sample = pd.DataFrame([{
        'Area_sqft': 1800, 'Bedrooms': 3, 'Bathrooms': 2, 'Floors': 2,
        'Age_years': 5, 'Garage': 1, 'Has_Garden': 1, 'Has_Pool': 0,
        'Location_enc': le.transform(['Good'])[0],
        'Total_rooms': 5, 'Is_new': 1, 'Luxury_score': 1
    }])
    price = model.predict(sample)[0]
    print(f"\n{'='*40}")
    print("  SAMPLE PREDICTION")
    print(f"{'='*40}")
    print("  Area: 1800 sqft | 3 BHK | Location: Good")
    print(f"  💰 Predicted Price: ₹{price:,.0f}")
    print(f"{'='*40}")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == '__main__':
    df        = generate_dataset()
    run_eda(df)

    X, y, le, features = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    results    = train_models(X_train, y_train, X_test, y_test, X_train_sc, X_test_sc)
    best_model = plot_results(results, y_test, features)

    # Save the best model and label encoder for interactive predictions
    import joblib
    joblib.dump(best_model, 'best_model.joblib')
    joblib.dump(le, 'label_encoder.joblib')
    print("[SAVED] Best model ('best_model.joblib') and LabelEncoder ('label_encoder.joblib') saved.")

    predict_sample(best_model, le)

    print("\n✅ Project complete! Check /plots/ for all visualisations.")
