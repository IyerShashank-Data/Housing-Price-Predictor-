# 🏡 California House Price Prediction using XGBoost

This project tackles the challenge of predicting median house prices across California neighborhoods using real-world data. By combining solid data preprocessing with a powerful XGBoost model, we achieve highly accurate price predictions — while ensuring a clean, leak-proof pipeline and explainable insights.

---

## 🎯 Objective

To build a machine learning model that can reliably predict housing prices based on features like income, location, and household statistics. The goal isn’t just to hit high accuracy — but to follow best practices and treat this as a real-world, end-to-end workflow.

---

## 📊 Dataset Summary

- **Source**: California Housing Dataset (`sklearn.datasets`)
- **Size**: 20,640 rows × 8 features
- **Target**: Median house price (per census block)

### Features include:
- `MedInc`: Median income in the area  
- `HouseAge`: Median age of houses  
- `AveRooms`, `AveBedrms`: Average number of rooms/bedrooms  
- `Population`, `AveOccup`: Local population and occupancy metrics  
- `Latitude`, `Longitude`: Geographic coordinates

---

## ⚙️ What Was Done

✅ Here's how the project was structured:

1. **Log-Transformed the Target**  
   House prices tend to be skewed — so we applied `np.log1p()` to stabilize variance and make modeling easier.

2. **Train-Test Split (Before Scaling)**  
   Data was split early to avoid any leakage during scaling.

3. **Feature Scaling**  
   Used `StandardScaler` on numeric features to fit only on training data.

4. **Modeling with XGBoost**  
   We used:
   XGBRegressor(max_depth=5, learning_rate=0.4, n_estimators=150)

5.**Model Performance**
   
   R2 value: 0.8430133277994334 
   
   Root Mean squared error: 0.13980306039411947 
   
   Relative Error: 6.76 %

## 🔍 What I Learned
Log-transforming the target can drastically improve regression performance.

Even great metrics need validation — leakage can fake perfection.

XGBoost is powerful, but preprocessing and pipeline discipline matter just as much.

Evaluation should always happen in the real-world units — not just log space.
