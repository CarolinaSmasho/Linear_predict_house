import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, make_scorer

# Custom MAPE scorer for original price scale
def custom_mape(y_true, y_pred):
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)
    return mean_absolute_percentage_error(y_true_orig, y_pred_orig) * 100

# Wrap custom_mape for cross-validation
mape_scorer = make_scorer(custom_mape, greater_is_better=False)

# =======================
# 1️⃣ Load dataset
# =======================
df = pd.read_csv("dataold.csv")
y = np.log1p(df['SalePrice'])  # Log-transform target
X = df.drop(columns=['SalePrice', 'Id'])

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# =======================
# 2️⃣ Handle outliers
# =======================
q_high = df['SalePrice'].quantile(0.99)
q_high_lot = X['LotArea'].quantile(0.99)
q_high_grliv = X['GrLivArea'].quantile(0.99) if 'GrLivArea' in X.columns else np.inf
mask = (df['SalePrice'] < q_high) & (X['LotArea'] < q_high_lot) & (X['GrLivArea'] < q_high_grliv)
X = X[mask]
y = y[mask]

# Cap LotArea and GrLivArea for prediction
lot_area_cap = X['LotArea'].quantile(0.99)
grliv_area_cap = X['GrLivArea'].quantile(0.99) if 'GrLivArea' in X.columns else np.inf

# =======================
# 3️⃣ Feature engineering
# =======================
if 'GrLivArea' in X.columns and 'OverallQual' in X.columns:
    X['Qual_GrLivArea'] = X['OverallQual'] * X['GrLivArea']
    numeric_cols.append('Qual_GrLivArea')

# =======================
# 4️⃣ Impute missing values
# =======================
if 'LotFrontage' in numeric_cols:
    X['LotFrontage'] = X.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
for c in numeric_cols:
    X[c] = X[c].fillna(X[c].median())
for c in categorical_cols:
    X[c] = X[c].fillna(X[c].mode()[0] if not X[c].mode().empty else "Missing")

# Transform skewed numeric features
for c in ['LotArea', 'LotFrontage', 'GrLivArea', 'TotalBsmtSF']:
    if c in X.columns:
        X[c] = np.log1p(X[c])

# =======================
# 5️⃣ One-hot encode categorical
# =======================
X_cat = pd.get_dummies(X[categorical_cols], drop_first=True)
X_num = X[numeric_cols]

# =======================
# 6️⃣ Scale numeric
# =======================
scaler = StandardScaler()
X_num_scaled = pd.DataFrame(scaler.fit_transform(X_num), columns=numeric_cols, index=X.index)

# =======================
# 7️⃣ Combine features
# =======================
X_processed = pd.concat([X_num_scaled, X_cat], axis=1)

# =======================
# 8️⃣ Train/test split
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# =======================
# 9️⃣ Ridge Regression with tuning
# =======================
param_grid = {'alpha': [0.1, 1, 10, 50, 100]}  # Narrower range
grid = GridSearchCV(Ridge(), param_grid, cv=5, scoring=mape_scorer)
grid.fit(X_train, y_train)
model = grid.best_estimator_
print(f"Best alpha: {grid.best_params_['alpha']}")

y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test)

# Clip predictions to avoid unrealistic values
y_pred = np.maximum(y_pred, 50000)  # Minimum house price

# =======================
# 10️⃣ Evaluate
# =======================
mse = mean_squared_error(y_test_orig, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test_orig, y_pred) * 100

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Cross-validation with fixed scorer
cv_scores = cross_val_score(model, X_processed, y, cv=5, scoring=mape_scorer)
cv_mape = -cv_scores.mean()  # Negate for positive MAPE
print(f"5-Fold Cross-Validation MAPE: {cv_mape:.2f}%")

# =======================
# 11️⃣ Predict from user input
# =======================
def predict_house(user_input):
    df_input = pd.DataFrame([user_input])

    # Impute missing values
    for c in numeric_cols:
        if c not in df_input.columns:
            df_input[c] = X[c].median()
    for c in categorical_cols:
        if c not in df_input.columns:
            df_input[c] = X[c].mode()[0]

    # Cap LotArea and GrLivArea
    df_input['LotArea'] = np.minimum(df_input['LotArea'], lot_area_cap)
    if 'GrLivArea' in df_input.columns:
        df_input['GrLivArea'] = np.minimum(df_input['GrLivArea'], grliv_area_cap)

    # Add interaction term
    if 'GrLivArea' in df_input.columns and 'OverallQual' in df_input.columns:
        df_input['Qual_GrLivArea'] = df_input['OverallQual'] * df_input['GrLivArea']

    # Transform skewed features
    for c in ['LotArea', 'LotFrontage', 'GrLivArea', 'TotalBsmtSF']:
        if c in df_input.columns:
            df_input[c] = np.log1p(df_input[c])

    # Scale numeric
    df_num = df_input[numeric_cols]
    df_num_scaled = pd.DataFrame(scaler.transform(df_num), columns=numeric_cols)

    # One-hot encode categorical
    df_cat = pd.get_dummies(df_input[categorical_cols], drop_first=True)
    df_cat = df_cat.reindex(columns=X_cat.columns, fill_value=0)

    # Combine
    df_processed = pd.concat([df_num_scaled, df_cat], axis=1)

    # Predict and convert back
    price_pred_log = model.predict(df_processed)[0]
    print(f"Log-transformed prediction: {price_pred_log:.2f}")
    price_pred = np.expm1(price_pred_log)
    price_pred = np.maximum(price_pred, 50000)  # Clip low predictions
    return price_pred

# =======================
# Example usage
# =======================
example_input = {
    'MSSubClass': 60,
    'LotFrontage': 80,
    'LotArea': 100000,  # Will be capped
    'OverallQual': 7,
    'OverallCond': 5,
    'YearBuilt': 2000,
    'YearRemodAdd': 2003,
    'Neighborhood': 'CollgCr',
    'BldgType': '1Fam',
    'HouseStyle': '2Story',
    'Exterior1st': 'VinylSd',
    'Exterior2nd': 'VinylSd',
    'ExterQual': 'Gd',
    'ExterCond': 'TA',
    'Foundation': 'PConc',
    'BsmtQual': 'Gd',
    'BsmtCond': 'TA',
    'Heating': 'GasA',
    'HeatingQC': 'Ex',
    'CentralAir': 'Y',
    'KitchenQual': 'Gd',
    'Functional': 'Typ',
    'Fireplaces': 1,
    'GarageCars': 2,
    'GarageArea': 500,
    'GrLivArea': 2000,  # Added
    'TotalBsmtSF': 1000  # Added
}

predicted_price = predict_house(example_input)
print(f"\nPredicted house price: ${predicted_price:,.2f}")

# =======================
# Debug: Check SalePrice distribution
# =======================
print("\nSalePrice Distribution:")
print(df['SalePrice'].describe())