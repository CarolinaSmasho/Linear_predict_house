import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# =======================
# 1️⃣ Load dataset
# =======================
df = pd.read_csv("train_and_test.csv")
y = np.log1p(df['SalePrice'])  # Log-transform target
X = df.drop(columns=['SalePrice', 'Id'])

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# =======================
# 2️⃣ Handle outliers
# =======================
# Remove top 1% of SalePrice and LotArea
q_high = df['SalePrice'].quantile(0.99)
q_high_lot = X['LotArea'].quantile(0.99)
mask = (df['SalePrice'] < q_high) & (X['LotArea'] < q_high_lot)
X = X[mask]
y = y[mask]

# =======================
# 3️⃣ Impute missing values
# =======================
# Impute LotFrontage by Neighborhood median
if 'LotFrontage' in numeric_cols:
    X['LotFrontage'] = X.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
for c in numeric_cols:
    X[c] = X[c].fillna(X[c].median())
for c in categorical_cols:
    X[c] = X[c].fillna(X[c].mode()[0] if not X[c].mode().empty else "Missing")

# Transform skewed numeric features
for c in ['LotArea', 'LotFrontage', 'GrLivArea']:
    if c in numeric_cols:
        X[c] = np.log1p(X[c])

# =======================
# 4️⃣ One-hot encode categorical
# =======================
X_cat = pd.get_dummies(X[categorical_cols], drop_first=True)  # Avoid multicollinearity
X_num = X[numeric_cols]

# =======================
# 5️⃣ Scale numeric
# =======================
scaler = StandardScaler()
X_num_scaled = pd.DataFrame(scaler.fit_transform(X_num), columns=numeric_cols, index=X.index)

# =======================
# 6️⃣ Combine features
# =======================
X_processed = pd.concat([X_num_scaled, X_cat], axis=1)

# =======================
# 7️⃣ Train/test split
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42  # Increased test size
)

# =======================
# 8️⃣ Ridge Regression
# =======================
model = Ridge(alpha=1.0)  # Regularized linear regression
model.fit(X_train, y_train)
y_pred_log = model.predict(X_test)

# Convert predictions back to original scale
y_pred = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test)

# =======================
# 9️⃣ Evaluate
# =======================
mse = mean_squared_error(y_test_orig, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test_orig, y_pred) * 100

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Cross-validation for robust evaluation
cv_scores = cross_val_score(model, X_processed, y, cv=5, scoring='neg_mean_absolute_percentage_error')
cv_mape = -cv_scores.mean() * 100
print(f"5-Fold Cross-Validation MAPE: {cv_mape:.2f}%")

# =======================
# 10️⃣ Predict from user input
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

    # Transform skewed features
    for c in ['LotArea', 'LotFrontage', 'GrLivArea']:
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

    # Predict and convert back to original scale
    price_pred_log = model.predict(df_processed)[0]
    return np.expm1(price_pred_log)

# =======================
# Example usage
# =======================
example_input = {
    'MSSubClass': 60,
    'LotFrontage': 80,
    'LotArea': 100000,
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
    'GarageArea': 500
}

predicted_price = predict_house(example_input)
print(f"\nPredicted house price: ${predicted_price:,.2f}")