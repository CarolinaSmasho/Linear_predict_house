import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

# =======================
# 1️⃣ โหลด dataset
# =======================
df = pd.read_csv("train.csv")
y = df['SalePrice']
X = df.drop(columns=['SalePrice', 'Id'])

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# =======================
# 2️⃣ Impute missing values
# =======================
for c in numeric_cols:
    X[c] = X[c].fillna(X[c].median())
for c in categorical_cols:
    X[c] = X[c].fillna(X[c].mode()[0] if not X[c].mode().empty else "Missing")

# =======================
# 3️⃣ One-hot encode categorical
# =======================
X_cat = pd.get_dummies(X[categorical_cols], drop_first=False)
X_num = X[numeric_cols]

# =======================
# 4️⃣ Scale numeric
# =======================
scaler = StandardScaler()
X_num_scaled = pd.DataFrame(scaler.fit_transform(X_num), columns=numeric_cols, index=X.index)

# =======================
# 5️⃣ Combine features
# =======================
X_processed = pd.concat([X_num_scaled, X_cat], axis=1)

# =======================
# 6️⃣ Train/test split
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.3, random_state=42
)

# =======================
# 7️⃣ Linear Regression
# =======================
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# =======================
# 8️⃣ Evaluate
# =======================
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
percent_error = 100 * np.abs((y_test.values - y_pred) / y_test.values)
mape = percent_error.mean()

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# =======================
# 9️⃣ Predict from user input
# =======================
def predict_house(user_input):
    """
    user_input: dict ของ feature เช่น
    {
        'MSSubClass': 60,
        'LotFrontage': 80,
        'LotArea': 9600,
        'OverallQual': 7,
        ...
        'Neighborhood': 'CollgCr',
        ...
    }
    """
    # แปลงเป็น DataFrame
    df_input = pd.DataFrame([user_input])

    # Impute numeric/categorical
    for c in numeric_cols:
        if c not in df_input.columns:
            df_input[c] = X[c].median()
    for c in categorical_cols:
        if c not in df_input.columns:
            df_input[c] = X[c].mode()[0]

    # Scale numeric
    df_num = df_input[numeric_cols]
    df_num_scaled = pd.DataFrame(scaler.transform(df_num), columns=numeric_cols)

    # One-hot encode categorical
    df_cat = pd.get_dummies(df_input[categorical_cols], drop_first=False)

    # Align columns with training set
    df_cat = df_cat.reindex(columns=X_cat.columns, fill_value=0)

    # Combine
    df_processed = pd.concat([df_num_scaled, df_cat], axis=1)

    # Predict
    price_pred = model.predict(df_processed)[0]
    return price_pred

# =======================
# ตัวอย่างใช้ฟังก์ชัน
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
