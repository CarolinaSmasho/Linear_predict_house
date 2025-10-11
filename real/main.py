import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_squared_error

# =======================
# 1️⃣ โหลด train data
# =======================
df_train = pd.read_csv("train.csv")
y_train = df_train['SalePrice']
X_train = df_train.drop(columns=['SalePrice', 'Id'])

numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# =======================
# 2️⃣ Impute missing values
# =======================
for c in numeric_cols:
    X_train[c] = X_train[c].fillna(X_train[c].median())
for c in categorical_cols:
    mode_val = X_train[c].mode()[0] if not X_train[c].mode().empty else "Missing"
    X_train[c] = X_train[c].fillna(mode_val)

# =======================
# 3️⃣ One-hot encode categorical
# =======================
X_train_cat = pd.get_dummies(X_train[categorical_cols], drop_first=False)
X_train_num = X_train[numeric_cols]

# =======================
# 4️⃣ Scale numeric
# =======================
scaler = StandardScaler()
X_train_num_scaled = pd.DataFrame(scaler.fit_transform(X_train_num), columns=numeric_cols, index=X_train.index)

# =======================
# 5️⃣ Combine features
# =======================
X_train_processed = pd.concat([X_train_num_scaled, X_train_cat], axis=1)

# =======================
# 6️⃣ Train Linear Regression
# =======================
model = LinearRegression()
model.fit(X_train_processed, y_train)

# =======================
# 7️⃣ โหลด test.csv และ sample_submission.csv
# =======================
df_test = pd.read_csv("test.csv")  # test set ไม่มี SalePrice
sample_submission = pd.read_csv("sample_submission.csv")  # template submission

X_test = df_test.drop(columns=['Id'])

# =======================
# 8️⃣ Impute missing values (ใช้ median/mode จาก train)
# =======================
for c in numeric_cols:
    if c in X_test.columns:
        X_test[c] = X_test[c].fillna(X_train[c].median())
for c in categorical_cols:
    if c in X_test.columns:
        mode_val = X_train[c].mode()[0] if not X_train[c].mode().empty else "Missing"
        X_test[c] = X_test[c].fillna(mode_val)

# =======================
# 9️⃣ One-hot encode test
# =======================
X_test_cat = pd.get_dummies(X_test[categorical_cols], drop_first=False)
X_test_cat = X_test_cat.reindex(columns=X_train_cat.columns, fill_value=0)
X_test_num_scaled = pd.DataFrame(scaler.transform(X_test[numeric_cols]), columns=numeric_cols, index=X_test.index)
X_test_processed = pd.concat([X_test_num_scaled, X_test_cat], axis=1)

# =======================
# 10️⃣ Predict test set
# =======================
y_test_pred = model.predict(X_test_processed)

# =======================
# 11️⃣ สร้าง submission.csv
# =======================
submission = pd.DataFrame({
    "Id": sample_submission["Id"],
    "SalePrice": y_test_pred
})
submission.to_csv("submission.csv", index=False)
print("submission.csv ถูกสร้างเรียบร้อย ✅")

# =======================
# 12️⃣ Predict จาก user input
# =======================
def predict_house(user_input):
    df_input = pd.DataFrame([user_input])

    # Impute
    for c in numeric_cols:
        if c not in df_input.columns:
            df_input[c] = X_train[c].median()
    for c in categorical_cols:
        if c not in df_input.columns:
            df_input[c] = X_train[c].mode()[0]

    # Scale numeric
    df_num_scaled = pd.DataFrame(scaler.transform(df_input[numeric_cols]), columns=numeric_cols)

    # One-hot encode categorical
    df_cat = pd.get_dummies(df_input[categorical_cols], drop_first=False)
    df_cat = df_cat.reindex(columns=X_train_cat.columns, fill_value=0)

    df_processed = pd.concat([df_num_scaled, df_cat], axis=1)

    return model.predict(df_processed)[0]

# =======================
# ตัวอย่างใช้ฟังก์ชัน
# =======================
example_input = {
    'MSSubClass': 60,
    'LotFrontage': 80,
    'LotArea': 20000,
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
