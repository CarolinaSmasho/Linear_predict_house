import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error

# =======================
# 1️⃣ โหลดข้อมูล
# =======================
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

y = np.log1p(train_df['SalePrice'])  # 🔹 Log-transform target
X = train_df.drop(columns=['SalePrice', 'Id'])
test_ids = test_df['Id']
X_test = test_df.drop(columns=['Id'])

# แยกประเภทของคอลัมน์
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# =======================
# 2️⃣ จัดการ missing values
# =======================
for c in numeric_cols:
    X[c] = X[c].fillna(X[c].median())
    X_test[c] = X_test[c].fillna(X[c].median())

for c in categorical_cols:
    fill_val = X[c].mode()[0] if not X[c].mode().empty else "Missing"
    X[c] = X[c].fillna(fill_val)
    X_test[c] = X_test[c].fillna(fill_val)

# =======================
# 3️⃣ One-hot encode
# =======================
X_all = pd.concat([X, X_test])
X_all_encoded = pd.get_dummies(X_all, drop_first=True)

# แยกกลับ train/test
X_encoded = X_all_encoded.iloc[:len(X), :]
X_test_encoded = X_all_encoded.iloc[len(X):, :]

# =======================
# 4️⃣ Scaling numeric features
# =======================
scaler = StandardScaler()
X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])
X_test_encoded[numeric_cols] = scaler.transform(X_test_encoded[numeric_cols])

# =======================
# 5️⃣ สร้างและฝึกโมเดล Linear Regression
# =======================
model = LinearRegression()
model.fit(X_encoded, y)

# =======================
# 6️⃣ ทำนายราคาบ้านใน test.csv
# =======================
y_pred_log = model.predict(X_test_encoded)
y_pred = np.expm1(y_pred_log)  # 🔹 แปลงกลับจาก log

# =======================
# 7️⃣ สร้างไฟล์ submission
# =======================
submission = pd.DataFrame({'Id': test_ids, 'SalePrice': y_pred})
submission.to_csv("my_submission_log.csv", index=False)

print("✅ Saved submission as my_submission_log.csv")
print(f"Predicted range: ${y_pred.min():,.0f} – ${y_pred.max():,.0f}")

# =======================
# ฟังก์ชัน predict_house รุ่นใหม่
# =======================
def predict_house_v2(user_input):
    df_input = pd.DataFrame([user_input])

    # Impute missing numeric columns with median from training data
    for c in numeric_cols:
        if c not in df_input.columns:
            df_input[c] = X[c].median()

    # Impute missing categorical columns with mode from training data
    for c in categorical_cols:
        if c not in df_input.columns:
            df_input[c] = X[c].mode()[0]

    # Scale numeric features using the same scaler
    df_num_scaled = pd.DataFrame(scaler.transform(df_input[numeric_cols]), 
                                 columns=numeric_cols, 
                                 index=df_input.index)

    # One-hot encode categorical features, matching training process (drop_first=True)
    df_cat = pd.get_dummies(df_input[categorical_cols], drop_first=True)

    # Reindex to match X_all_encoded columns, filling missing columns with 0
    df_cat = df_cat.reindex(columns=X_all_encoded.columns, fill_value=0)

    # Combine numeric and categorical features
    df_processed = pd.concat([df_num_scaled, df_cat], axis=1)

    # Ensure df_processed has the exact same columns as X_encoded
    df_processed = df_processed[X_encoded.columns]

    # Predict and return the price in original scale
    return np.expm1(model.predict(df_processed)[0])  # Convert back from log

# =======================
# ตัวอย่างใช้ฟังก์ชันรุ่นใหม่
# =======================
example_input = {
    'MSSubClass': 60,
    'LotFrontage': 80,
    'LotArea': 200,
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

predicted_price = predict_house_v2(example_input)
print(f"\nPredicted house price (v2): ${predicted_price:,.2f}")