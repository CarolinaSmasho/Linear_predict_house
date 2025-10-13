import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# =======================
# 1️⃣ โหลดข้อมูล
# =======================
train_df = pd.read_csv("real/train.csv")
test_df = pd.read_csv("real/test.csv")

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
submission.to_csv("real/my_submission_log.csv", index=False)
print("✅ Saved submission as my_submission_log.csv")
print(f"Predicted range: ${y_pred.min():,.0f} – ${y_pred.max():,.0f}")


# =======================
# 8️⃣ ฟังก์ชันทำนายจากข้อมูลผู้ใช้
# =======================
def predict_from_input(user_input: dict):
    """
    ฟังก์ชันทำนายราคาบ้านจาก input ของผู้ใช้
    user_input: dictionary เช่น
    {
        'OverallQual': 7,
        'GrLivArea': 1800,
        'GarageCars': 2,
        'TotalBsmtSF': 900,
        'FullBath': 2,
        'YearBuilt': 2005,
        'Neighborhood': 'CollgCr'
    }
    """

    # แปลงเป็น DataFrame
    user_df = pd.DataFrame([user_input])

    # เติมค่า missing ให้ครบ
    for c in numeric_cols:
        if c not in user_df.columns:
            user_df[c] = X[c].median()
    for c in categorical_cols:
        if c not in user_df.columns:
            user_df[c] = X[c].mode()[0]

    # One-hot encode และ align columns ให้ตรงกับโมเดล
    user_encoded = pd.get_dummies(user_df, drop_first=True)
    user_encoded = user_encoded.reindex(columns=X_encoded.columns, fill_value=0)

    # Scaling เฉพาะคอลัมน์ตัวเลข
    user_encoded[numeric_cols] = scaler.transform(user_encoded[numeric_cols])

    # ทำนายราคาบ้าน
    user_pred_log = model.predict(user_encoded)
    user_pred = np.expm1(user_pred_log)

    # หาความคล้ายคลึงกับข้อมูลจริง (optional)
    # similarity = cosine_similarity(user_encoded, X_encoded).flatten()
    # most_similar_idx = np.argmax(similarity)
    # similar_price = np.expm1(y.iloc[most_similar_idx])

    print(f"\n🏠 Predicted Price: ${user_pred[0]:,.2f}")
    # print(f"🔹 Most Similar House Price: ${similar_price:,.2f}")
    # print(f"🔹 Similarity Score: {similarity[most_similar_idx]:.4f}")

    return user_pred[0]


# =======================
# 🔹 ตัวอย่างการใช้งาน
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

predict_from_input(example_input)
