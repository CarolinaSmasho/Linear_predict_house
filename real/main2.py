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
