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

