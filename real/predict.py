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

# =======================
# ตัวอย่างใช้ฟังก์ชัน
# =======================
example_input = {
    'MSSubClass': 20,
    'MSZoning': 'RL',
    # 'LotFrontage': None,  # 'NA' typically represents missing data
    'LotArea': 10920,
    'Street': 'Pave',
    # 'Alley': None,  # 'NA' for missing data
    'LotShape': 'IR1',
    'LandContour': 'Lvl',
    'Utilities': 'AllPub',
    'LotConfig': 'Corner',
    'LandSlope': 'Gtl',
    'Neighborhood': 'NAmes',
    'Condition1': 'Norm',
    'Condition2': 'Norm',
    'BldgType': '1Fam',
    'HouseStyle': '1Story',
    'OverallQual': 6,
    'OverallCond': 5,
    'YearBuilt': 1960,
    'YearRemodAdd': 1960,
    'RoofStyle': 'Hip',
    'RoofMatl': 'CompShg',
    'Exterior1st': 'MetalSd',
    'Exterior2nd': 'MetalSd',
    'MasVnrType': 'BrkFace',
    'MasVnrArea': 212,
    'ExterQual': 'TA',
    'ExterCond': 'TA',
    'Foundation': 'CBlock',
    'BsmtQual': 'TA',
    'BsmtCond': 'TA',
    'BsmtExposure': 'No',
    'BsmtFinType1': 'BLQ',
    'BsmtFinSF1': 733,
    'BsmtFinType2': 'Unf',
    'BsmtFinSF2': 0,
    'BsmtUnfSF': 520,
    'TotalBsmtSF': 1253,
    'Heating': 'GasA',
    'HeatingQC': 'TA',
    'CentralAir': 'Y',
    'Electrical': 'SBrkr',
    '1stFlrSF': 1253,
    '2ndFlrSF': 0,
    'LowQualFinSF': 0,
    'GrLivArea': 1253,
    'BsmtFullBath': 1,
    'BsmtHalfBath': 0,
    'FullBath': 1,
    'HalfBath': 1,
    'BedroomAbvGr': 2,
    'KitchenAbvGr': 1,
    'KitchenQual': 'TA',
    'TotRmsAbvGrd': 5,
    'Functional': 'Typ',
    'Fireplaces': 1,
    'FireplaceQu': 'Fa',
    'GarageType': 'Attchd',
    'GarageYrBlt': 1960,
    'GarageFinish': 'RFn',
    'GarageCars': 1,
    'GarageArea': 352,
    'GarageQual': 'TA',
    'GarageCond': 'TA',
    'PavedDrive': 'Y',
    'WoodDeckSF': 0,
    'OpenPorchSF': 213,
    'EnclosedPorch': 176,
    '3SsnPorch': 0,
    'ScreenPorch': 0,
    'PoolArea': 0,
    # 'PoolQC': None,  # 'NA' for missing data
    'Fence': 'GdWo',
    # 'MiscFeature': None, #feasibly converted to None,  # 'NA' for missing data
    'MiscVal': 0,
    'MoSold': 5,
    'YrSold': 2008,
    'SaleType': 'WD',
    'SaleCondition': 'Normal'
}

# predicted_price = predict_house(example_input)
# print(f"\nPredicted house price: ${predicted_price:,.2f}")
