import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Define column names in the specified order
columns = [
    'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 
    'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 
    'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 
    'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 
    'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 
    'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 
    '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 
    'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 
    'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 
    'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 
    'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 
    'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition'
]

# Load training data
train_df = pd.read_csv("train.csv")
y = np.log1p(train_df['SalePrice'])  # Log-transform target
X = train_df.drop(columns=['SalePrice', 'Id'])

# Separate numeric and categorical columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Handle missing values in training data
for c in numeric_cols:
    X[c] = X[c].fillna(X[c].median())
for c in categorical_cols:
    fill_val = X[c].mode()[0] if not X[c].mode().empty else "Missing"
    X[c] = X[c].fillna(fill_val)

# Get input from terminal
input_data = input("Enter house features (space-separated, in order): ").strip().split()
input_df = pd.DataFrame([input_data], columns=columns)

# Handle missing values in input data
for c in numeric_cols:
    input_df[c] = pd.to_numeric(input_df[c], errors='coerce')
    input_df[c] = input_df[c].fillna(X[c].median())
for c in categorical_cols:
    fill_val = X[c].mode()[0] if not X[c].mode().empty else "Missing"
    input_df[c] = input_df[c].fillna(fill_val)

# One-hot encode
X_all = pd.concat([X, input_df])
X_all_encoded = pd.get_dummies(X_all, drop_first=True)

# Split back train and input
X_encoded = X_all_encoded.iloc[:len(X), :]
input_encoded = X_all_encoded.iloc[len(X):, :]

# Scale numeric features
scaler = StandardScaler()
X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])
input_encoded[numeric_cols] = scaler.transform(input_encoded[numeric_cols])

# Train Linear Regression model
model = LinearRegression()
model.fit(X_encoded, y)

# Predict house price
y_pred_log = model.predict(input_encoded)
y_pred = np.expm1(y_pred_log)  # Convert back from log

# Print prediction
print(f"Predicted house price: ${y_pred[0]:,.0f}")