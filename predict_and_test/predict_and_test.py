import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# =======================
# 1️⃣ Load train data
# =======================
try:
    df_train = pd.read_csv("train_1360.csv")
except FileNotFoundError:
    # This block would catch an error if the file was missing
    # but we proceed assuming it's found as in the previous step.
    exit()

y_train = df_train['SalePrice']
X_train = df_train.drop(columns=['SalePrice', 'Id'])

numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# =======================
# 2️⃣ Impute missing values (TRAIN)
# =======================
median_values = {}
mode_values = {}

for c in numeric_cols:
    median_values[c] = X_train[c].median()
    X_train[c] = X_train[c].fillna(median_values[c])
    
for c in categorical_cols:
    mode_val = X_train[c].mode()[0] if not X_train[c].mode().empty else "Missing"
    mode_values[c] = mode_val
    X_train[c] = X_train[c].fillna(mode_values[c])

# =======================
# 3️⃣ One-hot encode categorical (TRAIN)
# =======================
X_train_cat = pd.get_dummies(X_train[categorical_cols], drop_first=False)
X_train_num = X_train[numeric_cols]

# =======================
# 4️⃣ Scale numeric (TRAIN)
# =======================
scaler = StandardScaler()
X_train_num_scaled = pd.DataFrame(scaler.fit_transform(X_train_num), columns=numeric_cols, index=X_train.index)

# =======================
# 5️⃣ Combine features (TRAIN)
# =======================
X_train_processed = pd.concat([X_train_num_scaled, X_train_cat], axis=1)

# =======================
# 6️⃣ Train Linear Regression
# =======================
model = LinearRegression()
model.fit(X_train_processed, y_train)

# ==================================
# 7️⃣ Model Evaluation (TEST DATA)
# ==================================
TEST_FILE = "test_file.csv" 
try:
    # 7.1 Load test data
    df_test = pd.read_csv(TEST_FILE)
    
    # 7.2 Check and separate SalePrice/Id
    if 'SalePrice' not in df_test.columns:
        raise Exception(f"'SalePrice' column not found in {TEST_FILE}. Cannot calculate metrics.") 
    
    # Handle 'Id' column
    if 'Id' not in df_test.columns:
        df_test['Id'] = df_test.index

    y_test = df_test['SalePrice']
    X_test = df_test.drop(columns=['SalePrice', 'Id'], errors='ignore')
    
    # 7.3 Preprocess test data
    
    # Impute missing values (using median/mode from TRAIN)
    for c in numeric_cols:
        X_test[c] = X_test[c].fillna(median_values.get(c, 0))
    for c in categorical_cols:
        X_test[c] = X_test[c].fillna(mode_values.get(c, "Missing"))

    # Scale numeric (using scaler fitted on TRAIN)
    X_test_num = X_test[numeric_cols]
    X_test_num_scaled = pd.DataFrame(scaler.transform(X_test_num), columns=numeric_cols, index=X_test.index)
    
    # One-hot encode categorical (aligning columns with TRAIN)
    X_test_cat = pd.get_dummies(X_test[categorical_cols], drop_first=False)
    X_test_cat = X_test_cat.reindex(columns=X_train_cat.columns, fill_value=0) 
    
    # Combine features
    X_test_processed = pd.concat([X_test_num_scaled, X_test_cat], axis=1)
    
    # 7.4 Predict and Evaluate
    y_pred = model.predict(X_test_processed)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 7.5 Create output CSV file with Id, Prediction Price, Original Price, and Overall MSE
    df_output = pd.DataFrame({
        'Id': df_test['Id'],
        'Prediction Price': y_pred,
        'Original Price': y_test,
        # MSE is an overall metric, so it's constant for all rows in the test set.
        'Overall MSE': mse 
    })

    output_filename = 'prediction_results.csv'
    df_output.to_csv(output_filename, index=False)
    
    print(f"Successfully created file: {output_filename}")
    print(f"Mean Squared Error (MSE): ${mse:,.2f}")
    print(f"R-squared (R²): {r2:.4f}")

except Exception as e:
    print(f"An error occurred: {e}")