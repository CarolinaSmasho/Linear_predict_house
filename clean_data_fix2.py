import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Update file paths as needed
file_path = "/Users/notatord/Documents/Coding/linear-project/test.csv"  # Replace with actual path
out_path = "/Users/notatord/Documents/Coding/linear-project/clean2.csv"  # Replace with actual path

df = pd.read_csv(file_path)

print("=== Original data sample (first 5 rows) ===")
print(df.head())

print("\n=== Info ===")
df.info()

print("\n=== Basic statistics for numeric columns ===")
print(df.describe().T)

# Missing value summary (count and percent)
missing_count = df.isna().sum()
missing_pct = (missing_count / len(df)) * 100
missing_df = pd.DataFrame({"missing_count": missing_count, "missing_pct": missing_pct}).sort_values("missing_pct", ascending=False)
print("\n=== Missing values (top 30) ===")
print(missing_df.head(30))

# Drop columns with >50% missing
threshold = 50.0
cols_to_drop = missing_df[missing_df["missing_pct"] > threshold].index.tolist()
if cols_to_drop:
    print(f"\nDropping columns with >{threshold}% missing ({len(cols_to_drop)} columns): {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)
else:
    print(f"\nNo columns with >{threshold}% missing.")

# Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Drop Id if present from features
if 'Id' in numeric_cols:
    numeric_cols.remove('Id')
    df = df.drop(columns=['Id'])
    print("\nDropped 'Id' column from features.")

print(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols[:20]}{'...' if len(numeric_cols)>20 else ''}")
print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols[:20]}{'...' if len(categorical_cols)>20 else ''}")

# Detect target
target_col = 'SalePrice' if 'SalePrice' in df.columns else None
if target_col:
    print(f"\nDetected target column: {target_col}")

# Split numeric and categorical DF
df_numeric = df[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=df.index)
df_categorical = df[categorical_cols].copy() if categorical_cols else pd.DataFrame(index=df.index)

# Impute numeric with median
for c in df_numeric.columns:
    median_val = df_numeric[c].median()
    if pd.isna(median_val):
        median_val = 0
    df_numeric[c] = df_numeric[c].fillna(median_val)

# Impute categorical with mode
for c in df_categorical.columns:
    mode_val = df_categorical[c].mode(dropna=True)
    if not mode_val.empty:
        fill_val = mode_val[0]
    else:
        fill_val = "Missing"
    df_categorical[c] = df_categorical[c].fillna(fill_val)

# One-hot encode categorical and convert True/False to 1/0
if not df_categorical.empty:
    df_cat_dummies = pd.get_dummies(df_categorical, drop_first=False)
    # Convert boolean columns to 1/0
    df_cat_dummies = df_cat_dummies.astype(int)
else:
    df_cat_dummies = pd.DataFrame(index=df.index)

# Scale numeric features
if not df_numeric.empty:
    scaler = StandardScaler()
    df_numeric_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns, index=df_numeric.index)
else:
    df_numeric_scaled = pd.DataFrame(index=df.index)

# Combine processed features
processed_X = pd.concat([df_numeric_scaled, df_cat_dummies], axis=1)

# Separate target if exists
if target_col:
    y = df[target_col].copy().fillna(df[target_col].median())
    processed = pd.concat([processed_X, y.rename(target_col)], axis=1)
else:
    y = None
    processed = processed_X.copy()

# Save processed dataset
processed.to_csv(out_path, index=False)
print(f"\nProcessed dataset saved to: {out_path} (rows={processed.shape[0]}, cols={processed.shape[1]})")

print("\n=== Processed features sample (first 5 rows) ===")
print(processed.head())

print("\n=== Shapes ===")
print("Original shape:", df.shape)
print("Processed X shape:", processed_X.shape)
if y is not None:
    print("y shape:", y.shape)
else:
    print("No target detected; y = None")

print("\nTop 20 columns in processed features:")
print(pd.DataFrame({"columns": processed_X.columns.tolist()[:200]}))

# Summary dict to return
summary = {
    "original_rows": len(df),
    "processed_columns": processed_X.shape[1],
    "dropped_columns": cols_to_drop,
    "numeric_imputed_with_median": df_numeric.columns.tolist(),
    "categorical_encoded_count": df_cat_dummies.shape[1]
}
print("\nSummary:", summary)