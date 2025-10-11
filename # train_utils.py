# train_utils.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

# ====== ตั้งค่า ======
PROCESSED_PATH = "/mnt/data/processed_train.csv"  # ปรับตามตำแหน่งไฟล์ของคุณ
TARGET_COL = "SalePrice"
MODEL_SAVE_PATH = "/mnt/data/house_price_model.joblib"

# ====== ฟังก์ชันช่วยเหลือ ======
def load_processed(path=PROCESSED_PATH, target_col=TARGET_COL):
    """โหลดไฟล์ processed CSV และแยก X, y"""
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {path}")
    y = df[target_col].values.reshape(-1, 1)
    X = df.drop(columns=[target_col]).values
    feature_names = df.drop(columns=[target_col]).columns.tolist()
    return X, y.ravel(), feature_names, df

# ====== ฟังก์ชัน train หลัก ======
def train_model(path=PROCESSED_PATH,
                target_col=TARGET_COL,
                test_size=0.2,
                random_state=42,
                use_sklearn=True):
    """
    Train model from processed CSV.
    - use_sklearn=True uses sklearn.LinearRegression
    - returns: dict with model, metrics, X_test, y_test, feature_names, scaler(None)
    """
    # โหลดข้อมูล
    X, y, feature_names, df_full = load_processed(path, target_col)
    
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    
    # ถ้า X เป็นค่าที่ผ่าน scaling แล้ว (ไฟล์ processed ของเราทำ scaling ไว้แล้ว)
    # เราจะถือว่าไม่ต้อง scale ซ้ำ แต่ถ้ยังไม่ scale คุณสามารถเพิ่ม scaler ได้ที่นี่.
    # สำหรับความปลอดภัย: ถ้ค่าเฉลี่ยใด ๆ เป็น NaN ให้เติม 0
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)
    
    # ฝึกโมเดล
    if use_sklearn:
        model = LinearRegression()
        model.fit(X_train, y_train)
    else:
        # Normal equation (ใช้ pseudo-inverse เพื่อความคงตัว)
        # เพิ่ม bias column ถ้ายังไม่มี (สมมติ feature ไม่มี bias)
        # แต่ processed CSV ของเราจะไม่มี bias, ดังนั้นเพิ่มคอลัมน์ 1
        Xtr = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        theta = np.linalg.pinv(Xtr.T @ Xtr) @ Xtr.T @ y_train
        # เก็บ theta ใน object เทียม
        class _Model:
            def __init__(self, theta):
                self.theta = theta
            def predict(self, X):
                Xb = np.hstack([np.ones((X.shape[0], 1)), X])
                return Xb @ self.theta
        model = _Model(theta)
    
    # ประเมินผล
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results = {
        "model": model,
        "metrics": {"mse": mse, "rmse": rmse, "r2": r2},
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "feature_names": feature_names,
        "full_df": df_full
    }
    return results

# ====== ฟังก์ชันบันทึก/โหลดโมเดล ======
def save_model_bundle(model_obj, feature_names, save_path=MODEL_SAVE_PATH):
    """
    Save model + metadata (model object + feature names) using joblib.
    """
    bundle = {
        "model": model_obj,
        "feature_names": feature_names
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(bundle, save_path)
    print(f"Model saved to {save_path}")

def load_model_bundle(save_path=MODEL_SAVE_PATH):
    bundle = joblib.load(save_path)
    return bundle["model"], bundle["feature_names"]

# ====== ฟังก์ชันทำนายและแนะนำบ้านคล้ายกัน ======
def predict_price(model, input_vector):
    """
    input_vector: 1D numpy array มี shape (n_features,)
    returns predicted price (scalar)
    """
    input_vector = np.array(input_vector).reshape(1, -1)
    return model.predict(input_vector).ravel()[0]

def recommend_similar_houses(df_processed, query_vector, feature_names, top_k=5):
    """
    - df_processed: DataFrame ที่มี column ชื่อ feature_names (และอาจมี SalePrice)
    - query_vector: 1D array จำนวนเท่ากับ len(feature_names)
    - คืนค่า top_k rows ของ df_processed ที่มี cosine similarity สูงสุด (ไม่รวม query)
    """
    # เตรียม matrix ของ features (ถ้า df_processed เป็น DataFrame รวม SalePrice ให้ drop)
    X_all = df_processed[feature_names].values
    q = np.array(query_vector).reshape(1, -1)
    # คำนวณ cosine similarity
    sims = cosine_similarity(q, X_all)[0]  # shape (n_samples,)
    df_out = df_processed.copy()
    df_out["_cosine_sim"] = sims
    top = df_out.sort_values("_cosine_sim", ascending=False).head(top_k)
    return top.drop(columns=["_cosine_sim"])

# ====== ตัวอย่างการใช้งาน ======
if __name__ == "__main__":
    # รัน training
    results = train_model()
    print("Metrics:", results["metrics"])
    # บันทึกโมเดล
    save_model_bundle(results["model"], results["feature_names"])
    
    # ตัวอย่าง recommend: ใช้แถวที่ 0 เป็น query ทดลอง
    df_full = results["full_df"]
    feat_names = results["feature_names"]
    query = df_full[feat_names].iloc[0].values
    top5 = recommend_similar_houses(df_full, query, feat_names, top_k=5)
    print("Top 5 similar (sample):")
    print(top5.head())
