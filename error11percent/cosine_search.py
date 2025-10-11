import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# โหลดข้อมูล
df = pd.read_csv("dataold.csv")

# เลือกเฉพาะคอลัมน์ที่ใช้
data = df[["Neighborhood", "LotArea", "SalePrice"]].copy()

# แปลงหมวดหมู่เป็นตัวเลข
le = LabelEncoder()
data["Neighborhood"] = le.fit_transform(data["Neighborhood"])

inp = input("Neighborhood, LotArea, SalePrice: ").split(" ")
# สร้าง input
input_data = {
    "Neighborhood": inp[0],
    "LotArea": int(inp[1]),
    "SalePrice": int(inp[2])
}

# แปลง input
input_vector = [
    le.transform([input_data["Neighborhood"]])[0],
    input_data["LotArea"],
    input_data["SalePrice"]
]

# คำนวณ cosine similarity
similarities = cosine_similarity(
    [input_vector],
    data[["Neighborhood", "LotArea", "SalePrice"]]
)[0]

# เพิ่มคอลัมน์ similarity แล้วเรียงจากมาก→น้อย
data["similarity"] = similarities
similar_houses = data.sort_values(by="similarity", ascending=False).head(10)

print(similar_houses)


# Input Example
# CollgCr 8450 208500