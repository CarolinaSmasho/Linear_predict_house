import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# โหลดข้อมูล
df = pd.read_csv("dataold.csv")
data = df[["Neighborhood", "LotArea", "SalePrice"]].copy()

# รับ input
inp = input("Neighborhood, LotArea, SalePrice: ").split(" ")
input_data = {
    "Neighborhood": inp[0],
    "LotArea": int(inp[1]),
    "SalePrice": int(inp[2])
}

# สร้างคอลัมน์ Neighborhood_match
data["Neighborhood_match"] = data["Neighborhood"].apply(
    lambda x: 1 if x == input_data["Neighborhood"] else 0
)

# รวมข้อมูลสำหรับ scaling
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data[["Neighborhood_match", "LotArea", "SalePrice"]])

# แปลง input_vector แล้ว scale ด้วย scaler เดียวกัน
input_vector = scaler.transform([[1, input_data["LotArea"], input_data["SalePrice"]]])

# คำนวณ cosine similarity
similarities = cosine_similarity(input_vector, scaled_features)[0]

data["similarity"] = similarities
similar_houses = data.sort_values(by="similarity", ascending=False).head(10)

print(similar_houses)
