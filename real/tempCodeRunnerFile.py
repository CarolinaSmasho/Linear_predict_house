from predict import predict_house
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity



def parse_input(input_string):
    # Define the keys in the order of the input format
    keys = [
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
    
    # Split the input string by tabs
    values = input_string.strip().split('\t')
    
    # Create dictionary to store the input
    input_dict = {}
    
    # Map values to keys, handling data types and excluding None values
    for key, value in zip(keys, values):
        if value != 'NA':  # Only include non-'NA' values
            try:
                # Try converting to int for numeric fields
                if key in ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
                         'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                         'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
                         'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
                         'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',
                         'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
                         'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']:
                    input_dict[key] = int(value) if value else None
                else:
                    input_dict[key] = value
            except ValueError:
                input_dict[key] = value
    
    # Filter out any keys with None or empty string values
    filtered_dict = {k: v for k, v in input_dict.items() if v is not None and v != ''}
    
    return filtered_dict

inp_data = input("(B)uy or (S)ell or (0)Exit: ")

while inp_data != "0":
    if inp_data == "S":
        print("Enter house data in tab-separated format:")
        input_detail = input()
        example_input = parse_input(input_detail)
        predicted_price = predict_house(example_input)
        print(f"\nPredicted house price: ${predicted_price:,.2f}")
        print()
        
        
        
        
        
    elif inp_data == "B":
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
    inp_data = input("B or S or 0: ")
    
# 50	RM	51	6120	Pave	NA	Reg	Lvl	AllPub	Inside	Gtl	OldTown	Artery	Norm	1Fam	1.5Fin	7	5	1931	1950	Gable	CompShg	BrkFace	Wd Shng	None	0	TA	TA	BrkTil	TA	TA	No	Unf	0	Unf	0	952	952	GasA	Gd	Y	FuseF	1022	752	0	1774	0	0	2	0	2	2	TA	8	Min1	2	TA	Detchd	1931	Unf	2	468	Fa	TA	Y	90	0	205	0	0	0	NA	NA	NA	0	4	2008	WD	Abnorml 
# CollgCr 8450 208500