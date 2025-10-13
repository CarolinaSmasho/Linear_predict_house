import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from predict import predict_house
import os

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

def select_neighborhood(neighborhoods):
    """Allow user to select a neighborhood by entering a number."""
    while True:
        # Clear the console for a clean display
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Select a Neighborhood by entering the corresponding number:")
        for i, neighborhood in enumerate(neighborhoods, 1):
            print(f"{i}. {neighborhood}")
        
        try:
            choice = input("\nEnter the number of your choice (1-{}): ".format(len(neighborhoods)))
            choice = int(choice)
            if 1 <= choice <= len(neighborhoods):
                return neighborhoods[choice - 1]
            else:
                print("Invalid number. Please choose a number between 1 and {}.".format(len(neighborhoods)))
                input("Press Enter to continue...")
        except ValueError:
            print("Invalid input. Please enter a number.")
            input("Press Enter to continue...")

def main():
    # Load data
    df = pd.read_csv("dataold.csv")
    
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
            # Get unique neighborhoods from the dataset
            neighborhoods = sorted(df["Neighborhood"].unique())
            
            # Select neighborhood using text-based menu
            selected_neighborhood = select_neighborhood(neighborhoods)
            
            # Get LotArea and SalePrice input
            inp = input(f"Selected Neighborhood: {selected_neighborhood}\nEnter LotArea, SalePrice: ").split(" ")
            input_data = {
                "Neighborhood": selected_neighborhood,
                "LotArea": int(inp[0]),
                "SalePrice": int(inp[1])
            }
            
            # Prepare data for similarity calculation
            data = df[["Neighborhood", "LotArea", 'SalePrice']].copy()
            data["Neighborhood_match"] = data["Neighborhood"].apply(
                lambda x: 1 if x == input_data["Neighborhood"] else 0
            )
            
            # Scale features
            scaler = MinMaxScaler()
            scaled_features = scaler.fit_transform(data[["Neighborhood_match", "LotArea", "SalePrice"]])
            
            # Scale input vector
            input_vector = scaler.transform([[1, input_data["LotArea"], input_data["SalePrice"]]])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(input_vector, scaled_features)[0]
            
            # Add similarity scores to the dataframe and get top 10 similar houses
            data["similarity"] = similarities
            similar_houses = data.sort_values(by="similarity", ascending=False).head(10)
            
            print("\nTop 10 similar houses:")
            print(similar_houses)
        
        inp_data = input("\n(B)uy or (S)ell or (0)Exit: ")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")