from main import predict_house

inp_data = input("B or S or 0: ")



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


while inp_data != 0:
    if inp_data == "B":
        print("Enter house data in tab-separated format:")
        input_detail = input()
        example_input = parse_input(input_detail)
        
        
        
        
        
    elif inp_data == "S":
        print("S")
    inp_data = input("B or S or 0: ").strip().split()