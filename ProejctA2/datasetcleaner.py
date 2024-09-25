import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the datasets
data1 = pd.read_csv('RawData/melb_data.csv')
data2 = pd.read_csv('RawData/Melbourne_housing_FULL.csv')

# Specify the columns to keep
columns_to_keep = ['Bedrooms', 'Type', 'Price', 'Bathrooms', 'Garage', 'Lot_Area', 'SqFt', 'Year_Built', 'Suburb']

# Select only the specified columns from both datasets
data1 = data1[columns_to_keep]
data2 = data2[columns_to_keep]

# Combine the two datasets
combined_data = pd.concat([data1, data2], ignore_index=True)

# Remove duplicate rows
combined_data.drop_duplicates(inplace=True)

# Function to remove outliers based on the IQR method
def remove_outliers(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Remove rows where the column value is below lower_bound or above upper_bound
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# Define numeric columns to check for outliers
numeric_columns = ['Bedrooms', 'Price', 'Bathrooms', 'Garage', 'Lot_Area', 'SqFt', 'Year_Built']

# Remove outliers from numeric columns
combined_data = remove_outliers(combined_data, numeric_columns)

# Handle missing values for numeric columns (e.g., 'Price', 'Bathrooms', etc.)
for column in numeric_columns:
    combined_data[column].fillna(combined_data[column].median(), inplace=True)

# Handle missing values for categorical columns (e.g., 'Type', 'Suburb')
categorical_columns = ['Type', 'Suburb']
for column in categorical_columns:
    combined_data[column].fillna(combined_data[column].mode()[0], inplace=True)

# Save the combined raw data without outliers
combined_data.to_csv('DataSets/combined_housing_data.csv', index=False)

# Standardize numeric columns (Z-score standardization)
scaler = StandardScaler()
combined_data_scaled = combined_data.copy()
combined_data_scaled[numeric_columns] = scaler.fit_transform(combined_data[numeric_columns])

# Save the Z-score standardized dataset
combined_data_scaled.to_csv('DataSets/combined_housing_data_zscore.csv', index=False)

print("Data processing complete. Outliers removed, and raw and standardized datasets saved.")
