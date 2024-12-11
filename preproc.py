import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load the dataset
df = pd.read_csv('all_brands_combined_more.csv')

# List of numerical columns
numerical_columns = ['Year', 'Price', 'Mileage']

# Ensure numerical columns are converted to numeric values (coerce errors to NaN)
for col in numerical_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle missing values by filling with the median of each column
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())

# Min-Max normalization for 'Year' column
min_max_scaler = MinMaxScaler()
df['Year'] = min_max_scaler.fit_transform(df[['Year']])

# Standardization for 'Price' and 'Mileage' columns
standard_scaler = StandardScaler()
# df[['Price', 'Mileage']] = standard_scaler.fit_transform(df[['Price', 'Mileage']])
df['Mileage'] = standard_scaler.fit_transform(df[['Mileage']])

# List of categorical columns
categorical_columns = ['Brand', 'Model', 'Fuel Type', 'Drivetrain', 'Bodystyle']

# Apply one-hot encoding for categorical features
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Display the first few rows of the DataFrame after preprocessing
print(df_encoded.head())

# Save the preprocessed DataFrame with one-hot encoding to a new CSV file
df_encoded.to_csv('preprocessed_cars_data_with_onehot_more.csv', index=False)
print("Preprocessing complete! The cleaned dataset has been saved as 'preprocessed_cars_data_with_onehot.csv'.")
