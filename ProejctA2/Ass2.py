import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# Load the raw dataset including 'Suburb'
df_raw = pd.read_csv('DataSets/combined_housing_data.csv', usecols=['Bedrooms', 'Type', 'Price', 'Bathrooms', 'Garage', 'Lot_Area', 'SqFt', 'Year_Built', 'Suburb'])

# Function to split the data into features (X) and target (y)
def split_data(df):
    X = df[['Bedrooms', 'Type', 'Bathrooms', 'Garage', 'Lot_Area', 'SqFt', 'Year_Built', 'Suburb']]
    y = df['Price']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing pipeline with scaling for models that benefit from it
def get_preprocessor(apply_scaling):
    transformers = [
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Type', 'Suburb'])  # Encode categorical columns
    ]
    
    if apply_scaling:
        transformers.append(('num', StandardScaler(), ['Bedrooms', 'Bathrooms', 'Garage', 'Lot_Area', 'SqFt', 'Year_Built']))
    
    return ColumnTransformer(transformers=transformers)

# Function to train and evaluate a model with simplified output (first 10 results)
def train_and_evaluate_model(model_name, model, X_train, X_test, y_train, y_test, apply_scaling=False):
    # Get appropriate preprocessor based on whether scaling is applied
    preprocessor = get_preprocessor(apply_scaling)
    
    # Create a pipeline for preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        (model_name, model)
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Output first 10 results for ease of reading
    results = pd.DataFrame({'Actual Price': y_test[:10], 'Predicted Price': y_pred[:10]})
    
    # Round the results to avoid scientific notation
    results = results.round(0)
    print(f"\n--- {model_name} Results (first 10 predictions) ---")
    print(results.to_string(index=False))

    # Calculate and print the Mean Squared Error (MSE) and R-squared value
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error ({model_name}): {mse:.2f}")
    print(f"R-squared ({model_name}): {r2:.4f}")

# Split the raw data
X_train, X_test, y_train, y_test = split_data(df_raw)

# Models to be used
linear_regression = LinearRegression()
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
knn = KNeighborsRegressor(n_neighbors=5)

# Train and evaluate models without scaling (for Random Forest)
print("Results without Scaling (Random Forest):")
train_and_evaluate_model('Random Forest', random_forest, X_train, X_test, y_train, y_test, apply_scaling=False)

# Train and evaluate models with scaling (for Linear Regression and KNN)
print("\nResults with Scaling (Linear Regression and KNN):")
train_and_evaluate_model('Linear Regression', linear_regression, X_train, X_test, y_train, y_test, apply_scaling=True)
train_and_evaluate_model('K-Nearest Neighbors', knn, X_train, X_test, y_train, y_test, apply_scaling=True)
