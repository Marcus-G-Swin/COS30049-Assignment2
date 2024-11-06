'''import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# A simple regression model
class SimpleModel:
    def __init__(self):
        # Initialize the model (Linear Regression)
        self.model = LinearRegression()

    def train(self):
        # Example training data: X = [[square footage, bedrooms]], y = [price]
        X = np.array([[1500, 3], [1200, 2], [1800, 4], [2000, 5], [1400, 2], [1600, 3]])
        y = np.array([300000, 250000, 400000, 500000, 270000, 320000])
        
        # Train the model
        self.model.fit(X, y)
        
        # Save the model
        joblib.dump(self.model, 'simple_model.pkl')

        # Evaluation
        predictions = self.model.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)

        print(f"Model trained. MSE: {mse}, R²: {r2}")

    def predict(self, square_footage, bedrooms):
        # Load the model
        model = joblib.load('simple_model.pkl')
        
        # Make a prediction based on input
        return model.predict([[square_footage, bedrooms]])

# Example usage (for initial training)
if __name__ == "__main__":
    model = SimpleModel()
    model.train()
    '''
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI()

# Load dataset (update path if needed)
df_raw = pd.read_csv('D:/SWINUNI/6membersteam/COS30049-Assignment2/FastAPI/DataSets/combined_housing_data.csv', usecols=['Bedrooms', 'Type', 'Price', 'Bathrooms', 'Garage', 'Lot_Area', 'SqFt', 'Year_Built', 'Suburb'])

# Split data into features and target
def split_data(df, target='Price'):
    X = df[['Bedrooms', 'Type', 'Bathrooms', 'Garage', 'Lot_Area', 'SqFt', 'Year_Built', 'Suburb']]
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = split_data(df_raw)

# Preprocessing pipeline
def get_preprocessor():
    transformers = [
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Type', 'Suburb']),
        ('num', StandardScaler(), ['Bedrooms', 'Bathrooms', 'Garage', 'Lot_Area', 'SqFt', 'Year_Built'])
    ]
    return ColumnTransformer(transformers=transformers)

# Regression model class
class SimpleModel:
    def __init__(self):
        # Initialize preprocessing and model pipeline
        preprocessor = get_preprocessor()
        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('linear_regression', LinearRegression())])

    def train(self, X_train, y_train):
        # Train the model
        self.pipeline.fit(X_train, y_train)
        joblib.dump(self.pipeline, 'simple_model.pkl')
        
        # Evaluation
        y_pred = self.pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Model trained. MSE: {mse:.2f}, R²: {r2:.4f}")

    def predict(self, data):
        # Load the model
        model = joblib.load('simple_model.pkl')
        
        # Predict using the loaded model
        return model.predict([data])

# Initialize and train the model (only once)
model = SimpleModel()
model.train(X_train, y_train)

# Define request and response models
class PredictionRequest(BaseModel):
    Bedrooms: int
    Type: str
    Bathrooms: int
    Garage: int
    Lot_Area: float
    SqFt: float
    Year_Built: int
    Suburb: str

class PredictionResponse(BaseModel):
    predicted_price: float

# Endpoint for predicting house price
@app.post("/predict/", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    try:
        # Combine all features into a list, as expected by the `predict` function
        data = [
            request.Bedrooms, request.Type, request.Bathrooms, 
            request.Garage, request.Lot_Area, request.SqFt, 
            request.Year_Built, request.Suburb
        ]
        
        # Call predict with a single argument, as defined in SimpleModel
        predicted_price = model.predict(data)[0]

        return PredictionResponse(predicted_price=predicted_price)
    
    except Exception as e:
        print(f"Prediction error: {e}")  # Detailed error message
        raise HTTPException(status_code=500, detail=f"Error predicting price: {e}")

# Example usage:
# Start the FastAPI server and use this endpoint:
# POST http://localhost:8000/predict/
# with JSON data:
# {
#   "Bedrooms": 3,
#   "Type": "House",
#   "Bathrooms": 2,
#   "Garage": 1,
#   "Lot_Area": 5000,
#   "SqFt": 1500,
#   "Year_Built": 1995,
#   "Suburb": "SuburbName"
# }




