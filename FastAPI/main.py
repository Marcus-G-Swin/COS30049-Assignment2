import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with the specific origin, e.g., "http://localhost:3000"
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Define request model with optional fields
class PredictionRequest(BaseModel):
    square_footage: float = None
    bedrooms: int = None
    bathrooms: int = None
    garage: int = None
    lot_area: float = None
    year_built: int = None
    house_type: str = None
    suburb: str = None

# Define the prediction model class
class SimpleModel:
    def __init__(self):
        # Define preprocessor with StandardScaler and OneHotEncoder for categorical features
        self.pipeline = Pipeline([
            ('preprocessor', ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), ['SqFt', 'Bedrooms', 'Bathrooms', 'Garage', 'Lot_Area', 'Year_Built']),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), ['Type', 'Suburb'])
                ],
                remainder='drop'  # Drop any columns not specified
            )),
            ('linear_regression', LinearRegression())
        ])

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        X = df[['SqFt', 'Bedrooms', 'Bathrooms', 'Garage', 'Lot_Area', 'Year_Built', 'Type', 'Suburb']]
        y = df['Price'].values
        return X, y

    def train(self, file_path='combined_housing_data.csv'):
        X, y = self.load_data(file_path)
        self.pipeline.fit(X, y)
        joblib.dump(self.pipeline, 'simple_model.pkl')
        predictions = self.pipeline.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        print(f"Model trained. MSE: {mse:.2f}, R²: {r2:.4f}")

    def predict(self, data):
        # Load the trained model
        model = joblib.load('simple_model.pkl')
        
        # Create DataFrame only with provided data
        input_data = pd.DataFrame([data])
        
        # Keep only the columns present in input data and expected by the model
        relevant_columns = [col for col in input_data.columns if col in model.named_steps['preprocessor'].get_feature_names_out()]
        input_data = input_data[relevant_columns]
        
        # Predict based on available features
        return model.predict(input_data)

# Initialize and train the model
model = SimpleModel()
model.train(file_path='combined_housing_data.csv')

# Define the prediction endpoint
@app.post("/predict/")
async def predict_price(request: PredictionRequest):
    # Extract fields from the request and filter out None values
    data = {
        'SqFt': request.square_footage,
        'Bedrooms': request.bedrooms,
        'Bathrooms': request.bathrooms,
        'Garage': request.garage,
        'Lot_Area': request.lot_area,
        'Year_Built': request.year_built,
        'Type': request.house_type,
        'Suburb': request.suburb
    }
    # Remove None values to avoid missing data errors
    data = {k: v for k, v in data.items() if v is not None}
    
    # Enforce a minimum number of fields to avoid overly sparse input
    if len(data) < 2:
        raise HTTPException(status_code=400, detail="At least two fields are required for prediction.")
    
    try:
        predicted_price = model.predict(data)[0]
        return {"predicted_price": predicted_price}
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Error predicting price: {e}")
