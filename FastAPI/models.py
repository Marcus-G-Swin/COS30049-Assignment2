import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from fastapi import FastAPI, HTTPException

# Initialize FastAPI
app = FastAPI()

# Regression model class
class SimpleModel:
    def __init__(self):
        # Initialize the pipeline with preprocessing and linear regression
        self.pipeline = Pipeline([
            ('preprocessor', ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), ['SqFt', 'Bedrooms', 'Bathrooms', 'Garage', 'Lot_Area', 'Year_Built']),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), ['Type', 'Suburb'])
                ])),
            ('linear_regression', LinearRegression())
        ])

    def load_data(self, file_path):
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Features and target
        X = df[['SqFt', 'Bedrooms', 'Bathrooms', 'Garage', 'Lot_Area', 'Year_Built', 'Type', 'Suburb']]
        y = df['Price'].values

        return X, y

    def train(self, file_path='combined_housing_data.csv'):
        # Load data from CSV
        X, y = self.load_data(file_path)
        
        # Train the model
        self.pipeline.fit(X, y)
        
        # Save the model
        joblib.dump(self.pipeline, 'simple_model.pkl')

        # Evaluation
        predictions = self.pipeline.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        print(f"Model trained. MSE: {mse:.2f}, R²: {r2:.4f}")

    def predict(self, square_footage, bedrooms, bathrooms, garage, lot_area, year_built, house_type, suburb):
        # Load the model
        model = joblib.load('simple_model.pkl')
        
        # Prepare input data as a DataFrame
        input_data = pd.DataFrame([{
            'SqFt': square_footage,
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'Garage': garage,
            'Lot_Area': lot_area,
            'Year_Built': year_built,
            'Type': house_type,
            'Suburb': suburb
        }])

        # Predict price
        return model.predict(input_data)

# Initialize and train the model
model = SimpleModel()
model.train(file_path='combined_housing_data.csv')

# Endpoint for predicting house price
@app.get("/predict/{square_footage}/{bedrooms}/{bathrooms}/{garage}/{lot_area}/{year_built}/{house_type}/{suburb}")
async def predict_price(
    square_footage: float,
    bedrooms: int,
    bathrooms: int,
    garage: int,
    lot_area: float,
    year_built: int,
    house_type: str,
    suburb: str
):
    try:
        price = model.predict(
            square_footage, bedrooms, bathrooms, garage,
            lot_area, year_built, house_type, suburb
        )[0]
        return {"predicted_price": price}
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Error predicting price.")
