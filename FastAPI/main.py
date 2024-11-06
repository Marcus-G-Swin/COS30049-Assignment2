from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import SimpleModel

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # URL of React application
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model
model = SimpleModel()

@app.get("/")
async def root():
    return {"message": "Welcome to the House Price Prediction API"}

@app.get("/predict/{square_footage}/{bedrooms}")
async def predict_price(square_footage: int, bedrooms: int):
    price = model.predict(square_footage, bedrooms)[0]
    return {"predicted_price": round(price, 2)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)