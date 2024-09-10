from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI()

# Load the model
model_path = './models/model.pkl'
if not os.path.exists(model_path):
    raise RuntimeError(f"Model file not found at {model_path}")

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

class PredictRequest(BaseModel):
    engagement_score: float
    experience_score: float

@app.post("/predict/")
def predict(request: PredictRequest):
    try:
        # Extract features from the request
        features = np.array([[request.engagement_score, request.experience_score]])
        
        # Make prediction
        prediction = model.predict(features)
        
        # Return the prediction
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
