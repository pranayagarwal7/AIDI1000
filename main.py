from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,Field, field_validator
import pandas as pd
import pickle

class PredictionInput(BaseModel):
    bill_length_mm: float = Field(..., gt=0, description="Bill length in mm")
    bill_depth_mm: float = Field(..., gt=0, description="Bill depth in mm")
    flipper_length_mm: float = Field(..., gt=0, description="Flipper length in mm")
    body_mass_g: float = Field(..., gt=0, description="Body mass in grams")
    island: str = Field(..., strip_whitespace=True, description="Island name")
    sex: str = Field(..., strip_whitespace=True, description="Sex")
    
    @field_validator('island')
    @classmethod
    def validate_island(cls, v):
        valid_islands = ["Torgersen", "Biscoe", "Dream"]
        if v not in valid_islands:
            raise ValueError(f"island must be one of: {', '.join(valid_islands)}")
        return v.title()
    
    @field_validator('sex')
    @classmethod
    def validate_sex(cls, v):
        valid_sex = ["MALE", "FEMALE"]
        if v.upper() not in valid_sex:
            raise ValueError(f"sex must be one of: {', '.join(valid_sex)}")
        return v.upper()

class PredictionOutput(BaseModel):
    species: str
    
app = FastAPI(title= "Penguin Species Prediction",
              description="Predicts penguin species from body measurements using a trained ML model.")


model_path = 'penguin_classifier_model.pkl'
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    # Fail fast at startup if model can't be loaded
    raise RuntimeError(f"Could not load model from {model_path}: {e}")
    
@app.get("/")
def root():
    return {"message": "Penguin predictor is running. Visit /docs for API documentation."}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Predict penguin species from body measurements and metadata.
    """
    try:
        # Optional domain-level check → 400 if clearly bad input
        if input_data.body_mass_g < 2000:
            raise HTTPException(
                status_code=400,
                detail="body_mass_g must be at least 2000 grams for a reliable prediction.",
            )

        # Convert to DataFrame to match training format
        df = pd.DataFrame([input_data.dict()])

        # Make prediction
        pred = model.predict(df)[0]

        proba_dict = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)[0]
            classes = model.classes_
            proba_dict = {str(c): float(p) for c, p in zip(classes, probs)}

        return PredictionOutput(species=str(pred), probabilities=proba_dict)

    except HTTPException:
        # Preserve any explicit 400s raised above
        raise
    except ValueError as ve:
        # Model or preprocessing got bad input → treat as 400
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input for prediction: {ve}",
        )
    except Exception as e:
        # Catch‑all for unexpected server errors
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during prediction: {e}",
        )