"""
Pydantic Models for Bank Churn API
Defines request and response schemas for API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional


class CustomerFeatures(BaseModel):
    """
    Input features for churn prediction.
    """
    CreditScore: int = Field(..., ge=300, le=850, description="Customer credit score (300-850)")
    Age: int = Field(..., ge=18, le=100, description="Customer age")
    Tenure: int = Field(..., ge=0, le=10, description="Years with the bank")
    Balance: float = Field(..., ge=0, description="Account balance")
    NumOfProducts: int = Field(..., ge=1, le=4, description="Number of bank products")
    HasCrCard: int = Field(..., ge=0, le=1, description="Has credit card (0 or 1)")
    IsActiveMember: int = Field(..., ge=0, le=1, description="Is active member (0 or 1)")
    EstimatedSalary: float = Field(..., ge=0, description="Estimated salary")
    Geography_Germany: int = Field(..., ge=0, le=1, description="From Germany (0 or 1)")
    Geography_Spain: int = Field(..., ge=0, le=1, description="From Spain (0 or 1)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "CreditScore": 650,
                "Age": 35,
                "Tenure": 5,
                "Balance": 50000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 75000.0,
                "Geography_Germany": 0,
                "Geography_Spain": 1
            }
        }


class PredictionResponse(BaseModel):
    """
    Response model for churn prediction.
    """
    churn_probability: float = Field(..., description="Probability of customer churn (0-1)")
    prediction: int = Field(..., description="Binary prediction (0: no churn, 1: churn)")
    risk_level: str = Field(..., description="Risk level: Low, Medium, or High")
    
    class Config:
        json_schema_extra = {
            "example": {
                "churn_probability": 0.3245,
                "prediction": 0,
                "risk_level": "Medium"
            }
        }


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.
    """
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True
            }
        }