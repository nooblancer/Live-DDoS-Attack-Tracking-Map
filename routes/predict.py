"""Prediction route for DDoS classification.

POST /predict — validate input features, scale, run inference, return classification.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from models.schemas import PredictRequest, PredictResponse
from services.model_service import ModelService

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level singleton — set by main.py at startup
model_service: ModelService = ModelService()


def get_model_service() -> ModelService:
    """Return the module-level ModelService instance."""
    return model_service


@router.post("/predict", response_model=PredictResponse)
async def predict_endpoint(
    request: PredictRequest,
    svc: ModelService = Depends(get_model_service),
) -> PredictResponse:
    """Classify network flow features as Benign or DDoS.

    Validates the request body (Pydantic returns 422 automatically on bad input),
    checks that the model is loaded (503 if not), scales features, runs inference,
    and returns the predicted label with probability.
    """
    if not svc.is_loaded:
        raise HTTPException(status_code=503, detail="Model not available")

    try:
        features = request.model_dump()
        result = svc.predict(features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return PredictResponse(label=result["label"], probability=result["probability"])
