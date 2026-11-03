"""
FastAPI app for phishing URL detection.

Endpoints:
- GET  /          Welcome
- GET  /health    Health check
- POST /predict   URL prediction
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import PredictRequest, PredictResponse, HealthResponse, ErrorResponse
from api.predictor import predict, get_model_info

app = FastAPI(
    title="Phishing URL Detection API",
    description="Detects phishing URLs using a robust XGBoost model.",
    version="1.0.0",
)

# CORS — allow Streamlit frontend (Day 9) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "Phishing URL Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    info = get_model_info()
    return HealthResponse(
        status="healthy" if info["model_loaded"] else "unhealthy",
        model_loaded=info["model_loaded"],
        model_version=info["model_version"],
    )


@app.post("/predict", response_model=PredictResponse, responses={400: {"model": ErrorResponse}})
def predict_endpoint(request: PredictRequest):
    try:
        result = predict(request.url)
        return PredictResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")