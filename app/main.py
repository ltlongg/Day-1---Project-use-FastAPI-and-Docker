from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Depends, Query
from pymongo.database import Database

from app.database import get_db
from app.model import classifier
from app.schemas import (
    PredictionResponse,
    Prediction,
    TextRequest,
    PredictionHistoryItem,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML model on startup."""
    classifier.load_model()
    yield


app = FastAPI(
    title="Text Classification API",
    description="Sentiment Analysis API using DistilBERT + MongoDB Compbass storage",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Text Classification API (MongoDB) is running!"}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": classifier.model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest, db: Database = Depends(get_db)):
    """
    Classify text sentiment and save result to MongoDB.

    - **text**: Input text to analyze
    - Returns: Predicted label (POSITIVE/NEGATIVE) with confidence scores
    """
    results = classifier.predict(request.text)

    # Sort by score descending
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    best = sorted_results[0]

    # Save to MongoDB
    record_dict = {
        "text": request.text,
        "label": best["label"],
        "score": round(best["score"], 4),
        "created_at": datetime.now(timezone.utc),
    }
    db.predictions.insert_one(record_dict)

    return PredictionResponse(
        text=request.text,
        prediction=Prediction(label=best["label"], score=round(best["score"], 4)),
        all_scores=[
            Prediction(label=r["label"], score=round(r["score"], 4))
            for r in sorted_results
        ],
    )


@app.get("/history", response_model=list[PredictionHistoryItem])
def get_history(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(20, ge=1, le=100, description="Max records to return"),
    db: Database = Depends(get_db),
):
    """Get prediction history from MongoDB (newest first)."""
    # MongoDB returns cursor. Sort by created_at DESC
    cursor = db.predictions.find().sort("created_at", -1).skip(skip).limit(limit)
    
    # Fast copy
    records = list(cursor)
    
    # We must convert ObjectId to string to match our schema and let Pydantic handle it
    for record in records:
        record["_id"] = str(record["_id"])
        
    return records
