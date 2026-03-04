from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated

# Custom type for MongoDB ObjectId
PyObjectId = Annotated[str, BeforeValidator(str)]

class TextRequest(BaseModel):
    """Request body for text classification."""
    text: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "I love this product! It's amazing and works perfectly."}
            ]
        }
    }


class Prediction(BaseModel):
    """A single prediction result."""
    label: str
    score: float


class PredictionResponse(BaseModel):
    """Response from the /predict endpoint."""
    text: str
    prediction: Prediction
    all_scores: list[Prediction]


class PredictionHistoryItem(BaseModel):
    """A saved prediction record from the database."""
    id: PyObjectId = Field(alias="_id", default=None)
    text: str
    label: str
    score: float
    created_at: datetime

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )
