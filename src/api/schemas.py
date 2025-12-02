
from pydantic import BaseModel, Field

class MusicRequest(BaseModel):
    lyrics: str = Field(
        ..., 
        description="A letra da música a ser classificada",
        min_length=10,
        example="Eu sei que vou te amar, por toda a minha vida..."
    )

class PredictionResponse(BaseModel):
    genre: str
    confidence: float
    input_preview: str
    model_version: str = "1.0.0"