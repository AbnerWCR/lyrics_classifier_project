import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)


from contextlib import asynccontextmanager
import uvicorn
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Depends
from src.api.service import PredictionService
from src.env_loader import PATH_MODEL, PATH_VECTORIZER
from src.api.schemas import MusicRequest, PredictionResponse
import yaml

service = PredictionService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.load_model(PATH_MODEL, PATH_VECTORIZER)
    yield

app = FastAPI(
    title="Lyrics Classifier API",
    description="Microserviço para classificação de gênero musical de acordo com a letra.",
    lifespan=lifespan,
    version="1.0.0"
)

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "lyrics-classifier"
    }

@app.post("/predict", response_model=PredictionResponse, status_code=200)
def predict(request: MusicRequest):
    try:
        result = service.make_prediction(request.lyrics)

        return PredictionResponse(
            genre=result.get("prediction"),
            confidence=result.get("confidence"),
            input_preview=result.get("input_preview")
        )
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Erro ao classificar letra.")


def export():
    openapi_data = app.openapi()
    output_path = 'docs/swagger.yaml'
    with open(output_path, 'w') as f:
        yaml.dump(openapi_data, f, sort_keys=False)
    
    print(f"Swagger salvo com sucesso em: {output_path}")

if __name__ == "__main__":
    export()

    uvicorn.run(
        app=app,
        host="0.0.0.0",
        port=8000
    )