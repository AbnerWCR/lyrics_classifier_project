import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from fastapi import HTTPException
from typing import Optional, Union
from pathlib import Path

from src.core.lyrics_classifier import ILyricsClassifier, LyricsClassifier
from src.core.text_preprocessor import TextPreprocessor


class PredictionService:
    def __init__(self):
        self.classifier: Optional[ILyricsClassifier] = None

    def load_model(self, model_path: Union[str, Path], vectorizer_path: Union[str, Path]):
        try:
            self.classifier = LyricsClassifier(
                model_path=model_path, 
                vectorizer_path=vectorizer_path
            )
        except Exception as e:
            raise e

    def make_prediction(self, lyrics: str) -> dict:
        if not self.classifier:
            raise HTTPException(status_code=503, detail="Modelo não foi inicializado.")
        
        text_preprocessor = TextPreprocessor()
        lyrics = text_preprocessor.process(lyrics)
        
        return self.classifier.predict(lyrics)