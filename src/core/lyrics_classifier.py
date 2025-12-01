from pathlib import Path
from src.core.text_preprocessor import TextPreprocessor
import joblib
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Union

class ILyricsClassifier(ABC):
    def __init__(self, model_path: Union[str, Path], vectorizer_path: Union[str, Path]):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path

    @abstractmethod
    def predict(self, song_lyrics: str) -> dict[str, Any]:
        pass


class LyricsClassifier(ILyricsClassifier):
    def __init__(self, model_path, vectorizer_path):
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            
            self.preprocessor = TextPreprocessor() 
            print("LyricsClassifier success loaded!")
        except FileNotFoundError:
            raise Exception("Files with model/vectorizer is not found.")

    def predict(self, song_lyrics: str):
        clean_lyrics = self.preprocessor.process(song_lyrics, use_stemming=False)
        
        vectorized = self.vectorizer.transform([clean_lyrics])
        
        prediction = self.model.predict(vectorized)[0]
        probas = self.model.predict_proba(vectorized)[0]
        
        confidence = np.max(probas)
        
        return {
            "input_preview": song_lyrics[:30] + "...",
            "prediction": prediction,
            "confidence": round(confidence, 4)
        }