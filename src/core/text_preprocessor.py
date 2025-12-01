import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.rslp import RSLPStemmer

class TextPreprocessor:
    """Processing text from lyrics
    - stop_words: nltk corpus stopwords from portuguese
    """

    def __init__(self):
        nltk.download('stopwords')
        nltk.download('rslp')   
        self.stop_words = stopwords.words("portuguese")
        self.stemmer = RSLPStemmer()

    def remove_stopwords(self, text: str) -> str:
        words = text.split()
        filtered = [w for w in words if w not in self.stop_words]
        return " ".join(filtered)
    
    def lemmatize(self, text: str) -> str:
        words = text.split()
        stemmed = [self.stemmer.stem(w) for w in words]
        return " ".join(stemmed)

    def clean_text(self, text: str):

        if not isinstance(text, str):
            return ""
        
        text = re.sub(r'[^a-zA-Z\sà-úÀ-Ú]', '', text) 
        text = text.replace('\n', ' ')

        return text.lower().strip()
    
    def process(self, lyrics_text: str, use_stemming=False) -> str:
        lyrics_text = self.clean_text(lyrics_text)
        lyrics_text = self.remove_stopwords(lyrics_text)
        
        if use_stemming:
            lyrics_text = self.lemmatize(lyrics_text)
            
        return lyrics_text
        