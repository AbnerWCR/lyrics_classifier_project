from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
import joblib


class FeatureExtractor:

    def __init__(self, method='tfidf'):
        self.method = method
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), 
                                              max_features=5000, 
                                              min_df=2, #remove rare words
                                              max_df=0.9, #remove noise
                                              sublinear_tf=True)
        else:
            self.vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=5000)

    def fit_transform(self, corpus):
        print(f"[FeatureExtractor] Vectorizing with {self.method.upper()}")
        return self.vectorizer.fit_transform(corpus)

    def transform(self, text):
        return self.vectorizer.transform(text)

    def save_vectorizer(self, path='vectorizer.joblib'):
        joblib.dump(self.vectorizer, path)
        print(f"Vectorizer saved: {path}")