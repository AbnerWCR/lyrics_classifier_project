from sklearn.naive_bayes import BernoulliNB, MultinomialNB, CategoricalNB
from sklearn.ensemble import RandomForestClassifier
import joblib

class ModelTrainer:
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train, algorithm_type='naive_bayes'):
        print(f"[ModelTrainer] Training: {algorithm_type}")
        
        if algorithm_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif algorithm_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=12)
        else:
            raise ValueError("Algorithm is not recognized.")
            
        self.model.fit(X_train, y_train)

    def get_model(self):
        return self.model

    def save_model(self, path='model.joblib'):
        if self.model:
            joblib.dump(self.model, path)
            print(f"Model saved: {path}")
        else:
            print("Error: Not found trained model to save.")