import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from sklearn.model_selection import train_test_split
from src.core.corpus_loader import CorpusLoader 
from src.core.text_preprocessor import TextPreprocessor 
from src.core.feature_extractor import FeatureExtractor 
from src.core.model_trainer import ModelTrainer 
from src.core.evaluator import Evaluator 
from src.core.lyrics_classifier import LyricsClassifier
from src.env_loader import PATH_DATASET, PATH_MODEL, PATH_VECTORIZER, PATH_EVALUATE


def run_training_pipeline():
    print("\n>>> INICIANDO PIPELINE DE TREINAMENTO <<<")
    
    # 1. Ingestão
    loader = CorpusLoader(PATH_DATASET, ["musica", "genero"])
    df = loader.load_data()

    # 2. Pré-processamento
    preprocessor = TextPreprocessor()
    df['clean_lyrics'] = df['musica'].apply(lambda x: preprocessor.process(x, use_stemming=False))

    # 3. Split (Hold-out)
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_lyrics'], df['genero'], test_size=0.3, random_state=42
    )

    # 4. Extração de Features
    extractor = FeatureExtractor(method='tfidf')
    X_train_vec = extractor.fit_transform(X_train)
    X_test_vec = extractor.transform(X_test) 
    
    extractor.save_vectorizer(PATH_VECTORIZER)

    # 5. Treinamento
    trainer = ModelTrainer()
    trainer.train(X_train_vec, y_train, algorithm_type='random_forest')
    trainer.save_model(PATH_MODEL)

    # 6. Avaliação
    Evaluator.evaluate(trainer.get_model(), X_test_vec, y_test, PATH_MODEL, PATH_EVALUATE)

def run_inference_demo():
    print("\n>>> INICIANDO DEMO DE INFERÊNCIA (CONSUMO) <<<")
    
    classifier = LyricsClassifier(model_path=PATH_MODEL, vectorizer_path=PATH_VECTORIZER)

    nova_letra = """
    Eu sei que vou te amar
    Por toda a minha vida eu vou te amar
    Em cada despedida eu vou te amar
    """

    resultado = classifier.predict(nova_letra)
    print("\nResultado da Classificação:")
    print(resultado)

if __name__ == "__main__":
    run_training_pipeline()
    # run_inference_demo()