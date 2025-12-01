from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="../../.env")

PATH_DATASET = os.getenv("PATH_DATASET")
PATH_VECTORIZER = os.getenv("PATH_VECTORIZER")
PATH_MODEL = os.getenv("PATH_MODEL")
PATH_EVALUATE = os.getenv("PATH_EVALUATE")


def validate_dotenv_variables():
    if PATH_DATASET is None or len(PATH_DATASET) <= 0:
        raise EnvironmentError(f"Environment variable PATH_DATASET is not found on .env file")

    if PATH_VECTORIZER is None or len(PATH_VECTORIZER) <= 0:
        raise EnvironmentError(f"Environment variable PATH_VECTORIZER is not found on .env file")

    if PATH_MODEL is None or len(PATH_MODEL) <= 0:
        raise EnvironmentError(f"Environment variable PATH_MODEL is not found on .env file")
    
    if PATH_EVALUATE is None or len(PATH_EVALUATE) <= 0:
        raise EnvironmentError(f"Environment variable PATH_EVALUATE is not found on .env file")


validate_dotenv_variables()