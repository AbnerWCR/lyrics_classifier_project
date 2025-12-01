from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, CategoricalNB
from sklearn.ensemble import RandomForestClassifier
from typing import Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from fpdf import FPDF
import os


class Evaluator:

    @staticmethod
    def evaluate(model: Union[MultinomialNB, RandomForestClassifier], X_test, y_test, model_name, path_evaluate):
        print("[Evaluator] Starting Validation")
        predictions = model.predict(X_test)
        Evaluator.generate_report(y_test, predictions, model_name, model.classes_, path_evaluate)

    @staticmethod
    def generate_report(y_true, y_pred, model_name, class_names, path_evaluate):
    
        cm = metrics.confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        
        plt.title(f'Matriz de Confusão - {model_name}')
        plt.ylabel('Real')
        plt.xlabel('Predito')
        plt.tight_layout()
        
        path_model = str(model_name).split("/")
        img_filename = f"conf_matrix_{path_model[-1].replace(".joblib", "")}.png"
        plt.savefig(img_filename)
        plt.close()
        
        pdf = FPDF()
        pdf.add_page()
        
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, f"Relatório de Performance: {model_name}", ln=True, align='C')
        pdf.ln(10)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Métricas Detalhadas (Acurácia, Precision, Recall, F1):", ln=True)
        
        pdf.set_font("Courier", size=10) 
        report_text = metrics.classification_report(y_true, y_pred, target_names=class_names)
        pdf.multi_cell(0, 5, report_text)
        
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Matriz de Confusão:", ln=True)
        
        pdf.image(img_filename, x=10, w=190)
        
        pdf.output(path_evaluate)
        
        if os.path.exists(img_filename):
            os.remove(img_filename)