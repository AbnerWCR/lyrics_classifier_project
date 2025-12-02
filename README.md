# Classificação de Gêneros Musicais com NLP

Este projeto implementa um pipeline completo de classificação de gênero musical a partir de letras de músicas, utilizando técnicas de Processamento de Linguagem Natural (NLP) e algoritmos de Machine Learning. O sistema inclui módulos de pré-processamento, extração de features, treinamento, avaliação, inferência e disponibilização via API (FastAPI).

---

## 📁 Estrutura do Projeto

```text
src/
├── api/
│   ├── app.py           # Entrypoint da API (FastAPI)
│   ├── schemas.py       # Pydantic models
│   └── service.py       # Regra de negócio da API
├── core/
│   ├── corpus_loader.py
│   ├── evaluator.py
│   ├── feature_extractor.py
│   ├── lyrics_classifier.py
│   ├── model_trainer.py
│   └── text_preprocessor.py
├── training/
│   ├── training_pipeline.py  # Script principal de treino
│   └── env_loader.py
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🧩 Componentes Principais

### `src/core/`

#### **CorpusLoader**
Responsável por abstrair a origem e carregamento dos dados, permitindo leitura a partir de arquivos CSV, JSON ou outras fontes estruturadas.

#### **TextPreprocessor**
Executa o pré-processamento textual, incluindo:
- Normalização e limpeza;
- Remoção de stopwords;
- Lematização;
- Tratamento de caracteres especiais.

#### **FeatureExtractor**
Aplica métodos de vetorização, como:
- TF-IDF (Term Frequency–Inverse Document Frequency);
- Bag-of-Words (BoW).

Gera matrizes numéricas utilizadas pelos algoritmos de aprendizado.

#### **ModelTrainer**
Gerencia o treinamento do modelo de Machine Learning:
- Escolha do algoritmo;
- Treinamento supervisionado;
- Salvamento dos artefatos (`.joblib`).

#### **Evaluator**
Responsável pela avaliação do modelo, calculando métricas como:
- Acurácia;
- Precision;
- Recall;
- F1-Score.

#### **LyricsClassifier**
Wrapper final para inferência.
Carrega o modelo e o vetorizador salvos e expõe o método de previsão usado pela API.

---

### `src/api/`

#### **app.py**
Ponto de entrada da API utilizando FastAPI. Define rotas como:
- `GET /health`
- `POST /predict`  
  Recebe texto de letras e retorna o gênero previsto.

#### **service.py**
Implementa a lógica de serviço da API. Utiliza `LyricsClassifier` como singleton para evitar carregamentos repetidos em cada requisição.

#### **schemas.py**
Define contratos de entrada e saída via Pydantic:
- Payload da predição;
- Estrutura da resposta.

---

### `src/training/`

#### **training_pipeline.py**
Pipeline completo de treinamento:
1. Carrega dataset via `CorpusLoader`;
2. Pré-processa textos com `TextPreprocessor`;
3. Extrai features com `FeatureExtractor`;
4. Treina o modelo via `ModelTrainer`;
5. Avalia os resultados usando `Evaluator`;
6. Salva o modelo e o vetorizador nos caminhos definidos pelo `.env`.

#### **env_loader.py**
Carrega variáveis de ambiente necessárias para o processo de treinamento e execução.

---

## ⚙️ Variáveis de Ambiente

As seguintes variáveis devem ser definidas (ver `.env.example`):

| Variável             | Descrição                                        |
|----------------------|--------------------------------------------------|
| `PATH_DATASET`       | Caminho para o dataset de treino                 |
| `PATH_VECTORIZER`    | Caminho para salvar/carregar o vetorizador       |
| `PATH_MODEL`         | Caminho para salvar/carregar o modelo treinado   |
| `PATH_EVALUATE`      | Caminho para salvar métricas e relatórios        |

---

## 🧪 Execução do Treinamento

Execute o pipeline completo:

```bash
python3.13 src/training/training_pipeline.py
```

Os artefatos gerados (modelo e vetorizador) serão salvos conforme configurado no `.env`.

---

## 🚀 Execução da API

Inicie a API local:

```bash
python3.13 src/api/app.py
```

A API estará disponível em:

```
http://localhost:8000
```

---

## 📦 Dependências

Instale os requisitos do projeto:

```bash
pip install -r requirements.txt
```

---

