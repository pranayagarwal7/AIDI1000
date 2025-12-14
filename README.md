# Penguin Species Predictor

A complete machine learning project predicting penguin (seaborn dataset) species from body measurements using scikit-learn, with a FastAPI REST endpoint for predictions.

## 📋 Project Structure

aidi1000-project/
├── penguin_model.pkl # Trained ML model (Pipeline)
├── main.py # FastAPI app
├── notebooks/ # Jupyter notebooks for training
│ └── penguin_analysis.ipynb
├── data/ # Raw penguin data (optional)
└── README.md # This file


## 🐧 Dataset

Palmer Penguins dataset with features:
- Numeric: `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`
- Categorical: `island` (Torgersen, Biscoe, Dream), `sex` (MALE, FEMALE)
- Target: `species` (Adelie, Chinstrap, Gentoo)

## 🚀 Quick Start

### 1. Install dependencies
pip install fastapi uvicorn pandas scikit-learn seaborn matplotlib

### 2. Run FastAPI server
uvicorn main:app --reload

### 3. Test the API
Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive Swagger UI

## 🔬 ML Pipeline
**Preprocessing** (applied automatically by model):
Numeric columns → StandardScaler
Categorical → OneHotEncoder
All features → RandomForestClassifier 


**Model Evaluation** (example results):

| Metric      | Train | Test  | Overfitting? |
|-------------|-------|-------|--------------|
| F1-score    | 0.95  | 0.82  | Yes (gap)    |
| ROC-AUC     | 0.98  | 0.89  | Yes (gap)    |

**Overfitting Assessment**: Large gap between train/test scores indicates overfitting. Model memorizes training patterns but generalizes less well to unseen data.

## 📊 API Endpoints

### `POST /predict`
**Input schema**: