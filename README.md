# AIDI 1000 Final Project - Penguin Species Predictor

## How to Run

1. Install uv (once per computer)

```powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"```

Restart PowerShell → run ```uv --version```

2. Install Python 3.10.16

```uv python install 3.10.16```
3. Create your project

```mkdir fastapi-project && cd fastapi-project```
```uv init```
```uv python pin 3.10.16```
```uv venv```
```.venv\Scripts\Scripts\Activate.ps1```

4. Install dependencies

```uv add numpy pandas pydantic scikit-learn seaborn fastapi "uvicorn[standard]"```

5. Run the app

```uv run fastapi dev main.py```
→ http://127.0.0.1:8000  Interactive docs: http://127.0.0.1:8000/docs


## Dataset Used

**Palmer Penguins Dataset** (344 samples after cleaning)

| Feature | Type | Values |
|---------|------|--------|
| `bill_length_mm` | Numeric | 32-60 mm |
| `bill_depth_mm` | Numeric | 13-22 mm |
| `flipper_length_mm` | Numeric | 172-231 mm |
| `body_mass_g` | Numeric | 2850-6300 g |
| `island` | Categorical | Torgersen, Biscoe, Dream |
| `sex` | Categorical | MALE, FEMALE |
| **`species`** | **Target** | Adelie, Chinstrap, Gentoo |

**Preprocessing**: `dropna()` → `ColumnTransformer(StandardScaler + OneHotEncoder)`

## Model Chosen

**RandomForestClassifier(random_state=42)**
- Default hyperparameters (no tuning)
- All 7 features used (no selection/engineering)
- Pipeline: `preprocess → model`
- Saved as `penguin_model.pkl`

## Evaluation Metrics

| Metric | Train | Test | Gap |
|--------|-------|------|-----|
| **F1-Score (macro)** | **1.00** | **1.00** | **0.00** |
| **ROC-AUC (ovr, macro)** | **1.00** | **1.00** | **0.00** |

## Overfitting Assessment

**Model shows perfect performance (potential overfitting)**

**Evidence**: 
- Perfect F1-score (1.00 = 1.00)
- Perfect ROC-AUC (1.00 = 1.00)

**Explanation**: Perfect scores on both train and test suggest the model may be overfitting by memorizing the small dataset (344 samples). While train/test match indicates good generalization, perfect 1.00 scores are suspicious for real-world data and warrant caution. Consider collecting more data or adding regularization.
