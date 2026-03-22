"""
FastAPI — AVC Stroke Prediction API
Carrega o modelo exportado pelo main.ipynb e expõe endpoint de predição
com SHAP e laudo clínico via Gemini.
"""
import os
import time
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from dotenv import load_dotenv

load_dotenv()

# ── Artefatos ─────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "model_artifacts")

try:
    model           = joblib.load(f"{ARTIFACTS_DIR}/model.pkl")
    scaler          = joblib.load(f"{ARTIFACTS_DIR}/scaler.pkl")
    feature_columns = joblib.load(f"{ARTIFACTS_DIR}/feature_columns.pkl")
    X_train_sample  = joblib.load(f"{ARTIFACTS_DIR}/X_train_sample.pkl")
    with open(f"{ARTIFACTS_DIR}/model_name.txt") as f:
        model_name = f.read().strip()
except FileNotFoundError as e:
    raise RuntimeError(
        "Artefatos não encontrados. Execute o main.ipynb até a célula de exportação primeiro."
    ) from e

NUMERICAL_FEATURES   = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
CATEGORICAL_FEATURES = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

# ── Gemini ─────────────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "models/gemini-3-flash-preview")
gemini_client  = genai.Client(api_key=GOOGLE_API_KEY)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="AVC Prediction API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schema ─────────────────────────────────────────────────────────────────────
class PatientData(BaseModel):
    age:               float = Field(..., ge=1,   le=120,  example=67)
    hypertension:      int   = Field(..., ge=0,   le=1,    example=0)
    heart_disease:     int   = Field(..., ge=0,   le=1,    example=1)
    avg_glucose_level: float = Field(..., ge=50,  le=400,  example=228.69)
    bmi:               float = Field(..., ge=10,  le=60,   example=36.6)
    gender:            str   = Field(..., example="Male")
    ever_married:      str   = Field(..., example="Yes")
    work_type:         str   = Field(..., example="Private")
    Residence_type:    str   = Field(..., example="Urban")
    smoking_status:    str   = Field(..., example="formerly smoked")


# ── Helpers ────────────────────────────────────────────────────────────────────
def preprocess(data: PatientData) -> pd.DataFrame:
    raw = pd.DataFrame([data.model_dump()])
    encoded = pd.get_dummies(raw, columns=CATEGORICAL_FEATURES, drop_first=True)
    # Garantir mesmas colunas do treino
    for col in feature_columns:
        if col not in encoded.columns:
            encoded[col] = 0
    encoded = encoded[feature_columns].astype(float)
    encoded[NUMERICAL_FEATURES] = scaler.transform(encoded[NUMERICAL_FEATURES])
    return encoded


def compute_shap(X_processed: pd.DataFrame) -> np.ndarray | None:
    try:
        if model_name in ("Random Forest", "Decision Tree"):
            explainer = shap.TreeExplainer(model)
            vals = explainer.shap_values(X_processed)
            if isinstance(vals, list):
                vals = vals[1]
        elif model_name == "Logistic Regression":
            explainer = shap.LinearExplainer(model, X_train_sample)
            vals = explainer.shap_values(X_processed)
        else:
            # KNN / SVM: usa amostra pequena de background para KernelExplainer
            bg = shap.sample(X_train_sample, 30)
            explainer = shap.KernelExplainer(model.predict_proba, bg)
            vals = explainer.shap_values(X_processed, nsamples=50)
            if isinstance(vals, list):
                vals = vals[1]
        arr = np.array(vals, dtype=np.float64)
        return arr.flatten()
    except Exception:
        return None


def gemini_generate(prompt: str, retries: int = 3, wait: int = 15) -> str:
    for attempt in range(retries):
        try:
            return gemini_client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt
            ).text
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                time.sleep(wait)
            else:
                raise
    return ""


def build_report(
    data: PatientData,
    predicao: int,
    probabilidade: float,
    shap_vals: np.ndarray | None,
) -> str:
    risco = "ALTO RISCO de AVC" if predicao == 1 else "BAIXO RISCO de AVC"

    if shap_vals is not None:
        top5 = sorted(
            zip(feature_columns, shap_vals), key=lambda x: abs(x[1]), reverse=True
        )[:5]
        fatores_txt = "\n".join(
            f"  - {feat}: {'aumenta' if val > 0 else 'reduz'} o risco (SHAP: {val:+.4f})"
            for feat, val in top5
        )
    else:
        fatores_txt = (
            f"  - Idade: {data.age} anos\n"
            f"  - Glicose: {data.avg_glucose_level} mg/dL\n"
            f"  - IMC: {data.bmi}"
        )

    prompt = f"""Você é um assistente clínico especializado em AVC. Gere um laudo clínico em português brasileiro.

PREDIÇÃO: {risco} (probabilidade: {probabilidade:.1%})
DADOS DO PACIENTE:
  Idade: {data.age} anos | Hipertensão: {"Sim" if data.hypertension else "Não"} | Cardiopatia: {"Sim" if data.heart_disease else "Não"}
  Glicose: {data.avg_glucose_level} mg/dL | IMC: {data.bmi} | Tabagismo: {data.smoking_status}
  Sexo: {data.gender} | Estado civil: {data.ever_married} | Trabalho: {data.work_type} | Residência: {data.Residence_type}

FATORES MAIS RELEVANTES (SHAP):
{fatores_txt}

Gere o laudo com as seções obrigatórias:
**Perfil de risco** | **Fatores determinantes** | **Recomendações** | **Aviso**
Máximo 350 palavras. Seja claro, objetivo e acionável para o médico."""

    return gemini_generate(prompt)


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": model_name}


@app.post("/predict")
def predict(data: PatientData):
    try:
        X         = preprocess(data)
        predicao  = int(model.predict(X)[0])
        prob      = float(model.predict_proba(X)[0][1])
        shap_vals = compute_shap(X)
        relatorio = build_report(data, predicao, prob, shap_vals)

        shap_top5 = None
        if shap_vals is not None:
            top5 = sorted(
                zip(feature_columns, shap_vals.tolist()),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:5]
            shap_top5 = {feat: round(val, 4) for feat, val in top5}

        return {
            "predicao":         predicao,
            "risco":            "ALTO" if predicao == 1 else "BAIXO",
            "probabilidade":    round(prob, 4),
            "modelo_utilizado": model_name,
            "shap_top5":        shap_top5,
            "relatorio_clinico": relatorio,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
