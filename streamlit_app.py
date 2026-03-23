"""
Streamlit — Interface para o sistema de predição de risco de AVC.
Faz chamadas à FastAPI (api.py) e exibe o laudo clínico gerado pelo Gemini.
"""
import os

import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Predição de Risco de AVC",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Sistema de Predição de Risco de AVC")
st.markdown(
    "Preencha os dados do paciente e clique em **Realizar Predição** "
    "para obter a análise e o laudo clínico gerado por IA (Gemini)."
)
st.divider()

# ── Formulário ────────────────────────────────────────────────────────────────
with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dados Pessoais")
        age = st.number_input("Idade (anos)", min_value=1, max_value=120, value=55)
        gender = st.selectbox("Sexo", ["Male", "Female", "Other"],
                              format_func=lambda x: {"Male": "Masculino", "Female": "Feminino", "Other": "Outro"}[x])
        ever_married = st.selectbox("Estado Civil", ["Yes", "No"],
                                    format_func=lambda x: "Casado(a)" if x == "Yes" else "Solteiro(a)")
        work_type = st.selectbox(
            "Tipo de Trabalho",
            ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
            format_func=lambda x: {
                "Private": "Setor Privado",
                "Self-employed": "Autônomo",
                "Govt_job": "Setor Público",
                "children": "Criança",
                "Never_worked": "Nunca trabalhou",
            }[x],
        )
        Residence_type = st.selectbox("Tipo de Residência", ["Urban", "Rural"],
                                      format_func=lambda x: "Urbana" if x == "Urban" else "Rural")

    with col2:
        st.subheader("Dados Clínicos")
        hypertension = st.selectbox("Hipertensão", [0, 1],
                                    format_func=lambda x: "Sim" if x else "Não")
        heart_disease = st.selectbox("Cardiopatia", [0, 1],
                                     format_func=lambda x: "Sim" if x else "Não")
        avg_glucose_level = st.number_input(
            "Glicose Média (mg/dL)", min_value=50.0, max_value=400.0, value=106.0, step=0.5
        )
        bmi = st.number_input("IMC", min_value=10.0, max_value=60.0, value=28.5, step=0.1)
        smoking_status = st.selectbox(
            "Tabagismo",
            ["never smoked", "formerly smoked", "smokes", "Unknown"],
            format_func=lambda x: {
                "never smoked": "Nunca fumou",
                "formerly smoked": "Ex-fumante",
                "smokes": "Fumante",
                "Unknown": "Desconhecido",
            }[x],
        )

    submitted = st.form_submit_button("🔍 Realizar Predição", use_container_width=True, type="primary")

# ── Resultado ─────────────────────────────────────────────────────────────────
if submitted:
    payload = {
        "age":               age,
        "hypertension":      hypertension,
        "heart_disease":     heart_disease,
        "avg_glucose_level": avg_glucose_level,
        "bmi":               bmi,
        "gender":            gender,
        "ever_married":      ever_married,
        "work_type":         work_type,
        "Residence_type":    Residence_type,
        "smoking_status":    smoking_status,
    }

    with st.spinner("Analisando dados e gerando laudo clínico..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=90)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.ConnectionError:
            st.error("Não foi possível conectar à API. Verifique se o servidor está rodando: `uvicorn api:app --reload`")
            st.stop()
        except Exception as e:
            st.error(f"Erro: {e}")
            st.stop()

    st.divider()
    col_resultado, col_laudo = st.columns([1, 2])

    with col_resultado:
        risco = result["risco"]
        prob  = result["probabilidade"]

        if risco == "ALTO":
            st.error(f"## ⚠️ {risco} RISCO DE AVC")
        else:
            st.success(f"## ✅ {risco} RISCO DE AVC")

        st.metric("Probabilidade estimada", f"{prob:.1%}")
        st.caption(f"Modelo: **{result['modelo_utilizado']}**")

        if result.get("shap_top5"):
            st.subheader("Top 5 Fatores (SHAP)")
            for feat, val in result["shap_top5"].items():
                icon = "🔴" if val > 0 else "🟢"
                direction = "aumenta risco" if val > 0 else "reduz risco"
                st.write(f"{icon} `{feat}` — {direction} ({val:+.4f})")

    with col_laudo:
        st.subheader("📋 Laudo Clínico — Gemini")
        st.markdown(result["relatorio_clinico"])

    with st.expander("Ver dados enviados"):
        st.json(payload)

# ── Rodapé ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ Esta ferramenta é de apoio à decisão médica e **não substitui** a avaliação clínica de um profissional de saúde habilitado."
)
