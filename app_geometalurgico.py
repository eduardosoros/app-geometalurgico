import streamlit as st
import numpy as np
import joblib

# === Carregar Modelo, Scaler e LabelEncoder ===
modelo = joblib.load('modelo_geometalurgico.pkl')
scaler = joblib.load('scaler_geometalurgico.pkl')
le = joblib.load('labelencoder_geometalurgico.pkl')

# === Interface Streamlit ===
st.title("Classificador Geometalúrgico 🌱🔎")
st.write("Preencha os valores abaixo para prever o Domínio (GradeShell):")

# Campos de entrada - variáveis usadas no modelo
cu_sol = st.number_input('CuSol', min_value=0.0, step=0.01)
cucit_pct = st.number_input('CuCIT_pct', min_value=0.0, step=0.001)
cus = st.number_input('CuS', min_value=0.0, step=0.001)
rejeito_pc = st.number_input('Rejeito_pc', min_value=0.0, step=0.001)
cu_pct = st.number_input('Cu_pct', min_value=0.0, step=0.001)

# Botão de Previsão
if st.button('Prever Domínio'):
    # Montar array de entrada
    entrada = np.array([[cu_sol, cucit_pct, cus, rejeito_pc, cu_pct]])

    # Padronizar com o mesmo scaler usado no treinamento
    entrada_scaled = scaler.transform(entrada)

    # Fazer previsão
    pred = modelo.predict(entrada_scaled)

    # Converter de número para o nome do domínio
    resultado = le.inverse_transform(pred)

    st.success(f"Domínio Predito: {resultado[0]}")
