import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── DATOS DE LOS 4 CASOS ──────────────────────────────────────────────────────

casos = {
    "Caso 1: Temperatura vs Consumo": {
        "x_label": "Temperatura exterior (°C)",
        "y_label": "Consumo eléctrico (kWh)",
        "slider_label": "🌡️ Temperatura (°C)",
        "slider_min": 15,
        "slider_max": 45,
        "slider_default": 25,
        "x": np.array([20, 22, 24, 26, 28, 30, 32]),
        "y": np.array([18, 20, 23, 26, 29, 32, 35]),
    },
    "Caso 2: Humedad vs Consumo": {
        "x_label": "Humedad relativa (%)",
        "y_label": "Consumo eléctrico (kWh)",
        "slider_label": "💧 Humedad (%)",
        "slider_min": 30,
        "slider_max": 80,
        "slider_default": 55,
        "x": np.array([40, 45, 50, 55, 60, 65, 70]),
        "y": np.array([19, 20, 22, 24, 26, 27, 29]),
    },
    "Caso 3: Dispositivos Activos vs Consumo": {
        "x_label": "Número de dispositivos activos",
        "y_label": "Consumo eléctrico (kWh)",
        "slider_label": "🔌 Dispositivos activos",
        "slider_min": 1,
        "slider_max": 18,
        "slider_default": 8,
        "x": np.array([2, 4, 6, 8, 10, 12, 14]),
        "y": np.array([13, 16, 21, 24, 27, 33, 35]),
    },
    "Caso 4: Horas de Aire Acondicionado vs Consumo": {
        "x_label": "Horas de uso del aire acondicionado",
        "y_label": "Consumo eléctrico (kWh)",
        "slider_label": "❄️ Horas de uso (aire acondicionado)",
        "slider_min": 0,
        "slider_max": 12,
        "slider_default": 4,
        "x": np.array([1, 2, 3, 4, 5, 6, 7]),
        "y": np.array([15, 18, 21, 25, 29, 33, 37]),
    },
}

# ── INTERFAZ ──────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Predicción de Consumo Eléctrico", layout="centered")
st.title("⚡ Sistema Interactivo de Regresión Lineal")
st.markdown("**Predicción y optimización del consumo eléctrico en hogares inteligentes**")
st.divider()

# Selector de caso
caso_seleccionado = st.selectbox("📂 Selecciona el caso a analizar:", list(casos.keys()))
datos = casos[caso_seleccionado]

X = datos["x"].reshape(-1, 1)
y = datos["y"]

# ── MODELO ────────────────────────────────────────────────────────────────────
modelo = LinearRegression()
modelo.fit(X, y)

b0 = modelo.intercept_
b1 = modelo.coef_[0]
r2 = modelo.score(X, y)
mse = mean_squared_error(y, modelo.predict(X))
mae = mean_absolute_error(y, modelo.predict(X))
r  = np.sqrt(r2)

st.divider()

# Slider dinámico según el caso
valor_usuario = st.slider(
    datos["slider_label"],
    min_value=datos["slider_min"],
    max_value=datos["slider_max"],
    value=datos["slider_default"]
)

consumo_estimado = modelo.predict([[valor_usuario]])[0]
st.markdown(f"### 🔌 Consumo estimado: **{consumo_estimado:.2f} kWh**")
st.divider()

# ── GRÁFICO ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))

ax.scatter(datos["x"], datos["y"], color="steelblue", zorder=5, label="Datos reales")

x_linea = np.linspace(datos["slider_min"], datos["slider_max"], 200).reshape(-1, 1)
y_linea = modelo.predict(x_linea)
ax.plot(x_linea, y_linea, color="tomato", linewidth=2, label="Recta de regresión")

ax.scatter([valor_usuario], [consumo_estimado], color="green", s=130, zorder=6,
           label=f"Tu predicción ({valor_usuario} → {consumo_estimado:.1f} kWh)")

ax.set_xlabel(datos["x_label"])
ax.set_ylabel(datos["y_label"])
ax.set_title(f"Regresión Lineal — {caso_seleccionado}")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)

st.pyplot(fig)
st.divider()

# ── MÉTRICAS ──────────────────────────────────────────────────────────────────
st.subheader("📊 Métricas del modelo")

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("β₀", f"{b0:.2f}")
col2.metric("β₁", f"{b1:.2f}")
col3.metric("R", f"{r:.4f}")
col4.metric("R²", f"{r2:.4f}")
col5.metric("MSE", f"{mse:.2f}")
col6.metric("MAE", f"{mae:.2f}")

st.divider()
st.markdown(f"**Ecuación del modelo:** ŷ = {b0:.2f} + {b1:.2f}x")
st.caption("Trabajo Final — Matemática Discreta | Sistema de Regresión Lineal Simple")