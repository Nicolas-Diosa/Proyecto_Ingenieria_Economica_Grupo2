import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm
import os

# --- 1. Carga del Modelo y Datos de Entrada ---

MODEL_FILE = 'bankruptcy_model.joblib'
INPUT_CSV = 'datos_para_predecir.csv'

# Verificar que los archivos necesarios existan
if not os.path.exists(MODEL_FILE):
    print(f"Error: Archivo del modelo no encontrado: '{MODEL_FILE}'")
    print("Por favor, ejecuta primero 'logistic_regression_pd_model.py' para entrenar y guardar el modelo.")
    exit()

if not os.path.exists(INPUT_CSV):
    print(f"Error: Archivo de datos de entrada no encontrado: '{INPUT_CSV}'")
    exit()

# Cargar el modelo y las columnas de características
try:
    model_payload = joblib.load(MODEL_FILE)
    logit_model = model_payload['model']
    feature_columns = model_payload['feature_columns']
except Exception as e:
    print(f"Error al cargar el archivo del modelo: {e}")
    exit()

# Cargar el dataset de entrada que el usuario proporcionaría
try:
    df = pd.read_csv(INPUT_CSV)
except Exception as e:
    print(f"Error al leer el archivo CSV de entrada: {e}")
    exit()

print(f"Modelo '{MODEL_FILE}' y datos de '{INPUT_CSV}' cargados exitosamente.")
print("Iniciando predicción...")

# --- 2. Ingeniería de Características (Idéntica al entrenamiento) ---

# Asegurar que las columnas necesarias sean numéricas
# Usamos las columnas originales que dan lugar a las características del modelo
original_cols = ['X1', 'X6', 'X10', 'X14', 'X17', 'fyear']
for col in original_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Ordenar los datos por empresa y año
df.sort_values(['company_name', 'fyear'], inplace=True)

# Agrupar por empresa para calcular métricas de series de tiempo
grouped = df.groupby('company_name')
feature_df = pd.DataFrame()

for name, group in grouped:
    temp_group = group.copy()
    
    # Recrear las mismas características que en el entrenamiento
    temp_group['ROA'] = temp_group['X6'] / temp_group['X10']
    temp_group['Debt_Ratio'] = temp_group['X17'] / temp_group['X10']
    temp_group['X6_growth'] = temp_group['X6'].pct_change()
    temp_group['X1_growth'] = temp_group['X1'].pct_change()
    temp_group['debt_ratio_change'] = temp_group['Debt_Ratio'].diff()
    temp_group['X6_volatility_3y'] = temp_group['X6'].rolling(window=3).std()
    
    feature_df = pd.concat([feature_df, temp_group])

# --- 3. Preparación de Datos para Predicción ---

# Reemplazar infinitos y eliminar NaNs, igual que en el entrenamiento
feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
predict_df = feature_df.dropna(subset=feature_columns)

if predict_df.empty:
    print("\nError: No se pudieron calcular las características para la predicción.")
    print("Asegúrate de que el archivo CSV de entrada contenga suficientes datos históricos (al menos 3 años consecutivos por empresa).")
    exit()

# Seleccionar las columnas de características y añadir la constante
X_pred = predict_df[feature_columns]
X_pred_const = sm.add_constant(X_pred, has_constant='add')

# --- 4. Realizar y Mostrar la Predicción ---

# Usar el modelo cargado para predecir la probabilidad
# El resultado es la probabilidad de que la clase sea 1 (failed)
probabilities = logit_model.predict(X_pred_const)

# Añadir las probabilidades al DataFrame para una fácil interpretación
predict_df['prediction_probability'] = probabilities

print("\n" + "="*60)
print("         Resultados de la Predicción de Quiebra")
print("="*60)

# Mostrar la predicción para el año más reciente de cada empresa
# que tenga datos completos
latest_predictions = predict_df.loc[predict_df.groupby('company_name')['fyear'].idxmax()]

for index, row in latest_predictions.iterrows():
    company = row['company_name']
    year = int(row['fyear'])
    prob = row['prediction_probability']
    
    print(f"\nEmpresa: '{company}' (basado en datos hasta el año {year})")
    print(f"  -> Probabilidad de Quiebra (PD): {prob:.4f} ({prob*100:.2f}%)")
    
    if prob > 0.5:
        print("  -> Diagnóstico: ALTO RIESGO")
    elif prob > 0.2:
        print("  -> Diagnóstico: RIESGO MODERADO")
    else:
        print("  -> Diagnóstico: BAJO RIESGO")

print("\n" + "="*60)
print("Nota: La predicción se basa en el último año para el cual se pudieron calcular todas las métricas.")
