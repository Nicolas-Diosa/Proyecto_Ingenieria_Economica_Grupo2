
import pandas as pd
import statsmodels.api as sm
import numpy as np
import joblib

# --- 1. Carga y Preparación de Datos ---

# Cargar el dataset
try:
    df = pd.read_csv('american_bankruptcy_dataset.csv')
except FileNotFoundError:
    print("Error: El archivo 'american_bankruptcy_dataset.csv' no se encontró.")
    exit()

# Mapear la variable objetivo a valores numéricos
df['status_label'] = df['status_label'].map({'alive': 0, 'failed': 1})

# Columnas financieras clave para el análisis
cols_to_process = ['X1', 'X6', 'X10', 'X14', 'X17', 'fyear', 'status_label']

# Asegurar que todas las columnas a procesar sean numéricas
for col in cols_to_process:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Eliminar filas donde las columnas clave o la variable objetivo son nulas
df.dropna(subset=cols_to_process, inplace=True)

# Ordenar los datos por empresa y año, es crucial para cálculos de series de tiempo
df.sort_values(['company_name', 'fyear'], inplace=True)

print("Datos cargados y limpios. Iniciando ingeniería de características...")

# --- 2. Ingeniería de Características Avanzada (Fórmulas Económicas) ---

# Agrupar por empresa para calcular métricas a lo largo del tiempo
grouped = df.groupby('company_name')

# Crear un nuevo DataFrame para las características
feature_df = pd.DataFrame()

# Para cada empresa, calcular las nuevas variables
for name, group in grouped:
    temp_group = group.copy()
    
    # a) Ratios Financieros Básicos (como en el script original)
    temp_group['ROA'] = temp_group['X6'] / temp_group['X10']
    temp_group['Debt_Ratio'] = temp_group['X17'] / temp_group['X10']
    
    # b) Tasas de Crecimiento (YoY - Year over Year)
    # Usamos pct_change() que calcula (actual - previo) / previo
    temp_group['X6_growth'] = temp_group['X6'].pct_change() # Crecimiento del Ingreso Neto
    temp_group['X1_growth'] = temp_group['X1'].pct_change() # Crecimiento de Activos Corrientes
    
    # c) Cambio en Ratios
    temp_group['debt_ratio_change'] = temp_group['Debt_Ratio'].diff() # Cambio absoluto en el ratio de deuda
    
    # d) Volatilidad (como señal de riesgo/inestabilidad)
    # Usamos la desviación estándar móvil de los últimos 3 años del Ingreso Neto
    temp_group['X6_volatility_3y'] = temp_group['X6'].rolling(window=3).std()
    
    feature_df = pd.concat([feature_df, temp_group])

print("Ingeniería de características completada.")

# --- 3. Limpieza Final y Preparación para el Modelo ---

# Las operaciones de series de tiempo (pct_change, rolling) crean NaNs en las primeras filas de cada grupo.
# Debemos eliminarlos para tener un dataset limpio para el modelo.
final_df = feature_df.dropna()

# Reemplazar valores infinitos que puedan surgir de divisiones por cero
final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
final_df.dropna(inplace=True)

if final_df.empty:
    print("Error: El proceso de ingeniería de características no produjo datos válidos.")
    print("Esto puede ocurrir si no hay suficientes datos históricos por empresa (se necesitan al menos 3 años por empresa para la volatilidad).")
    exit()

# --- 4. Definición y Entrenamiento del Modelo de Regresión Logística ---

# Definir la variable dependiente (Y)
y = final_df['status_label']

# Definir las variables independientes (X)
# Ahora usamos nuestro conjunto de características mucho más rico
feature_columns = [
    'ROA', 
    'Debt_Ratio', 
    'X6_growth', 
    'X1_growth', 
    'debt_ratio_change', 
    'X6_volatility_3y'
]
X = final_df[feature_columns]

# statsmodels requiere que añadamos explícitamente una constante (el intercepto β₀)
X_const = sm.add_constant(X)

# Crear y entrenar el modelo Logit
logit_model = sm.Logit(y, X_const)
result = logit_model.fit()

# --- 5. Presentación de Resultados ---

print("="*80)
print("      Resultados del Modelo Avanzado de Predicción de Incumplimiento")
print("="*80)
print(result.summary())
print("\n")
print("="*80)
print("Variables utilizadas en el modelo:")
for col in feature_columns:
    print(f"- {col}")
print("="*80)

# --- 6. Guardar el Modelo y las Columnas para la Predicción ---

# Guardamos el modelo entrenado, la lista de columnas necesarias y la constante
model_payload = {
    'model': result,
    'feature_columns': feature_columns
}

joblib.dump(model_payload, 'bankruptcy_model.joblib')

print("\nModelo entrenado y guardado exitosamente en 'bankruptcy_model.joblib'")
print("Este archivo contiene todo lo necesario para hacer futuras predicciones.")
