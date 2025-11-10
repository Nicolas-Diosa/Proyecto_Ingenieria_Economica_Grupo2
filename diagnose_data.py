
import pandas as pd
import numpy as np

# --- 1. Carga de Datos ---
try:
    df = pd.read_csv('american_bankruptcy_dataset.csv')
    print("Archivo 'american_bankruptcy_dataset.csv' cargado exitosamente.")
    print(f"El dataset tiene {df.shape[0]} filas y {df.shape[1]} columnas.\n")
except FileNotFoundError:
    print("Error: El archivo 'american_bankruptcy_dataset.csv' no se encontró.")
    exit()

# --- 2. Diagnóstico de Columnas Requeridas ---
cols_to_diagnose = ['X1', 'X6', 'X10', 'X14', 'X17', 'fyear', 'status_label']

print("="*60)
print("        Análisis de Calidad de Datos por Columna")
print("="*60)
print("Analizando el porcentaje de valores no-numéricos o faltantes...\n")

results = {}

for col in cols_to_diagnose:
    if col not in df.columns:
        print(f"ADVERTENCIA: La columna '{col}' no existe en el archivo CSV.")
        continue
    
    # Contar valores nulos iniciales
    initial_nulls = df[col].isnull().sum()
    
    # Forzar conversión a número. Los errores se convierten en NaN.
    numeric_col = pd.to_numeric(df[col], errors='coerce')
    
    # Contar valores nulos después de la conversión
    final_nulls = numeric_col.isnull().sum()
    
    # El total de valores problemáticos son los nuevos NaNs creados
    problematic_values = final_nulls - initial_nulls
    
    # Porcentaje total de valores inutilizables
    total_unusable_percent = (final_nulls / len(df)) * 100
    
    results[col] = {
        "Total Rows": len(df),
        "Missing or Non-Numeric": final_nulls,
        "% Unusable": f"{total_unusable_percent:.2f}%"
    }

# Imprimir resultados en una tabla
if results:
    print(f"{ 'Columna':<15} | { 'Filas Totales':<15} | { 'Valores Faltantes':<20} | { '% Inutilizable':<15}")
    print("-" * 80)
    for col, data in results.items():
        print(f"{col:<15} | {data['Total Rows']:<15} | {data['Missing or Non-Numeric']:<20} | {data['% Unusable']:<15}")

print("\n" + "="*60)
print("Este reporte muestra cuántos valores en cada columna son nulos o no pudieron ser convertidos a números.")
print("Una columna con un alto porcentaje '%' inutilizable es la causa probable del error.")
