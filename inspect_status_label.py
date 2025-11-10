import pandas as pd

# --- 1. Carga de Datos ---
try:
    df = pd.read_csv('american_bankruptcy_dataset.csv')
    print("Archivo 'american_bankruptcy_dataset.csv' cargado.\n")
except FileNotFoundError:
    print("Error: El archivo 'american_bankruptcy_dataset.csv' no se encontró.")
    exit()

# --- 2. Inspección de Columnas ---
print("="*50)
print("Nombres exactos de las columnas en el archivo CSV:")
print("="*50)
print(df.columns.tolist())
print("\n")

# --- 3. Inspección de la Columna 'status_label' ---
status_col_name = None
# Buscar el nombre correcto de la columna, ignorando mayúsculas/minúsculas
for col in df.columns:
    if col.lower() == 'status_label':
        status_col_name = col
        break

if status_col_name:
    print("="*50)
    print(f"Análisis de la columna '{status_col_name}':")
    print("="*50)
    
    # Mostrar los valores únicos que contiene
    unique_values = df[status_col_name].unique()
    print(f"Valores únicos encontrados: {unique_values}\n")
    
    # Contar cuántas veces aparece cada valor
    value_counts = df[status_col_name].value_counts(dropna=False)
    print("Conteo de cada valor (value_counts):")
    print(value_counts)
    
else:
    print("ERROR: No se encontró una columna llamada 'status_label' en el archivo.")
    print("Por favor, verifica el nombre de la columna en el CSV.")

print("\n" + "="*50)
