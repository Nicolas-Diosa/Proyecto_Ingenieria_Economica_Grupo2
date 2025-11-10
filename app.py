import os
import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename

# --- Configuración de la Aplicación Flask ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey' # Necesario para los mensajes flash

# --- Lógica de Predicción (Adaptada de predict_pd.py) ---

def allowed_file(filename):
    """Verifica si la extensión del archivo es válida."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_prediction(file_path):
    """
    Procesa un archivo CSV y devuelve las predicciones de quiebra.
    """
    MODEL_FILE = 'bankruptcy_model.joblib'

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Archivo del modelo no encontrado: '{MODEL_FILE}'")

    model_payload = joblib.load(MODEL_FILE)
    logit_model = model_payload['model']
    feature_columns = model_payload['feature_columns']

    df = pd.read_csv(file_path)

    # --- Ingeniería de Características (Idéntica al entrenamiento) ---
    original_cols = ['X1', 'X6', 'X10', 'X14', 'X17', 'fyear']
    for col in original_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.sort_values(['company_name', 'fyear'], inplace=True)
    grouped = df.groupby('company_name')
    feature_df = pd.DataFrame()

    for name, group in grouped:
        temp_group = group.copy()
        temp_group['ROA'] = temp_group['X6'] / temp_group['X10']
        temp_group['Debt_Ratio'] = temp_group['X17'] / temp_group['X10']
        temp_group['X6_growth'] = temp_group['X6'].pct_change()
        temp_group['X1_growth'] = temp_group['X1'].pct_change()
        temp_group['debt_ratio_change'] = temp_group['Debt_Ratio'].diff()
        temp_group['X6_volatility_3y'] = temp_group['X6'].rolling(window=3).std()
        feature_df = pd.concat([feature_df, temp_group])

    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    predict_df = feature_df.dropna(subset=feature_columns)

    if predict_df.empty:
        return None

    X_pred = predict_df[feature_columns]
    X_pred_const = sm.add_constant(X_pred, has_constant='add')
    
    probabilities = logit_model.predict(X_pred_const)
    predict_df['prediction_probability'] = probabilities

    latest_predictions = predict_df.loc[predict_df.groupby('company_name')['fyear'].idxmax()].copy()
    
    # --- Asignar Diagnóstico y Color ---
    results = []
    for index, row in latest_predictions.iterrows():
        prob = row['prediction_probability']
        diagnosis = ''
        color_class = ''
        if prob > 0.5:
            diagnosis = 'ALTO RIESGO'
            color_class = 'danger' # Rojo de Bootstrap
        elif prob > 0.2:
            diagnosis = 'RIESGO MODERADO'
            color_class = 'warning' # Amarillo de Bootstrap
        else:
            diagnosis = 'BAJO RIESGO'
            color_class = 'success' # Verde de Bootstrap

        results.append({
            'company': row['company_name'],
            'year': int(row['fyear']),
            'probability': f"{prob*100:.2f}%",
            'diagnosis': diagnosis,
            'color_class': color_class
        })
    return results

# --- Rutas de la Aplicación Web ---

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No se encontró el campo del archivo.')
            return redirect(request.url)
        
        file = request.files['file']

        if file.filename == '':
            flash('Ningún archivo seleccionado.')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                predictions = get_prediction(filepath)
                if predictions is None:
                    flash('No se pudieron generar predicciones. Asegúrate de que el CSV tenga suficientes datos (al menos 3 años por empresa).')
                    return redirect(request.url)
                # Pasa los resultados a la plantilla de resultados
                return render_template('results.html', predictions=predictions)
            except Exception as e:
                flash(f'Ocurrió un error al procesar el archivo: {e}')
                return redirect(request.url)
        else:
            flash('Formato de archivo no permitido. Por favor, sube un archivo .csv')
            return redirect(request.url)

    return render_template('index.html')

if __name__ == '__main__':
    # Crear la carpeta de subidas si no existe
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
