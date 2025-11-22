# Predictor de Quiebra Empresarial - Ingeniería Económica
<br>
<div align="center">
  <img src="https://www.pngkey.com/png/detail/268-2688228_universidad-nacional-colombia-logo.png" width="230" alt="Logo Universidad Nacional de Colombia">
</div>
<h2 align="center">UNIVERSIDAD NACIONAL DE COLOMBIA</h2> 
<p align="center">
  <strong>Ingeniería Económica - 2015703</strong>
</p>
<br><br>

### Autores: 

- Fabián David Mora Martínez (fmoram@unal.edu.co)
- Ever Nicolás Muñoz Cortés (evmunoz@unal.edu.co)
- Isaias David Gallardo Felizzola (igallardo@unal.edu.co)
- Ángel Manuel Cortavarria Salas (acortavarria@unal.edu.co)
- Nicolás Alejandro Diosa Benavides (ndiosab@unal.edu.co)
- Juan Esteban Ocampo Vidal (jocampov@unal.edu.co)
- Paula Alejandra Murcia Ramírez (pmurciar@unal.edu.co)

### Docente: 
Diego Alejandro Hernández Castañeda 
<br><br>

---

## Contenido

- [1. Objetivos](#1-objetivos)
  - [1.1 Objetivo General](#11-objetivo-general)
  - [1.2 Objetivos Específicos](#12-objetivos-específicos)
- [2. Principios de Ingeniería Económica Aplicados](#2-principios-de-ingeniería-económica-aplicados)
- [3. Estructura del Proyecto](#3-estructura-del-proyecto)
- [4. Requisitos de Instalación](#4-requisitos-de-instalación)
  - [4.1 Instalación con Docker (Recomendado)](#41-instalación-con-docker-recomendado)
  - [4.2 Instalación Manual sin Docker](#42-instalación-manual-sin-docker)
- [5. Ejecución del Proyecto](#5-ejecución-del-proyecto)
  - [5.1 Mediante Docker](#51-mediante-docker)
  - [5.2 Sin Docker](#52-sin-docker)
- [6. Instrucciones de Uso](#6-instrucciones-de-uso)
- [7. Tecnologías Utilizadas](#7-tecnologías-utilizadas)
- [8. Conclusiones](#8-conclusiones)
- [9. Referencias](#9-referencias)

<br><br>

---

## 1. Objetivos

### 1.1 Objetivo General

Desarrollar una aplicación web interactiva que, aplicando conceptos de ingeniería económica y un modelo de Machine Learning, permita predecir la probabilidad de quiebra de una empresa a partir de sus datos financieros históricos.

### 1.2 Objetivos Específicos

- Implementar un modelo de regresión logística en Python para clasificar a las empresas en categorías de riesgo (bajo, moderado, alto).
- Realizar una ingeniería de características (feature engineering) para transformar datos financieros crudos en variables significativas, como tasas de crecimiento, volatilidad de ingresos y ratios financieros, aplicando fórmulas vistas en el curso.
- Diseñar y construir una interfaz de usuario web intuitiva y estéticamente agradable utilizando Flask y Bootstrap, que facilite la carga de datos y la visualización de resultados.
- Empaquetar la aplicación y todas sus dependencias en contenedores Docker para garantizar un despliegue y ejecución sencillos y consistentes en cualquier entorno.
<br><br>

---

## 2. Principios de Ingeniería Económica Aplicados

Para construir un modelo predictivo robusto, se trascendió el análisis estático de un solo período y se incorporaron conceptos dinámicos que evalúan el desempeño de una empresa a través del tiempo.

- **Tasas de Crecimiento (Growth Rates)**: Se calcula la tasa de cambio porcentual anual de métricas clave como los ingresos netos (`X6`) y los activos corrientes (`X1`). Una tasa de crecimiento negativa o decreciente es un fuerte indicador de contracción y posible dificultad financiera.

- **Volatilidad como Medida de Riesgo**: Se utiliza la desviación estándar móvil (rolling standard deviation) de los ingresos netos. Una alta volatilidad sugiere inestabilidad y falta de previsibilidad en las ganancias, lo cual es una señal de riesgo elevado para inversores y acreedores.

- **Ratios Financieros Clave**: El modelo utiliza ratios fundamentales que son pilares en el análisis financiero:
  - **ROA (Return on Assets)**: Mide la rentabilidad de la empresa en relación con sus activos totales. Un ROA bajo o negativo indica un uso ineficiente de los activos.
  - **Debt Ratio (Ratio de Endeudamiento)**: Compara la deuda total con los activos totales. Un ratio elevado puede indicar un apalancamiento excesivo y un mayor riesgo de insolvencia.

- **Regresión Logística**: Aunque es una técnica estadística, su aplicación aquí está motivada económicamente. Se utiliza para modelar cómo un conjunto de variables financieras (los principios anteriores) influye en la probabilidad de un evento binario: la quiebra (`status_label` = 1) o la supervivencia (`status_label` = 0) de una empresa.
<br><br>

---

## 3. Estructura del Proyecto
```
.
├── Dockerfile                  # Define el entorno de la aplicación para Docker.
├── app.py                      # Lógica principal de la aplicación web (Flask).
├── bankruptcy_model.joblib     # Archivo del modelo de Machine Learning entrenado.
├── datos_para_predecir.csv     # Un archivo CSV de ejemplo para probar la aplicación.
├── american_bankruptcy_dataset.csv # El dataset completo usado para el entrenamiento.
├── docker-compose.yml          # Orquesta la construcción y ejecución del contenedor.
├── logistic_regression_pd_model.py # Script para entrenar y guardar el modelo de ML.
├── requirements.txt            # Lista de dependencias de Python.
├── README.md                   # Este archivo.
├── templates/
│   ├── index.html              # Página de inicio con el formulario de subida.
│   └── results.html            # Página que muestra los resultados de la predicción.
└── uploads/                    # Carpeta para almacenar los CSV subidos (se crea autom.).
```
<br><br>

---

## 4. Requisitos de Instalación

### 4.1 Instalación con Docker (Recomendado)
Este método es el más sencillo, ya que Docker gestiona automáticamente todas las dependencias y la configuración.

- **Docker Desktop**: Asegúrate de tenerlo instalado y en ejecución. Puedes descargarlo desde [este enlace](https://www.docker.com/products/docker-desktop/).

### 4.2 Instalación Manual sin Docker

- **Python (3.8 o superior)**: Puedes descargarlo desde [este enlace](https://www.python.org/downloads/). Asegúrate de marcar la casilla "Add Python to PATH" durante la instalación.
- **pip**: El gestor de paquetes de Python, usualmente viene incluido con la instalación de Python.
<br><br>

---

## 5. Ejecución del Proyecto

### 5.1 Mediante Docker

1.  **Construye y levanta el contenedor:**
    Abre una terminal en la raíz del proyecto y ejecuta:
    ```bash
    docker-compose up --build
    ```
    La primera vez puede tardar unos minutos mientras se construye la imagen.

2.  **Accede a la aplicación:**
    Abre tu navegador web y ve a: [http://localhost:5000](http://localhost:5000)

### 5.2 Sin Docker

1.  **Crea un entorno virtual (fuertemente recomendado):**
    ```bash
    # Crea el entorno
    python -m venv venv
    # Actívalo (en Windows)
    venv\Scripts\activate
    # Actívalo (en macOS/Linux)
    # source venv/bin/activate
    ```

2.  **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Ejecuta la aplicación Flask:**
    ```bash
    python app.py
    ```

4.  **Accede a la aplicación:**
    Abre tu navegador y ve a: [http://localhost:5000](http://localhost:5000)
<br><br>

---

## 6. Instrucciones de Uso

1.  Una vez que la aplicación esté corriendo, la página de inicio te mostrará un formulario para subir un archivo.

    <img src="https://i.imgur.com/your-upload-page-image.gif" alt="Inserción de Archivo">
    *(Reemplaza este enlace con una captura o GIF de tu página de inicio)*

2.  Haz clic en "Seleccionar archivo" y elige un archivo `.csv`. Puedes usar el `datos_para_predecir.csv` incluido en el proyecto para una prueba rápida. El archivo debe tener columnas como `company_name`, `fyear`, y las variables financieras `X1`, `X6`, etc.

3.  Presiona el botón **"Analizar y Predecir"**.

4.  La aplicación procesará los datos, los pasará por el modelo y te redirigirá a la página de resultados. Allí verás una tarjeta por cada empresa analizada, con su probabilidad de quiebra y un diagnóstico de riesgo claro y codificado por colores.

    <img src="https://i.imgur.com/your-results-page-image.gif" alt="Resultados del Análisis">
    *(Reemplaza este enlace con una captura o GIF de tu página de resultados)*

<br><br>

---

## 7. Tecnologías Utilizadas

#### Backend
- **Python**: Lenguaje principal de la aplicación.
- **Flask**: Micro-framework para construir la aplicación web.
- **Pandas & NumPy**: Para la manipulación y el análisis de datos.
- **Scikit-learn (Joblib) & Statsmodels**: Para la creación y gestión del modelo de regresión logística.

#### Frontend
- **HTML5**: Estructura de las páginas web.
- **CSS3**: Estilos personalizados y animaciones.
- **Bootstrap 5**: Framework de CSS para un diseño responsivo y moderno.
- **Bootstrap Icons**: Para la iconografía.
- **AOS (Animate on Scroll)**: Librería de JavaScript para animaciones al desplazar.

#### Despliegue
- **Docker & Docker Compose**: Para la containerización y orquestación de la aplicación.
<br><br>

---

## 8. Conclusiones

- Se desarrolló con éxito un modelo de Machine Learning capaz de predecir la quiebra empresarial, integrando de manera efectiva los principios de ingeniería económica en la selección y creación de variables predictivas.

- Se construyó una aplicación web funcional y estéticamente agradable que abstrae la complejidad del modelo, ofreciendo una herramienta de diagnóstico financiero accesible para usuarios no técnicos.

- Se garantizó la portabilidad y reproducibilidad del proyecto mediante la containerización con Docker, lo que simplifica enormemente su despliegue y elimina problemas de dependencias entre diferentes entornos de desarrollo.
<br><br>

---

## 9. Referencias

- **Texto Guía**: Blank, L., & Tarquin, A. (2012). *Ingeniería Económica* (7.a ed.). McGraw-Hill.
- **Dataset**: `futureinternet-14-00244-v2.pdf` - Documento que describe las variables del dataset original.

---
<div align="center">
    <img src="https://media.tenor.com/mG24i4G4qZkAAAAC/thumbs-up-computer.gif" width="300" alt="Todo listo">
</div>
<br>
