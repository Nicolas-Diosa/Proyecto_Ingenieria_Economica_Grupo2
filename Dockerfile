# Usar una imagen base oficial de Python
FROM python:3.9-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo de dependencias al directorio de trabajo
COPY requirements.txt .

# Instalar las dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todos los archivos del proyecto al directorio de trabajo
COPY . .

# Exponer el puerto en el que corre la aplicación Flask
EXPOSE 5000

# Comando para correr la aplicación
# Usamos --host=0.0.0.0 para que sea accesible desde fuera del contenedor
CMD ["flask", "run", "--host=0.0.0.0"]
