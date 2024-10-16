FROM python:3.12-bookworm
# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de requerimientos y la aplicación a la imagen
COPY requirements.txt requirements.txt
COPY appsettings.json appsettings.json

# Install system dependencies
RUN apt-get update && apt-get install -y \
    unixodbc-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install ODBC driver for SQL Server
RUN apt-get update && apt-get install -y gnupg2 \
    && curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17 \
    && rm -rf /var/lib/apt/lists/*
    
# Instala las dependencias de la aplicación
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código de la aplicación al directorio de trabajo
COPY . .

# Expone el puerto en el que corre la aplicación Flask
EXPOSE 5000

# Define el comando para ejecutar la aplicación
CMD ["python", "Api-RAG.py"]