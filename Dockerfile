# 1. Usamos Python 3.10 Slim
FROM python:3.10-slim

# 2. Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Instalar dependencias del sistema
# CAMBIO IMPORTANTE: Reemplazado 'libgl1-mesa-glx' por 'libgl1' y agregado 'libglib2.0-0'
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Directorio de trabajo
WORKDIR /app

# 5. Copiar requirements primero (para caché)
COPY requirements.txt .

# 6. Instalar dependencias de Python
# Usamos --extra-index-url para encontrar las versiones +cu118 de PyTorch
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

# 7. Copiar el código de la aplicación
# (El .dockerignore evitará copiar los venv y archivos pesados)
COPY . .

# 8. Exponer puerto
EXPOSE 8501

# 9. Ejecutar
CMD ["streamlit", "run", "apps/app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
