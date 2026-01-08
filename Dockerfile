# Utilise une image Python officielle
FROM python:3.9-slim

# Definir le repertoire de travail
WORKDIR /app

# Copier les fichiers de dependances
COPY requirements.txt .

# Installer les dependances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application, les données et l'UI
COPY app/ ./app/
COPY model/ ./model/
COPY data/ ./data/
COPY streamlit_app.py .

# Exposer le port
EXPOSE 8000

# Commande pour demarrer l'application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]