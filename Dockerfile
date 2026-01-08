# Utilise une image Python officielle
FROM python:3.9-slim

# Definir le repertoire de travail
WORKDIR /app

# Copier les fichiers de dependances
COPY requirements.txt .

# Installer les dependances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application, les données, l'UI et le script d'entrée
COPY app/ ./app/
COPY model/ ./model/
COPY data/ ./data/
COPY streamlit_app.py .
COPY entrypoint.sh .

# Rendre le script exécutable
RUN chmod +x entrypoint.sh

# Exposer les ports (8000 pour API, 8501 pour UI)
EXPOSE 8000
EXPOSE 8501

# Utiliser le script d'entrée
ENTRYPOINT ["./entrypoint.sh"]