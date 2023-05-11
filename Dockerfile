# Utiliser une image de base Python 3.9
FROM python:3.9-slim-buster

# Créer un répertoire de travail
WORKDIR /app

# Copier le fichier requirements.txt dans le répertoire de travail
COPY requirements.txt .

# Installer les dépendances de build et nettoyer après l'installation
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential libgomp1 && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get remove -y gcc build-essential && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Copier le reste des fichiers de l'application dans le répertoire de travail
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
