FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Créer un utilisateur non-root pour la sécurité
RUN useradd --create-home --shell /bin/bash app

# Copier les requirements et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY . .

# Créer les dossiers nécessaires avec les bonnes permissions
RUN mkdir -p uploads processed final archive logs database data && \
    chown -R app:app /app

# Changer vers l'utilisateur non-root
USER app

# Exposer le port
EXPOSE 5000

# Variables d'environnement par défaut
ENV FLASK_ENV=production
ENV DATABASE_URL=sqlite:///database/sage_x3.db
ENV PYTHONPATH=/app

# Commande de démarrage
CMD ["python", "-m", "gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "app:app"]

