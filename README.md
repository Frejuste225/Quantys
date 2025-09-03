# 📊 Moulinette d'Inventaire Sage X3

**Automation des traitements d'inventaire pour Sage X3**  
*Solution complète pour la gestion des écarts d'inventaire en temps réel*

![Workflow](https://i.imgur.com/JyR4YjX.png)

---

## 🚀 Fonctionnalités Principales

| Fonctionnalité | Description | Technologies |
|----------------|------------|--------------|
| **Import Sage X3** | Traitement des fichiers CSV avec en-têtes E/L et données S | Pandas, OpenPyXL |
| **Gestion Multi-Inventaires** | Support des fichiers avec plusieurs lignes L (inventaires multiples) | Python, Pandas |
| **Types de Lots Avancés** | Reconnaissance de 3 types de numéros de lot avec priorités | RegEx, Python |
| **Calcul Automatique** | Détection des écarts entre stocks théoriques/réels | NumPy, Pandas |
| **Répartition Intelligente** | Distribution FIFO/LIFO avec priorité sur les types de lots | Python, Pandas |
| **Traçabilité Complète** | Conservation des quantités réelles saisies dans le fichier final | Python, Pandas |
| **API RESTful** | Interface moderne pour intégration | Flask, CORS |
| **Gestion de Sessions** | Suivi complet des opérations | Python, Logging |

---

## 🛠 Installation

### Prérequis

- Python 3.9+
- Pipenv (recommandé)

```bash
# Cloner le dépôt
git clone https://github.com/votre-repo/moulinette-sage.git
cd moulinette-sage/backend

# Installer les dépendances
pipenv install
pipenv shell

# Lancer le serveur
python app.py

## Variables d'Environnement

```ini
    .env.example:
    UPLOAD_FOLDER=uploads
    MAX_FILE_SIZE=16777216  # 16MB
    LOG_LEVEL=INFO
```

## 📚 Utilisation

### Structure du Fichier Final

Le fichier CSV final généré contient les quantités réelles saisies dans la **colonne G** (`QUANTITE_REELLE_IN_INPUT`), permettant une traçabilité complète :

- **Colonne F** : Quantité théorique ajustée (après calcul des écarts)
- **Colonne G** : Quantité réelle saisie lors de l'inventaire (**NOUVELLE FONCTIONNALITÉ**)
- **Colonne H** : Indicateur de compte (1=normal, 2=ajusté)

```csv
S;SESSION;INV001;1000;SITE01;95;95;2;ART001;EMP001;A;UN;0;ZONE1;LOT001
#                        ↑  ↑  ↑
#                        F  G  H
#                     Théo Réel Ind
```

### Types de Numéros de Lot Supportés

L'application reconnaît et traite les numéros de lot avec ordre de priorité :

1. **Type 1 (Priorité Haute)** : `CPKU070725xxxx`, `CB2TV020425xxxx`
   - Format : `[SITE][DDMMYY][NUMERO]`
   - Extraction automatique de la date pour tri FIFO/LIFO

2. **Type 2 (Priorité Moyenne)** : `LOT311224`
   - Format : `LOT[DDMMYY]`
   - Extraction de la date pour tri chronologique

3. **LOTECART (Cas Spécial)** : Détecté quand quantité théorique = 0
   - Détecté automatiquement quand quantité théorique = 0 ET quantité réelle > 0
   - Pas de tri par date, premier lot disponible

### Gestion des Inventaires Multiples

Support des fichiers avec plusieurs lignes L :
```csv
E;BKE022508SES00000003;test depot conf;1;BKE02;;;;;;;;;;
L;BKE022508SES00000003;BKE022508INV00000006;1;BKE02;;;;;;;;;;
L;BKE022508SES00000003;BKE022508INV00000007;1;BKE02;;;;;;;;;;
S;BKE022508SES00000003;BKE022508INV00000006;1000;BKE02;...
S;BKE022508SES00000003;BKE022508INV00000007;2000;BKE02;...
```

sequenceDiagram
    Utilisateur->>Backend: 1. Upload fichier CSV
    Backend->>Utilisateur: Template Excel
    Utilisateur->>Backend: 2. Fichier complété
    Backend->>Utilisateur: Fichier corrigé

## Endpoints API

| Méthode |             Endpoint          |	Description |
:---------:-------------------------------:--------------:
|  POST	  |          /api/upload          |	Import fichier Sage X3
|  POST	  |          /api/process         |	Traitement fichier complété
|  GET	  |  /api/download/<type>/<id>	  | Téléchargement fichiers
|  GET	  |        /api/sessions          |Liste des sessions

Exemple de requête :

```bash
curl -X POST -F "file=@inventaire.csv" http://localhost:5000/api/upload
```

## 🧩 Structure du Code

```txt
backend/
├── app.py               # Point d'entrée
├── processor.py         # Cœur métier
├── config.py            # Configuration
├── requirements.txt     # Dépendances
└── data/                # Stockage
    ├── uploads/         # Fichiers bruts
    ├── processed/       # Templates
    └── final/           # Résultats
```

## 🛡 Sécurité

- Validation stricte des fichiers entrants
- Limitation de taille des fichiers (16MB)
- Journalisation complète des opérations
- Gestion des erreurs détaillée

```python
# Exemple de validation
def validate_file(file):
    if not file.mimetype in ['text/csv', 'application/vnd.ms-excel']:
        raise InvalidFileTypeError
```

---

## 📈 Performances

|   Taille  | Fichier   |  Temps Moyen  |    Mémoire Utilisée   |
:-----------:-----------:---------------:-----------------------:
|   1,000   | lignes    |      1.2s     |       ~50MB           |
|   10,000  | lignes    |      4.5s     |       ~120MB          |
|   50,000  | lignes    |     12.8s     |       ~450MB          |

---

## 🤝 Contribution

1. Forker le projet
2. Créer une branche (git checkout -b feature/amelioration)
3. Commiter vos changements (git commit -m 'Nouvelle fonctionnalité')
4. Pousser vers la branche (git push origin feature/amelioration)
5. Ouvrir une Pull Request

Bonnes pratiques :

- Respecter PEP 8
- Documenter les nouvelles fonctions
- Ajouter des tests unitaires

## 📜 Licence

[MIT](https://opensource.org/licenses/MIT) - Copyright © 2023 [Kei Prince Frejuste]

---

<div align="center"> <img src="https://i.imgur.com/5Xw5r3a.png" width="200"> <p><em>Logo du Projet</em></p> </div>

