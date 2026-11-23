
# mlops-ml-project — baseline ML reproductible

## Description

Projet de classification Machine Learning utilisant
le dataset Iris et la régression logistique.

## Installation

Créer un environnement virtuel :

    python -m venv .venv

Activer sous Windows CMD :

    .venv\Scripts\activate.bat

Installer les dépendances :

    pip install -r requirements.txt

## Entraînement

    python scripts/train.py

## Évaluation

    python scripts/evaluate.py

## Tests

    python -m pytest -q

## Artefacts

Les fichiers générés dans artifacts/ :

- model.joblib : pipeline entraîné
- metrics.json : accuracy et f1_macro
- confusion_matrix.png : matrice de confusion
- report.json : rapport de classification détaillé

## Configuration

La configuration du projet est disponible dans :

    config/train.yaml
