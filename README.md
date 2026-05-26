# 🧪 DoE Analyzer — Application d'analyse de plans d'expériences

**Projet pédagogique** — Cours Plans d'expériences  
INSA Strasbourg — 2026

---

## Description

Application interactive développée avec Python et Streamlit pour analyser des plans d'expériences (DOE — Design of Experiments).

## Fonctionnalités

| Onglet | Contenu |
|--------|---------|
| 📊 Données | Import CSV/Excel, statistiques, corrélations |
| 📈 Modélisation | Régression linéaire, Surface de réponse (RSM), équations, R², RMSE |
| 🔍 Influence | Effets principaux, importance des variables par permutation |
| 📉 ANOVA | Analyse de variance, p-values, facteurs significatifs |
| 🎯 Optimisation | Multi-objectif, désirabilité, front de Pareto |

## Installation

```bash
# 1. Cloner ou télécharger le dossier doe_app/

# 2. Créer l'environnement virtuel
python -m venv .venv

# 3. Activer l'environnement
# Windows :
.\.venv\Scripts\Activate
# Mac/Linux :
source .venv/bin/activate

# 4. Installer les dépendances
pip install -r requirements.txt

# 5. Lancer l'application
streamlit run app.py
```

## Structure

```
doe_app/
├── app.py                  # Application principale
├── requirements.txt        # Dépendances Python
├── README.md               # Ce fichier
├── modules/
│   ├── data_import.py      # Import et exploration des données
│   ├── modeling.py         # Modélisation (linéaire, RSM, ANOVA)
│   ├── optimization.py     # Optimisation multi-objectif
│   └── utils.py            # Fonctions utilitaires
├── projets/                # Vos fichiers de données
└── .streamlit/
    └── config.toml         # Thème de l'application
```

## Format des données

L'application accepte :
- **CSV** : séparateurs `,` ou `;`
- **Excel** : `.xlsx` ou `.xls`

Vos données doivent contenir :
- Des colonnes **facteurs** (variables d'entrée contrôlées)
- Des colonnes **réponses** (variables mesurées)

## Bibliothèques utilisées

- `streamlit` — interface web interactive
- `pandas` / `numpy` — manipulation des données
- `scikit-learn` — régression linéaire et RSM
- `statsmodels` — ANOVA et statistiques avancées
- `plotly` — graphiques interactifs
- `scipy` — calculs scientifiques

## Méthodes d'optimisation

L'optimisation multi-objectif utilise la **méthode de la désirabilité** (Derringer & Suich) :
- Désirabilité individuelle pour chaque objectif (maximiser / minimiser / cibler)
- Désirabilité globale = moyenne géométrique pondérée
- Visualisation du **front de Pareto** pour les compromis bi-objectif
