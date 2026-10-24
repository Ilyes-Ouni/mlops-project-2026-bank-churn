# FICHE DE CADRAGE : SYNOPSIS DU PROJET MLOPS (SESSION 2026–2027)

> Fichier déposé à la racine du dépôt pour le **Jalon 1**. Validation enseignant : `haythemghz@yahoo.fr`.

---

## 1. Informations Générales & Équipe

* **Titre du Projet :** Plateforme MLOps de rétention bancaire (Bank Customer Churn)
* **Taille du groupe :** 2 étudiants
* **Membres de l'équipe :**
  1. **Étudiant 1 (Lead Data & DVC) :** Ilyes El Ouni — Email : `ilyes.ouni700@gmail.com`
  2. **Étudiant 2 (Lead Modeling & MLflow) :** À renseigner — Email : `...`
  3. Serving/CI et Observabilité : partagés (binôme)
* **Lien vers le Dépôt GitHub :** `https://github.com/Ilyes-Ouni/mlops-project-2026-bank-churn`

---

## 2. Problématique Métier & Cas d'Usage

* **Contexte Business :** Banque de détail. Des clients deviennent dormants (retraits, solde sous le minimum) et le coût d'acquisition dépasse le coût de rétention.
* **Décision assistée :** Si `P(churn)` dépasse le seuil métier, déclencher une offre de rétention (appel conseiller / waiver de frais).
* **Objectif métier quantifié (KPI) :** Détecter au moins 70 % des churners (rappel) tout en gardant un taux de faux positifs < 15 % sur la population contactée.

---

## 3. Formulation Machine Learning & Choix du Dataset

* **Type de problème ML :** Classification binaire
* **Variable Cible (`Target`) :** `churn` — 1 = client devenu dormant / parti, 0 = actif
* **Source du Dataset :** https://www.kaggle.com/datasets/pentakrishnakishore/bank-customer-churn-data (Apache 2.0)
* **Volume et dimensions :** 28 382 lignes, 21 colonnes
* **Variables prédictives clés :**
  * *Numériques :* vintage, âge, soldes (courant / mois précédent / trimestres), crédits et débits mensuels
  * *Catégorielles :* genre, occupation, catégorie de patrimoine (`customer_nw_category`)
* **Métriques ML cibles :** F1 (classe 1) ≥ 0.55, ROC-AUC ≥ 0.80, latence d'inférence < 150 ms

---

## 4. Architecture MLOps Prévisionnelle (périmètre binôme)

| Composant MLOps | Outil & Configuration retenue |
|---|---|
| **Versioning Données** | DVC + remote local (`.dvc_remote`) + script de téléchargement |
| **Tracking & Registry** | MLflow Tracking local (`http://127.0.0.1:5000`) & Model Registry |
| **Microservice Inférence** | FastAPI `/predict`, `/health` + schémas Pydantic |
| **Packaging** | Docker multi-stage non-root + `docker-compose.yml` |
| **CI** | GitHub Actions : Ruff/Black/Flake8 + Pytest (couverture > 75 %) |
| **Rapports CML** | Métriques / courbes en commentaire de PR |
| **Orchestration** | Prefect 3.x : Tasks, Flow d'entraînement, Work Pool, CRON |
| **Surveillance de Dérive** | KS (numérique), χ² (catégoriel), PSI + Evidently HTML + DriftSimulator |
| **Hors scope binôme (stretch)** | Blue/Green-Canary, Prometheus `/metrics`, réentraînement autonome, Streamlit |

---

## 5. Scénario de Dérive de Données (Data Drift) Simulé

* **Variables qui dérivent :** hausse des débits / baisse des soldes, bascule d'occupation (plus de `self_employed`), dérive du mix `city`.
* **Méthode d'injection :** module `DriftSimulator` (décalage gaussien sur soldes/débits + permutation de catégories).
* **Seuil d'alerte :** PSI ≥ 0.20 sur une feature clé **ou** `drift_share` ≥ 30 %.
* **Comportement :** rapport Evidently HTML + alerte ; réentraînement Prefect automatique = stretch (non promis).

---

## 6. Planning Prévisionnel & Validation Enseignant

* [x] **Jalon 1 (S2) :** Dépôt GitHub, scaffold, dataset DVC, EDA (synopsis à valider)
* [ ] **Jalon 2 (S4) :** Pipeline DVC + MLflow + Pytest
* [ ] **Jalon 3 (S5) :** FastAPI + Docker + CI + CML
* [ ] **Jalon 4 (S6) :** Prefect 3.x + DriftSimulator
* [ ] **Jalon Final (S8) :** Tag `v1.0-release`, rapport PDF, soutenance

> **Avis de l'Enseignant :** `[  ] VALIDÉ  /  [  ] À REVOIR`  
> **Commentaires :**
