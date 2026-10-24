# FICHE DE CADRAGE : SYNOPSIS DU PROJET MLOPS (SESSION 2026–2027)

> **Instructions :** Remplissez ce document (1 à 2 pages max) et déposez-le à la racine de votre dépôt Git sous le nom `SYNOPSIS.md` avant la fin du **Jalon 1 (Semaine 2)** pour validation par l'enseignant (`haythemghz@yahoo.fr`).

---

## 1. Informations Générales & Équipe

* **Titre du Projet :** [Nom clair et professionnel du projet]
* **Taille du groupe :** [1 / 2 / 3 / 4 étudiants]
* **Membres de l'équipe :**
  1. **Étudiant 1 (Lead Data & DVC) :** Nom & Prénom — Email : `...`
  2. **Étudiant 2 (Lead Modeling & MLflow) :** Nom & Prénom — Email : `...`
  3. **Étudiant 3 (Lead Serving & CI/CD) :** Nom & Prénom — Email : `...` *(si applicable)*
  4. **Étudiant 4 (Lead Observabilité & SRE) :** Nom & Prénom — Email : `...` *(si applicable)*
* **Lien vers le Dépôt GitHub :** `https://github.com/mon-organisation/mon-projet-mlops`

---

## 2. Problématique Métier & Cas d'Usage

* **Contexte Business :** [Quel est le secteur d'activité et la problématique à résoudre ?]
* **Décision assistée ou automatisée :** [Quelle action concrète découle de la prédiction du modèle ?]
* **Objectif métier quantifié (KPI) :** [Ex: Détecter 95% des fraudes en réduisant le faux positif à moins de 2%]

---

## 3. Formulation Machine Learning & Choix du Dataset

* **Type de problème ML :** [Classification Binaire / Classification Multi-classes / Régression]
* **Variable Cible (`Target`) :** [Nom précis de la colonne cible et signification de ses valeurs]
* **Source du Dataset :** [Lien URL exact vers la source : Kaggle / UCI / OpenML / etc.]
* **Volume et dimensions :** [Ex : 25 000 lignes, 18 colonnes]
* **Variables prédictives clés :**
  * *Variables numériques :* [Ex: Montant, Âge, Revenu, Solde, Durée]
  * *Variables catégorielles :* [Ex: Type de contrat, Canal d'acquisition, Statut matrimonial]
* **Métriques ML cibles :** [Ex: F1-Score $\ge 0.85$, ROC-AUC $\ge 0.90$, Latence d'inférence $< 150$ ms]

---

## 4. Architecture MLOps Prévisionnelle

| Composant MLOps | Outil & Configuration retenue |
|---|---|
| **Versioning Données** | DVC avec remote storage local ou cloud (`dvc.yaml`) |
| **Tracking & Registry** | MLflow Tracking Server local (`http://127.0.0.1:5000`) & Model Registry |
| **Microservice Inférence** | API REST FastAPI avec documentation Swagger automatique |
| **Validation des données** | Schémas Pydantic stricts pour les payloads d'entrée |
| **Packaging & Isolation** | Image Docker multi-stage non-root & `docker-compose.yml` |
| **Intégration Continue (CI)** | GitHub Actions : Linting (`flake8`, `black`) + Pytest |
| **Rapports CML** | Publication automatique des courbes et métriques en commentaire de PR |
| **Orchestration** | Prefect 3.x (Tasks atomiques, Flow d'entraînement, Work Pool) |
| **Déploiement Continu** | Stratégie Zero-Downtime [Blue/Green OU Canary] |
| **Surveillance de Dérive** | Tests statistiques Kolmogorov-Smirnov, $\chi^2$ et calcul formel du PSI |
| **Export Métriques** | Format OpenMetrics Prometheus (`/metrics`) |
| **Boucle de Rétroaction** | Déclenchement automatique du réentraînement si `drift_share >= 0.30` |

---

## 5. Scénario de Dérive de Données (Data Drift) Simulé

* **Quelles variables vont dériver en production ?** [Ex: Hausse anormale des montants de transaction ou bascule des contrats vers du court terme]
* **Méthode d'injection de la dérive :** [Ex: Module `DriftSimulator` ajoutant un décalage gaussien et une permutation de catégories]
* **Seuil d'alerte :** [Ex: Valeur du PSI $\ge 0.20$ sur une feature clé ou proportion de drift global $\ge 30\%$]
* **Comportement autonome du système :** [Ex: Émission d'une alerte Prometheus et appel automatique du Flow Prefect de réentraînement avec mise à jour du Model Registry]

---

## 6. Planning Prévisionnel & Validation Enseignant

* [ ] **Jalon 1 (S2) :** Dépôt initialisé, Synopsis validé, Dataset importé sous DVC
* [ ] **Jalon 2 (S4) :** Pipeline DVC + MLflow fonctionnels, 1ers tests Pytest
* [ ] **Jalon 3 (S5) :** API FastAPI opérationnelle, Dockerfile validé, Workflow CI actif
* [ ] **Jalon 4 (S6) :** Flow Prefect orchestré, Détection de drift & Prometheus
* [ ] **Jalon Final (S8) :** Tag `v1.0-release`, Rapport PDF et Soutenance

> **Avis de l'Enseignant :** `[  ] VALIDÉ  /  [  ] À REVOIR`  
> **Commentaires :**
