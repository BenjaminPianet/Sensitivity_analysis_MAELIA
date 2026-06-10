# Pipeline web d'analyse de sensibilité MAELIA

Cette application transforme un répertoire de logs MAELIA/GAMA et le fichier `dataset_metamodel.csv` associé en analyses de sensibilité lisibles : ANOVA/Kruskal, interactions à deux facteurs, performances de métamodèles, indices de Sobol total et régions sensibles par arbres de décision.

Le point crucial est le suivant : les logs MAELIA contiennent les sorties du modèle, mais pas le plan de paramètres SMT. Le fichier `dataset_metamodel.csv` est donc indispensable, car il relie chaque simulation à ses 26 paramètres agronomiques.

## Workflow GAMA + application web

1. Dans le notebook de simulation, générer le plan SMT et les fichiers d'entrée MAELIA.
2. Lancer dans GAMA l'expérience correspondant au terrain choisi, par exemple `terrainSA`.
3. Conserver le dossier de logs généré, par exemple `simulations/log_terrainSA`.
4. Exporter ou copier `dataset_metamodel.csv` dans ce dossier de logs. Le notebook `batch_simulations_smt_terrainSA.ipynb` le fait normalement lors de l'export final.
5. Lancer l'application web et saisir le chemin du dossier de logs. Si le dataset n'est pas dans ce dossier, indiquer son chemin exact dans le champ optionnel.
6. Sélectionner les sorties à analyser puis lancer l'analyse.

## Lancer l'application web

```bash
cd /Users/benjamin/files/Repositories/Sensitivity_analysis_MAELIA
uvicorn maelia_sa_pipeline.api:app --reload --host 127.0.0.1 --port 8000
```

Interface utilisateur : http://127.0.0.1:8000/

Documentation API interactive : http://127.0.0.1:8000/docs

## Aide intégrée

L'interface contient maintenant un README intégré expliquant comment utiliser GAMA avec l'application et comment lire les mesures. Les champs, sorties, figures, métriques et 26 paramètres du plan SMT disposent aussi d'une aide contextuelle : survoler un terme pendant deux secondes affiche sa définition.

## Mesures affichées

- `R² entraînement` : part de variance expliquée sur les données d'entraînement. Il mesure l'ajustement, pas la généralisation.
- `Q² test` : R² calculé sur des simulations non vues pendant l'entraînement. C'est l'indicateur principal de généralisation.
- `ANOVA/Kruskal à un facteur` : R² descriptif associé aux groupes d'un paramètre.
- `ANOVA à deux facteurs` : matrice du R² d'interaction uniquement, donc le surplus explicatif propre au couple de paramètres.
- `Sobol total` : contribution globale d'un paramètre, interactions comprises, estimée via le métamodèle sur le domaine faisable.
- `Régions sensibles` : feuilles de l'arbre de décision résumées par leur moyenne et leur écart à la moyenne globale.
- `Arbre complet` : règles de seuil successives qui définissent les régimes locaux.

## Paramètres du plan SMT

Le plan actuel contient 26 paramètres : activations (`n_ferti`, `has_prepa`), préparation du sol (`nb_prepa`, `prepa_1`, `prepa_2`, `Delta_PREPA_Semis`), fertilisations (`nb_f1`, `type_f1_1`, etc.), dates (`Date_Semis`, `Date_F1`, `Date_F2`, `Date_F3`, `Date_Recolte`) et doses (`Dose_F1_1` à `Dose_F3_2`). Les dates sont exprimées en jours de campagne, avec `1 = 1er août`. Les variables dépendantes ne doivent être interprétées que si leur événement parent est actif.

## Ligne de commande sans serveur

```bash
python -m maelia_sa_pipeline.cli   --log-dir /Users/benjamin/files/Repositories/Sensitivity_analysis_MAELIA/simulations/log_terrainSA
```

## Sources méthodologiques

- Borgonovo et al. (2022), protocole d'analyse de sensibilité pour modèles agent-based.
- ten Broeke et al. (2016), choix des méthodes de sensibilité selon l'objectif.
- Thiele et al. (2014), guide pratique pour estimation de paramètres et analyse de sensibilité.
- Sobol' (2001) et Saltelli (2002), indices variance-based et indices totaux.
- Shapley (1953) et Song et al. (2016), effets de Shapley pour l'analyse de sensibilité globale.
- Breiman et al. (1984), arbres de classification et régression.
- Geurts et al. (2006), Chen et Guestrin (2016), Rasmussen et Williams (2006), métamodèles ExtraTrees, XGBoost et Gaussian Processes.
