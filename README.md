# Analyse de sensibilité MAELIA

Ce dépôt regroupe les notebooks, scripts, figures et une app web utilisés pour analyser la sensibilité des sorties MAELIA. La démarche suit trois étapes :

1. analyser les premiers résultats lorsque le sol et le climat varient ;
2. isoler les opérations techniques avec `terrainSA` et entraîner un métamodèle ;
3. identifier des régions sensibles locales avec des arbres de décision.

Les figures terrainSA les plus récentes proviennent du run web `analysis/web_runs/20260602_170718_66253e83`, calculé sur `10000` points du plan SMT.

## Partie 1 - Premières analyses : sol et climat variables

Les premières analyses sont réalisées sur un terrain où le climat et le type de sol changent entre parcelles. Dans ce cadre, les sorties sont fortement structurées par le contexte pédoclimatique. Les paramètres techniques existent bien dans le signal, mais leur contribution est largement masquée par les contrastes entre zones météo et types de sol.

Ce résultat est central pour l'interprétation : lorsque le sol et le climat varient, l'analyse de sensibilité répond d'abord à une question spatiale, pas seulement agronomique. Elle montre quelles zones et quels sols expliquent les variations, mais ne permet pas encore d'isoler proprement les effets des itinéraires techniques.

![ANOVA terrainTest](figs/ANOVA.png)

![PRCC terrainTest](figs/PRCC.png)

![Sobol S1 terrainTest](figs/SOBOL_S1.png)

Les figures par groupes sol-climat confirment cette lecture : les distributions des sorties sont nettement séparées par les contextes environnementaux.

![Rendement selon sol et climat](figs/rdt_sol_climat.png)

![Lixiviation azotée selon sol et climat](figs/Nlixi_sol_climat.png)

![Carbone organique selon sol et climat](figs/Corg_sol_climat.png)

L'entraînement d'un métamodèle XGBoost n'a pas donné de résultats suffisamment satisfaisants sur ces données pédoclimatiques variables. D'où la construction de `terrainSA`.

## Partie 2 - terrainSA : opérations techniques et métamodèle

Pour isoler les leviers techniques, `terrainSA` clone la parcelle `beauce_5_1`. Les simulations comparent alors des itinéraires techniques dans un contexte constant : même sol, même géométrie et même zone météo. Cette construction réduit le bruit lié au milieu et rend les effets agronomiques plus lisibles.

Le plan SMT actuel encode les dates en jours de campagne, avec `1 = 1er août`. Les principaux paramètres calendaires sont donc `Date_Semis`, `Delta_PREPA_Semis`, `Date_F1`, `Date_F2`, `Date_F3` et `Date_Recolte`.

Le métamodèle utilisé par l'app web généralise bien sur le jeu de test :

| Sortie | R2 entraînement | Q2 test | MAE test |
|---|---:|---:|---:|
| `N_lixi` | 0.929 | 0.923 | 0.085 |
| `dCorg` | 0.994 | 0.993 | 1.968 |
| `rdt` | 0.919 | 0.910 | 0.029 |

![Performance N_lixi](analysis/web_runs/20260602_170718_66253e83/N_lixi/metamodel_performance_N_lixi.png)

![Performance dCorg](analysis/web_runs/20260602_170718_66253e83/dCorg/metamodel_performance_dCorg.png)

![Performance rdt](analysis/web_runs/20260602_170718_66253e83/rdt/metamodel_performance_rdt.png)

### ANOVA à un facteur

L'ANOVA/Kruskal à un facteur met en évidence des leviers différents selon les sorties :

| Sortie | Paramètres dominants | Lecture rapide |
|---|---|---|
| `N_lixi` | `Date_Semis`, `Delta_PREPA_Semis`, `has_prepa`, `prepa_1`, `Date_Recolte` | la lixiviation est d'abord gouvernée par le calendrier semis-préparation-récolte ; |
| `dCorg` | `Date_Recolte`, `Date_Semis` | le carbone organique dépend surtout de la durée et du positionnement du cycle ; |
| `rdt` | `Delta_PREPA_Semis`, `has_prepa`, `prepa_1`, `Date_Recolte`, `Date_Semis` | le rendement est sensible au délai préparation-semis et au calendrier de fin de cycle. |

![ANOVA N_lixi](analysis/web_runs/20260602_170718_66253e83/N_lixi/anova_1facteur_N_lixi.png)

![ANOVA dCorg](analysis/web_runs/20260602_170718_66253e83/dCorg/anova_1facteur_dCorg.png)

![ANOVA rdt](analysis/web_runs/20260602_170718_66253e83/rdt/anova_1facteur_rdt.png)

### Interactions à deux facteurs

Les heatmaps affichent uniquement le `R2_interaction`. Les interactions les plus lisibles concernent surtout les couples qui combinent préparation du sol, délai préparation-semis, date de semis et date de récolte. Pour `N_lixi`, le couple `has_prepa x Delta_PREPA_Semis` ressort nettement. Pour `dCorg`, l'interaction reste dominée par le couple `Date_Semis x Date_Recolte`. Pour `rdt`, les interactions confirment que le délai de préparation module les effets du calendrier.

![Interactions ANOVA N_lixi](analysis/web_runs/20260602_170718_66253e83/N_lixi/anova_2facteurs_R2_interaction_N_lixi.png)

![Interactions ANOVA dCorg](analysis/web_runs/20260602_170718_66253e83/dCorg/anova_2facteurs_R2_interaction_dCorg.png)

![Interactions ANOVA rdt](analysis/web_runs/20260602_170718_66253e83/rdt/anova_2facteurs_R2_interaction_rdt.png)

### Sobol total

Les indices de Sobol d'ordre total, estimés via le métamodèle, confirment les mêmes tendances globales :

| Sortie | Principaux indices Sobol total |
|---|---|
| `N_lixi` | `Date_Semis` ≈ 0.675, `has_prepa` ≈ 0.175, `Date_Recolte` ≈ 0.109, `Delta_PREPA_Semis` ≈ 0.078 |
| `dCorg` | `Date_Recolte` ≈ 0.718, `Date_Semis` ≈ 0.291 |
| `rdt` | `has_prepa` ≈ 0.301, `Date_Semis` ≈ 0.297, `Date_Recolte` ≈ 0.289, `Delta_PREPA_Semis` ≈ 0.140 |

![Sobol total N_lixi](analysis/web_runs/20260602_170718_66253e83/N_lixi/sobol_total_N_lixi.png)

![Sobol total dCorg](analysis/web_runs/20260602_170718_66253e83/dCorg/sobol_total_dCorg.png)

![Sobol total rdt](analysis/web_runs/20260602_170718_66253e83/rdt/sobol_total_rdt.png)

## Partie 3 - Régions sensibles par arbres de décision

Les arbres de décision cherchent des seuils et des régions locales compréhensibles. Ils ne remplacent pas le métamodèle global ; ils servent à formuler des règles du type : dans telle zone du plan, la réponse change de régime.

Les résultats récents montrent une lecture plus cohérente avec le nouveau plan de campagne : les seuils portent principalement sur `Date_Semis`, `Date_Recolte`, `Delta_PREPA_Semis` et `has_prepa`.

### Régions locales principales

Pour `N_lixi`, les régions les plus contrastées sont structurées par `Date_Semis`, `Date_Recolte` et `Delta_PREPA_Semis`. Les faibles lixiviations apparaissent notamment pour des semis plus précoces et des récoltes non trop tardives, tandis que certaines combinaisons de semis tardifs et de préparation moins anticipée augmentent la lixiviation moyenne.

Pour `dCorg`, la séparation est très nette autour du couple `Date_Recolte` / `Date_Semis`. Les récoltes tardives combinées à certains semis précoces conduisent aux pertes de carbone les plus fortes, tandis que des cycles plus courts ou mieux positionnés réduisent ces pertes.

Pour `rdt`, les régions sensibles articulent surtout `Delta_PREPA_Semis`, `Date_Recolte`, `Date_Semis` et `has_prepa`. Les meilleures régions combinent une préparation suffisamment anticipée, une récolte tardive et un calendrier de semis favorable.

![Régions sensibles N_lixi](analysis/web_runs/20260602_170718_66253e83/N_lixi/decision_tree_regions_N_lixi.png)

![Régions sensibles dCorg](analysis/web_runs/20260602_170718_66253e83/dCorg/decision_tree_regions_dCorg.png)

![Régions sensibles rdt](analysis/web_runs/20260602_170718_66253e83/rdt/decision_tree_regions_rdt.png)

Les arbres complets restent disponibles pour inspecter les règles exactes :

![Decision tree N_lixi](analysis/web_runs/20260602_170718_66253e83/N_lixi/decision_tree_N_lixi.png)

![Decision tree dCorg](analysis/web_runs/20260602_170718_66253e83/dCorg/decision_tree_dCorg.png)

![Decision tree rdt](analysis/web_runs/20260602_170718_66253e83/rdt/decision_tree_rdt.png)

## App web

L'app web permet de lancer la même analyse depuis une interface locale :

```bash
/Users/benjamin/.pyenv/versions/MAELIA_SA/bin/python -m uvicorn maelia_sa_pipeline.api:app --host 127.0.0.1 --port 8000
```

Puis ouvrir : `http://127.0.0.1:8000`.

L'app attend un dossier de logs et un `dataset_metamodel.csv` compatible avec le plan SMT courant. Elle refuse maintenant les datasets issus de l'ancien plan, notamment lorsque `feat_14` ne correspond pas à `Date_Semis` dans l'intervalle attendu `[45, 106]`.

## Fichiers utiles

- Analyse terrainSA et métamodèles : `analysis/Analyse_terrainSA.ipynb`
- Analyse des seuils historique : `analysis/Analyse_seuils_decision_tree.ipynb`
- Dernier run web terrainSA : `analysis/web_runs/20260602_170718_66253e83/`
- Résultats terrainSA historiques : `analysis/terrainSA_results/`
- Notebook de lancement terrainSA : `simulations/batch_simulations_smt_terrainSA.ipynb`
- Figures historiques terrainTest : `figs/`
