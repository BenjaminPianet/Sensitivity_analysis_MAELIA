# Analyse de sensibilité MAELIA

Ce dépôt regroupe les notebooks, scripts et figures utilisés pour analyser la sensibilité des sorties MAELIA. La démarche suit quatre étapes :

1. analyser les premiers résultats lorsque le sol et le climat varient ;
2. isoler les opérations techniques avec `terrainSA` et entraîner un métamodèle ;
3. identifier des seuils locaux avec des arbres de décision ;
4. visualiser les effets moyens et individuels des principaux paramètres temporels par courbes PDP/ICE.

Les figures présentées ci-dessous sont celles générées par les notebooks et présentes dans le dépôt GitHub.

## Organisation rapide

- `analysis/terrainSA_results/` : résultats globaux terrainSA, ANOVA, Sobol, Shapley et comparaison des métamodèles.
- `analysis/decision_tree_thresholds/` : arbres de décision, règles de seuil et régimes locaux.
- `analysis/pdp_ice_dynamic/` : dataset dynamique final, trajectoires temporelles et courbes PDP/ICE.
- `figs/` : figures historiques issues des premières analyses avec sol et climat variables.
- `simulations/` : notebooks de génération du plan SMT et de lancement des expériences GAMA.
- `maelia_sa_pipeline/` : application web locale pour relancer la pipeline d'analyse.

## Partie 1 - Premières analyses : sol et climat variables

Les premières analyses sont réalisées sur un terrain où le climat et le type de sol changent entre parcelles. Dans ce cadre, les sorties sont fortement structurées par le contexte pédoclimatique. Les paramètres techniques existent bien dans le signal, mais leur contribution est largement masquée par les contrastes entre zones météo et types de sol.

Ce résultat est central pour l'interprétation : lorsque le sol et le climat varient, l'analyse de sensibilité répond d'abord à une question spatiale, pas seulement agronomique. Elle montre quelles zones et quels sols expliquent les variations, mais ne permet pas encore d'isoler proprement les effets des itinéraires techniques.

![ANOVA terrainTest](figs/ANOVA.png)

![PRCC terrainTest](figs/PRCC.png)

![Sobol S1 terrainTest](figs/SOBOL_S1.png)

Les distributions par groupes sol-climat confirment cette lecture : les sorties sont nettement séparées par les contextes environnementaux.

![Rendement selon sol et climat](figs/rdt_sol_climat.png)

![Lixiviation azotée selon sol et climat](figs/Nlixi_sol_climat.png)

![Carbone organique selon sol et climat](figs/Corg_sol_climat.png)

L'entraînement d'un métamodèle XGBoost n'a pas donné de résultats suffisamment satisfaisants sur ces données pédoclimatiques variables. D'où la construction de `terrainSA`.

## Partie 2 - terrainSA : opérations techniques et métamodèle

Pour isoler les leviers techniques, `terrainSA` clone la parcelle `beauce_5_1`. Les simulations comparent alors des itinéraires techniques dans un contexte constant : même sol, même géométrie et même zone météo. Cette construction réduit le bruit lié au milieu et rend les effets agronomiques plus lisibles.

Le plan SMT actuel encode les dates en jours de campagne, avec `1 = 1er août`. Les principaux paramètres calendaires sont `Date_Semis`, `Delta_PREPA_Semis`, `Date_F1`, `Date_F2`, `Date_F3` et `Date_Recolte`.

### Métamodèle

Le notebook compare ExtraTrees, XGBoost et Gaussian Process. Les meilleurs modèles retenus sont :

| Sortie | Métamodèle retenu | Q2 test | R2 entraînement effectif |
|---|---|---:|---:|
| `N_lixi` | ExtraTrees | 0.987 | 0.999 |
| `dCorg` | XGBoost | 0.999 | 0.999 |
| `rdt` | ExtraTrees | 0.985 | 0.999 |

Ces scores indiquent que le métamodèle est suffisamment fidèle pour soutenir les analyses globales sur `terrainSA`. Les performances très élevées sont cohérentes avec le fait que le terrain est volontairement homogénéisé : le signal provient principalement des opérations techniques et de leur calendrier.

### ANOVA à un facteur

L'ANOVA/Kruskal à un facteur met en évidence des leviers différents selon les sorties :

| Sortie | Paramètres dominants | Lecture rapide |
|---|---|---|
| `N_lixi` | `Date_Semis`, `Delta_PREPA_Semis`, `has_prepa`, `prepa_1`, `Date_Recolte` | la lixiviation est d'abord gouvernée par le calendrier semis-préparation-récolte ; |
| `dCorg` | `Date_Recolte`, `Date_Semis` | le carbone organique dépend surtout de la durée et du positionnement du cycle ; |
| `rdt` | `Delta_PREPA_Semis`, `has_prepa`, `prepa_1`, `Date_Recolte`, `Date_Semis` | le rendement est sensible au délai préparation-semis et au calendrier de fin de cycle. |

![ANOVA terrainSA à un facteur](analysis/terrainSA_results/anova_1facteur_top.png)

### Interactions à deux facteurs

Les heatmaps ci-dessous affichent uniquement le `R2_interaction`. Les interactions sont plus faibles que les effets principaux, mais elles précisent les zones où un paramètre dépend du niveau d'un autre. Pour `N_lixi`, les interactions concernent surtout la préparation du sol et le délai préparation-semis. Pour `dCorg` et `rdt`, les couples autour de `Date_Semis`, `Date_Recolte` et `Delta_PREPA_Semis` sont les plus interprétables.

![Interactions ANOVA N_lixi](analysis/terrainSA_results/heatmap_anova_2facteurs_R2_interaction_N_lixi.png)

![Interactions ANOVA dCorg](analysis/terrainSA_results/heatmap_anova_2facteurs_R2_interaction_dCorg.png)

![Interactions ANOVA rdt](analysis/terrainSA_results/heatmap_anova_2facteurs_R2_interaction_rdt.png)

### Sobol et Shapley

Les indices de Sobol d'ordre total et les valeurs de Shapley confirment que les dates de campagne dominent les sorties :

| Sortie | Principaux facteurs Sobol total |
|---|---|
| `N_lixi` | `Date_Semis`, `Date_Recolte`, `has_prepa`, `Delta_PREPA_Semis` |
| `dCorg` | `Date_Recolte`, `Date_Semis` |
| `rdt` | `Date_Recolte`, `Date_Semis`, `has_prepa`, `Delta_PREPA_Semis` |

![Sobol total terrainSA](analysis/terrainSA_results/sobol_total_top.png)

![Shapley terrainSA](analysis/terrainSA_results/shapley_top.png)

## Partie 3 - Seuils locaux par arbres de décision

Le notebook `analysis/Analyse_seuils_decision_tree.ipynb` prolonge l'analyse en cherchant des seuils interprétables. L'objectif n'est pas de remplacer le métamodèle global, mais d'obtenir des règles locales : au-delà de tel seuil, la réponse change de régime.

Les arbres de régression contraints obtiennent les performances suivantes :

| Sortie | Q2 test arbre | Paramètres principalement utilisés |
|---|---:|---|
| `N_lixi` | 0.887 | `Date_Semis`, `Delta_PREPA_Semis`, `Date_Recolte`, `has_prepa` |
| `dCorg` | 0.966 | `Date_Recolte`, `Date_Semis` |
| `rdt` | 0.879 | `Delta_PREPA_Semis`, `Date_Recolte`, `Date_Semis`, `has_prepa` |

Pour `N_lixi`, les premiers seuils portent sur `Date_Semis` autour de 73 jours de campagne, puis sur `Date_Recolte` et `Delta_PREPA_Semis`. Pour `dCorg`, le couple `Date_Recolte` / `Date_Semis` structure fortement les régimes. Pour `rdt`, les seuils combinent surtout délai préparation-semis, date de récolte et date de semis.

![Decision tree N_lixi](analysis/decision_tree_thresholds/decision_tree_N_lixi.png)

![Decision tree dCorg](analysis/decision_tree_thresholds/decision_tree_dCorg.png)

![Decision tree rdt](analysis/decision_tree_thresholds/decision_tree_rdt.png)

Les importances internes aux arbres résument ces règles : les dates de campagne dominent, tandis que les doses et types d'engrais ont ici un effet secondaire dans le contexte homogénéisé `terrainSA`.

![Importance arbre N_lixi](analysis/decision_tree_thresholds/decision_tree_importance_N_lixi.png)

![Importance arbre dCorg](analysis/decision_tree_thresholds/decision_tree_importance_dCorg.png)

![Importance arbre rdt](analysis/decision_tree_thresholds/decision_tree_importance_rdt.png)

## Partie 4 - PDP/ICE sur les paramètres temporels principaux

Le notebook `analysis/Analyse_PDP_ICE_terrainSA.ipynb` prolonge l'analyse en reconstruisant les sorties finales à partir des logs dynamiques. Le fichier central est `analysis/pdp_ice_dynamic/final_dynamic_dataset.csv` : il associe chaque point du plan SMT aux valeurs finales observées de `N_lixi`, `DeltaCorg` et `Yield`.

Les courbes PDP/ICE sont volontairement limitées aux paramètres non catégoriels les plus interprétables :

| Colonne SMT | Paramètre agronomique | Lecture |
|---|---|---|
| `feat_14` | `Date_Semis` | date de semis en jour de campagne |
| `feat_15` | `Delta_PREPA_Semis` | délai entre préparation du sol et semis |
| `feat_19` | `Date_Recolte` | date de récolte en jour de campagne |

Les paramètres catégoriels, comme la présence ou le type de préparation du sol, sont exclus de cette lecture PDP/ICE principale. Ils restent importants pour les seuils locaux et certaines analyses globales, mais leurs courbes partielles sont moins naturelles à lire qu'une variation continue de date ou de délai.

Le métamodèle final du notebook obtient de bons scores de généralisation sur le dataset dynamique final :

| Sortie | Q2 test | Lecture rapide |
|---|---:|---|
| `N_lixi` | 0.985 | les courbes PDP/ICE peuvent être interprétées avec une bonne confiance globale |
| `DeltaCorg` | 0.959 | le signal temporel est bien capturé malgré une variabilité résiduelle plus marquée |
| `Yield` | 0.970 | le rendement final est correctement reproduit par le métamodèle |

![Importances finales PDP ICE](analysis/pdp_ice_dynamic/final_feature_importances.png)

### Date de semis

La date de semis agit comme un paramètre structurant du cycle cultural. Les courbes ICE montrent la dispersion entre scénarios, tandis que la courbe noire résume l'effet moyen prédit. Pour le carbone organique, la tendance moyenne indique notamment que le positionnement du semis modifie fortement la valeur finale simulée, ce qui confirme l'importance du calendrier dans `terrainSA`.

![PDP ICE N_lixi Date Semis](analysis/pdp_ice_dynamic/pdp_ice_final_N_lixi_feat_14.png)

![PDP ICE DeltaCorg Date Semis](analysis/pdp_ice_dynamic/pdp_ice_final_DeltaCorg_feat_14.png)

![PDP ICE Yield Date Semis](analysis/pdp_ice_dynamic/pdp_ice_final_Yield_feat_14.png)

### Délai préparation-semis

`Delta_PREPA_Semis` permet d'observer comment l'espacement entre la préparation du sol et le semis influence les sorties. Ce paramètre est particulièrement utile pour interpréter les régimes mis en évidence par les arbres de décision : il combine une opération technique et son positionnement temporel.

![PDP ICE N_lixi Delta Prepa Semis](analysis/pdp_ice_dynamic/pdp_ice_final_N_lixi_feat_15.png)

![PDP ICE DeltaCorg Delta Prepa Semis](analysis/pdp_ice_dynamic/pdp_ice_final_DeltaCorg_feat_15.png)

![PDP ICE Yield Delta Prepa Semis](analysis/pdp_ice_dynamic/pdp_ice_final_Yield_feat_15.png)

### Date de récolte

La date de récolte contrôle la durée effective du cycle. Elle est donc naturellement liée à la dynamique du carbone organique et au rendement. Les courbes PDP/ICE permettent ici de distinguer l'effet moyen d'une modification de date et la diversité des réponses individuelles selon le reste de l'itinéraire technique.

![PDP ICE N_lixi Date Recolte](analysis/pdp_ice_dynamic/pdp_ice_final_N_lixi_feat_19.png)

![PDP ICE DeltaCorg Date Recolte](analysis/pdp_ice_dynamic/pdp_ice_final_DeltaCorg_feat_19.png)

![PDP ICE Yield Date Recolte](analysis/pdp_ice_dynamic/pdp_ice_final_Yield_feat_19.png)

Des trajectoires temporelles interactives sont également disponibles dans `analysis/pdp_ice_dynamic/` au format HTML. Elles servent à vérifier si les différences finales observées par PDP/ICE correspondent à des divergences progressives ou à des transitions plus localisées dans le temps.

## App web

L'app web permet de relancer une analyse complète depuis une interface locale. La synthèse ci-dessus reste toutefois fondée sur les figures générées par les notebooks et versionnées dans le dépôt.

```bash
/Users/benjamin/.pyenv/versions/MAELIA_SA/bin/python -m uvicorn maelia_sa_pipeline.api:app --host 127.0.0.1 --port 8000
```

Puis ouvrir : `http://127.0.0.1:8000`.

## Fichiers utiles

- Analyse terrainSA et métamodèles : `analysis/Analyse_terrainSA.ipynb`
- Analyse des seuils : `analysis/Analyse_seuils_decision_tree.ipynb`
- Analyse PDP/ICE et trajectoires dynamiques : `analysis/Analyse_PDP_ICE_terrainSA.ipynb`
- Résultats terrainSA notebook : `analysis/terrainSA_results/`
- Résultats arbres de décision notebook : `analysis/decision_tree_thresholds/`
- Résultats PDP/ICE dynamiques : `analysis/pdp_ice_dynamic/`
- Notebook de lancement terrainSA : `simulations/batch_simulations_smt_terrainSA.ipynb`
- Figures historiques terrainTest : `figs/`
