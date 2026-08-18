# Analyse de sensibilité MAELIA

Application web pour explorer l'effet des règles de décision, du cliamt et du sol sur les
sorties du modèle MAELIA : rendement, lixiviation d'azote, variation du carbone
organique du sol.



## Lancer

```bash
cd <le dossier qui contient app_maelia>
python3 -m uvicorn app_maelia.api:app --reload --port 8000
```

Puis ouvrir <http://127.0.0.1:8000/>.

Dépendances : `fastapi`, `uvicorn`, `pandas`, `numpy`, `scikit-learn`, `smt`.

## Les deux façons de travailler

L'application pose une seule question au départ : **les simulations existent-elles
déjà ?**

**Analyser des données existantes.** On part de simulations déjà produites. L'espace
exploré n'est alors pas un choix : il est imposé par le plan qui les a produites. On
peut le restreindre — c'est le rôle des bornes de l'arborescence — mais pas
l'élargir, faute de simulations pour remplir ce qu'on ajouterait.

Deux jeux sont livrés dans `donnees_app/` ; le champ « ou un dossier de votre choix »
accepte n'importe quel dossier de votre machine, pourvu qu'il contienne un
`dataset_metamodel.csv`. S'il contient aussi un `space_spec.json`, celui-ci décrit
exactement l'espace qui a produit ces simulations et fait foi.

**Construire un plan sur mesure.** On définit un espace libre, dont l'application tire
un plan d'expérience à simuler ensuite avec GAMA. Deux caractéristiques du milieu
peuvent y être ajoutées — le **climat** et le **type de sol**. Elles deviennent des
paramètres comme les autres, à ceci près qu'elles ne sont pas tirées au sort : elles
sont portées par l'îlot auquel appartient chaque parcelle. Les activer oblige donc
l'application à construire un terrain, ce qui est annoncé avant, pas découvert après.

Les deux mécanismes diffèrent. Le climat est **géographique** : MAELIA retient la zone
météo qui recouvre le plus l'îlot, il faut donc huit emplacements séparés. Le sol est un
**attribut** : l'îlot le désigne par son identifiant parmi les sols de sa zone
hydrographique, rien ne se déplace. C'est ce qui permet de croiser les deux — à chaque
emplacement climatique, on empile autant d'îlots que de sols. Huit climats et trois sols
donnent 24 îlots, 288 parcelles, et douze points par case.

## Où sont les fichiers

| Dossier | Contenu |
|---|---|
| `espaces_app/` | les espaces de conception livrés (`.json`) |
| `donnees_app/` | les jeux de simulations fournis, et les séries climatiques observées |
| `terrains_app/` | les terrains GAMA que l'application construit |
| `simulations_app/` | les plans produits : `dateDose`, XML d'expérience, DOE tiré |
| `results_app/` | les résultats d'analyse exportés |
| `ui/` `outils/` `tests_app/` | interface, dépendances internes, tests |

Le panneau « Où sont les fichiers », en bas de l'interface, donne **une seule ancre
absolue** — le dossier de l'application — et écrit tout le reste relativement à elle :
`espaces_app`, `simulations_app`, `results_app`. Ce qui vit dehors s'écrit relativement
au dossier personnel, avec un tilde. Le chemin complet reste disponible au survol, pour
le copier dans un terminal. Quand quelque chose ne démarre pas, c'est la première chose
à regarder.

## Ce qui reste à l'extérieur

Deux chemins seulement, parce qu'ils désignent des logiciels installés ailleurs :

En pratique, GAMA s'installe presque toujours au même endroit et l'application le
trouve seule : les deux champs affichent en filigrane le chemin réellement utilisé, et
il n'y a rien à saisir tant que cela convient. Pour une installation hors norme,
saisissez le chemin dans le panneau « Où sont les fichiers » et enregistrez. Le réglage est écrit dans `reglages.json`, il prend
effet aussitôt et survit au redémarrage. L'application dit tout de suite si le chemin
existe, et un chemin encore inexistant est accepté — on peut préparer sa configuration
avant d'installer GAMA.

Trois sources sont essayées, dans cet ordre :

1. `reglages.json`, ce que vous avez saisi — il l'emporte toujours ;
2. les variables d'environnement `MAELIA_ROOT` et `GAMA_HEADLESS` ;
3. les emplacements d'installation habituels, sous macOS, Linux et Windows.

```bash
# Équivalent en ligne de commande, si vous préférez
export MAELIA_ROOT=/chemin/vers/MAELIA
export GAMA_HEADLESS=/chemin/vers/gama-headless.sh
```

Ces deux réglages ne servent qu'à **générer et lancer des simulations**. Pour analyser
des données déjà produites, l'application n'a besoin ni de MAELIA ni de GAMA.

## Comment le plan est tiré

Le plan n'est pas un tirage au hasard. Le squelette de l'itinéraire — combien d'apports,
combien de préparations — est fixé d'abord, et chacune des douze combinaisons reçoit
exactement la même part du budget de simulations. À l'intérieur d'une combinaison, la
liste des paramètres ne bouge plus : on y tire un hypercube latin, qui répartit chaque
paramètre régulièrement sur toute sa plage au lieu de le laisser s'agglutiner.

Ce n'est donc pas un tirage au sort point par point : les points ne sont pas
indépendants, et la répartition entre combinaisons n'est pas laissée au hasard.

Trois conséquences pratiques. À la même graine, le même plan : deux générations
successives produisent des fichiers identiques. Aucune combinaison n'est sous-dotée par
accident — c'est ce qui rend comparables les analyses par sous-espace. Et chaque point
porte des valeurs qui lui sont propres : deux plans de graines différentes explorent des
valeurs différentes, et se complètent si on les fusionne. Le tirage est instantané,
quelle que soit la taille demandée.

Le plan est mélangé avant écriture, parce que le point d'indice *i* s'exécute sur la
parcelle *i modulo* leur nombre : rangé par sous-espace, il alignerait les combinaisons
sur les îlots, et donc sur les climats.

## Lire les résultats

Dans le classement HSIC, les effets simples et les interactions d'ordre 2 et 3 portent
des couleurs distinctes, avec la légende correspondante.

La comparaison des métamodèles se lit dans son seul tableau : quatre familles
entraînées sur le même partage, la coche marque celle qui est retenue, l'écart entre R²
et Q² dit le surapprentissage. C'est la seule mesure de prévisibilité de l'application —
le métamodèle isolé a été retiré, il n'en donnait qu'une redite.

## Voir le terrain dans QGIS

Les shapefiles de MAELIA ne portent pas le climat : ouverts tels quels, les huit îlots
sont indiscernables. L'application produit donc une couche qui le joint :

```bash
python3 -c "from app_maelia import terrain; print(terrain.exporter_carte())"
```

Le fichier `terrains_app/terrainSA_climats/carte_climats.gpkg` s'ouvre d'un glisser-
déposer dans QGIS et contient quatre couches — `ilots`, `parcelles`, `zones_meteo` et
`contour`. Colorez `ilots` par le champ **climat** (Propriétés → Symbologie →
Catégorisé) pour voir quel climat occupe quel emplacement.

La couche `contour` explique le reste : MAELIA supprime toute zone météo qui n'intersecte
pas ce contour, puis rattache silencieusement les îlots orphelins à la zone survivante
la plus proche. C'est pourquoi les huit îlots sont placés à l'intérieur d'un contour
érodé, à 3,1 km les uns des autres au minimum. Le fichier que GAMA écrit après chaque
run, `corresponsanceIlotZoneMeteo.csv`, permet de vérifier que l'affectation obtenue est
bien celle qui était voulue.

## Refermer la boucle

En mode sur mesure, « Générer les fichiers » écrit un dossier daté dans
`simulations_app/` contenant le plan, les fichiers d'entrée MAELIA, la spécification de
l'espace, et les commandes à exécuter. Le panneau les donne dans l'ordre : installer les
itinéraires dans le terrain, lancer GAMA, puis **rassembler les sorties** —

```bash
python3 -m app_maelia.collecte simulations_app/<votre_plan>
```

Cette dernière relie chaque point du plan aux sorties de la parcelle sur laquelle il a
tourné et écrit `dataset_metamodel.csv` dans le dossier du plan, à côté du
`space_spec.json` qui s'y trouve déjà. Il ne reste qu'à saisir ce dossier en mode
« données » : l'analyse porte alors exactement sur l'espace qui a servi à tirer le plan.

Deux précautions y sont prises. GAMA n'écrase jamais ses dossiers de sortie : après
trois exécutions du même plan, trois dossiers portent le même préfixe, et seules les
sorties **postérieures au plan** sont retenues. Et une campagne sans récolte — un blé
semé à l'automne n'est moissonné que l'été suivant — est écartée du calcul du rendement
au lieu d'y compter un zéro.

Pour un premier parcours de bout en bout, voir **[TUTORIEL.md](TUTORIEL.md)**.

