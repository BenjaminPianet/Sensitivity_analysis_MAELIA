# Premier parcours, de zéro aux résultats

Pour faire ce tutoriel, vous aurez besoin d'avoir installé GAMA et téléchargé MAELIA. À la
fin, vous aurez produit vos propres données et su quels paramètres influencent le rendement.

La simulation occupe l'essentiel du parcours, et vous n'avez rien à y faire.

À chaque étape, un encadré **« Ce que vous devez voir »** vous permet de vérifier que
vous êtes sur la bonne voie avant de continuer.

---

## 1. Lancer l'application

Depuis le dossier qui contient `app_maelia`, ouvrez un terminal et copiez-collez :

```bash
python3 -m uvicorn app_maelia.api:app --reload --port 8000
```

Ouvrez <http://127.0.0.1:8000/> dans la barre de recherche de votre navigateur.

> **Ce que vous devez voir** : une page en deux colonnes. À gauche, « Que voulez-vous
> faire ? » avec deux choix. À droite, un bandeau « Couverture » annonçant
> 2 400 simulations disponibles : ce sont les données livrées avec l'application, que
> nous n'utiliserons pas ici.

---

## 2. Vérifier que GAMA est trouvé

Tout en bas de la colonne de droite, dépliez **« Où sont les fichiers »**.

> **Ce que vous devez voir** : deux lignes en vert :
>
> ```
> Installation MAELIA     ~/Workspace_GAMA/MAELIA
> Script gama-headless    /Applications/Gama.app/Contents/headless/gama-headless.sh
> ```
>
> Si l'une est en rouge, saisissez le bon chemin dans le champ correspondant et
> enregistrez. Le réglage prend effet aussitôt, sans redémarrer.

C'est la seule configuration du tutoriel.

---

## 3. Choisir de produire des données

Cochez **« Construire un plan sur mesure »**.

L'arborescence de gauche change : elle décrit maintenant un espace **libre**, celui que
vous allez explorer. Chaque ligne est un paramètre de l'itinéraire technique : date de
semis, nombre d'apports d'azote, doses, date de récolte.

Laissez tout tel quel pour ce premier parcours. Ne cochez ni le climat ni le sol : ils
imposeraient de construire un terrain, ce qui allonge le parcours sans rien apprendre
de plus pour commencer.

> **Ce que vous devez voir** : le panneau « Plan d'expérience correspondant » apparaît,
> avec **14 variables du plan** et **12 sous-espaces**. Les douze sous-espaces sont les
> combinaisons du nombre d'apports (0 à 3) et du nombre de préparations du sol (0 à 2).

---

## 4. Régler la taille du plan

Dans le panneau du plan, portez **« Points à simuler »** à **500**.

Pourquoi 500 ? Parce que l'application refuse d'analyser ce qu'elle juge insuffisant, et
que trois seuils s'appliquent : 120 points pour le R² par paramètre, 300 pour les
métamodèles, et 40 par sous-espace : soit 480 : pour l'analyse par sous-espace. Avec
100 points, les trois verdicts seraient rouges et rien ne serait calculé.

Cliquez sur **« Vérifier les calendriers »**.

> **Ce que vous devez voir** : `500 / 500 calendriers valides`, aucun échec. Cette vérification simule le déroulement de chaque itinéraire sur les dix
> campagnes sans rien écrire : elle vérifie les plans infaisables avant les simulations de GAMA.

---

## 5. Générer les fichiers

Cliquez sur **« Générer les fichiers »**.

> **Ce que vous devez voir** : dans votre dossier `app_maelia`, un dossier daté dans `simulations_app/`, du type
> `20260817_152813_tutoriel500`, contenant :
>
> ```
> dateDose/   points.csv   doe.npy   manifest.json   space_spec.json   xml/
> ```
>
> Et dans votre navigateur (là où l'application tourne), quatre étapes numérotées, avec les commandes à copier.

Rien n'a encore été simulé, ce sera à vous de lancer lancer.

---

## 6. Simuler

Copiez les commandes des étapes **1** et **2** dans un terminal (le chemin n'importe pas ici), l'une après l'autre.

La première recopie les itinéraires dans le terrain. La seconde lance GAMA cinq fois,
une par lot de 100 parcelles.

GAMA n'affiche pas grand-chose pendant qu'il travaille ; c'est normal. La durée dépend
de votre machine, du terrain et du nombre de campagnes — laissez-le aller jusqu'au bout,
il rend la main de lui-même.

> **Ce que vous devez voir**, une fois fini : cinq dossiers de sortie dans
> `~/Workspace_GAMA/MAELIA/models/main/log/`, nommés `terrainSA_smt_000_<date>` à
> `terrainSA_smt_004_<date>`. Chacun contient neuf fichiers, dont `sorties_CN.csv` et
> `suiviOTParParcelle.csv` : les deux que la suite va lire.

---

## 7. Rassembler les sorties

Copiez la commande de l'étape **3** :

```bash
python3 -m app_maelia.collecte simulations_app/<votre_dossier>
```

Elle relie chaque point du plan aux sorties de la parcelle sur laquelle il a tourné, et
écrit `dataset_metamodel.csv` dans le dossier du plan.

> **Ce que vous devez voir**
>
> ```
> 500 points assemblés depuis 5 run(s) · écrit dans …/dataset_metamodel.csv
>   smt_000 ← terrainSA_smt_000_2026-08-17_…
>   …
> ```
>
> Si un run manque, il est nommé : relancez GAMA pour celui-là, puis rappelez la
> commande.

---

## 8. Analyser

Revenez à l'application. Cochez **« Analyser des données existantes »**, puis collez le
chemin du dossier du plan dans **« ou un dossier de votre choix »**.

L'application y trouve le jeu de données **et** la spécification de l'espace : l'analyse
portera donc exactement sur le plan que vous avez tiré, sans que vous ayez à le lui
décrire.

Cliquez sur **« Lancer l'analyse »**.

> **Ce que vous devez voir** : le bandeau annonce **500 simulations disponibles**,
> 12 sous-espaces, et les trois verdicts au vert.

### Les résultats attendus

Vos chiffres ne seront pas identiques à ceux présentés ci-dessous car le tirage dépend de la graine, et
500 points laissent une marge d'incertitude. Mais **le classement et les ordres de
grandeur doivent se retrouver.**

**Le rendement.** C'est la sortie la mieux expliquée.

| paramètre | R² | points actifs |
|---|---|---|
| Nombre d'apports d'azote | 0,48 | 500 |
| Dose du 1ᵉʳ apport | 0,42 | 374 |
| Dose du 2ᵉ apport | 0,36 | 248 |
| Dose du 3ᵉ apport | 0,34 | 123 |
| Profondeur de préparation 1 | 0,02 | 333 |

Quatre leviers azotés en tête, le travail du sol très loin derrière. Les métamodèles
atteignent **Q² ≈ 0,97** : le rendement est presque entièrement déterminé par
l'itinéraire.

**Le carbone organique du sol.** Même hiérarchieavec moins d'écart entre les paramètres : le nombre
d'apports à 0,31, les trois doses entre 0,19 et 0,28, le nombre de préparations à 0,10.
Q² ≈ 0,79.

**L'azote lixivié.** La sortie la moins prévisible : la dose du troisième apport arrive
en tête à 0,29, suivie de la première à 0,26. Q² ≈ 0,77, et les quatre familles
divergent nettement (0,65 à 0,77) : signe que le résultat dépend du modèle choisi et
demande de la prudence.


---

## Et ensuite

**Faire varier le milieu.** Reprenez à l'étape 3 en cochant le climat, le sol, ou les
deux. L'application construit alors le terrain nécessaire : huit emplacements pour les
climats, autant d'îlots empilés que de sols : et le plan croise les strates de façon
exactement équilibrée, au prix de plus de parcelles par lot.

**Restreindre l'espace.** Dans l'arborescence, resserrez les bornes d'un paramètre ou
décochez des niveaux : le bandeau de couverture vous dit aussitôt combien de simulations
tombent encore dans l'espace décrit, et ce qui reste calculable.

**Comprendre le tirage.** Le plan n'est pas un tirage au hasard point par point : voir
la section « Comment le plan est tiré » du [README](README.md).
