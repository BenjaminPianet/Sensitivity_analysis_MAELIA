"""Construction d'un terrain MAELIA portant les huit climats en proportions égales.

Le terrain `terrainSA` actuel n'a qu'un seul îlot, et ses cent parcelles partagent la
même géométrie : elles tombent donc toutes dans la même zone météo. Or MAELIA affecte
le climat **à l'îlot**, en retenant la zone météo dont l'intersection avec l'îlot est
la plus grande (`zoneMeteoMoyenne.gaml`). Faire varier le climat impose donc de créer
plusieurs îlots, spatialement séparés, chacun dans son polygone météo.

Ce module produit `terrainSA_climats` :

  - **huit îlots**, copies translatées de l'îlot d'origine, disposés sur une grille de
    10 km de pas ; chacun porte le sol, la pente et l'exploitation de l'original, de
    sorte que **le climat soit la seule chose qui les distingue** ;
  - **huit polygones météo** de 6 km de côté, centrés sur les îlots — l'espacement
    garantit qu'aucun îlot n'en intersecte deux ;
  - **douze parcelles par îlot**, soit 96 au total : les huit climats sont présents en
    proportions strictement égales ;
  - les **séries climatiques** téléchargées par `climate.py`, au format MAELIA.

Le terrain d'origine n'est jamais modifié : tout est écrit dans un dossier distinct.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.affinity import translate
from shapely.geometry import Point, box

from . import climate as climate_mod
from .chemins import CLIMATS_OBSERVEE, TERRAINS

SOURCE_TERRAIN = TERRAINS / "terrainSA"
TARGET_NAME = "terrainSA_climats"

# Noms des terrains selon ce qui varie. Un terrain par combinaison de strates : le
# mélanger avec un autre rendrait les analyses incomparables.
NOMS_TERRAIN = {
    (True, False): "terrainSA_climats",
    (False, True): "terrainSA_sols",
    (True, True): "terrainSA_climats_sols",
}

PARCELLES_PAR_ILOT = 12          # 12 × 8 climats = 96 parcelles
METEO_SIDE_M = 2_600             # côté du polygone météo centré sur chaque îlot
GRID_COLS = 4                    # disposition 4 × 2
CONTOUR = Path("modeleHydrographique/zonesHydrographiques/contourZH.shp")

# Nom du scénario climatique sous lequel les séries sont aussi déposées. GAMA rejette
# un nomScenarioClimatique vide ; il faut donc un scénario nommé, et le launcher
# impose par défaut « rcp8.5 ». On écrit sous les deux emplacements.
SCENARIO_NAME = "era5_climats"

SPATIAL = Path("modeleAgricole/ilots/dansZone")
METEO = Path("modeleCommun/meteo")
BLOCS = ("blocsDonnees.csv", "blocsDonnees_cor.csv",
         "blocsDateDose.csv", "blocsDateDose_cor.csv")


@dataclass(frozen=True)
class BuiltTerrain:
    path: Path
    ilots: list[str]
    parcelles: list[str]
    assignments: pd.DataFrame     # parcelle -> îlot -> climat


def _positions_dans_contour(contour_shape, n: int) -> list[tuple[float, float]]:
    """Positions des îlots, **à l'intérieur du contour de la zone d'étude**.

    C'est la contrainte décisive : ``zoneMeteo.gaml`` supprime toute zone météo qui
    n'intersecte pas ce contour, et les îlots orphelins retombent alors sur la zone
    survivante la plus proche. Des îlots dispersés au-delà se retrouvent donc tous
    sous le même climat, sans qu'aucune erreur ne soit levée.

    Le contour n'étant pas rectangulaire, on ne peut pas se contenter d'une grille sur
    son enveloppe : on érode le polygone de la demi-largeur d'un carreau météo, puis on
    retient parmi les points intérieurs les ``n`` les plus écartés les uns des autres.
    """
    margin = METEO_SIDE_M / 2 + 200          # le polygone météo doit tenir en entier
    interieur = contour_shape.buffer(-margin)
    if interieur.is_empty:
        raise ValueError("Contour trop petit pour y placer les polygones météo.")

    xmin, ymin, xmax, ymax = interieur.bounds
    pas = METEO_SIDE_M / 4
    candidats = []
    y = ymin
    while y <= ymax:
        x = xmin
        while x <= xmax:
            if interieur.contains(Point(x, y)):
                candidats.append((x, y))
            x += pas
        y += pas
    if len(candidats) < n:
        raise ValueError(f"Seulement {len(candidats)} emplacements disponibles pour {n} îlots.")

    # Sélection gloutonne du point le plus éloigné de ceux déjà retenus.
    retenus = [max(candidats, key=lambda p: (p[0] - xmin) ** 2 + (p[1] - ymin) ** 2)]
    while len(retenus) < n:
        retenus.append(max(
            candidats,
            key=lambda p: min((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 for q in retenus)))

    ecart_min = min(
        ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
        for i, a in enumerate(retenus) for b in retenus[i + 1:])
    if ecart_min < METEO_SIDE_M:
        raise ValueError(
            f"Écartement minimal {ecart_min:.0f} m inférieur au côté d'un carreau "
            f"({METEO_SIDE_M} m) : les zones météo se chevaucheraient.")
    return retenus


def sols_disponibles(terrain: str | Path | None = None) -> list[dict]:
    """Types de sol déclarés par le terrain, dans l'ordre du fichier.

    MAELIA n'attribue pas le sol par la géométrie mais par l'attribut ``ID_SOL`` de
    l'îlot, cherché dans les sols rattachés à sa zone hydrographique
    (``ilot.gaml:79``). Faire varier le sol revient donc à écrire un autre identifiant,
    sans rien déplacer — c'est ce qui permet d'empiler plusieurs îlots au même endroit,
    donc sous le même climat, avec des sols différents.
    """
    racine = Path(terrain) if terrain else SOURCE_TERRAIN
    couche = gpd.read_file(racine / "modeleCommun/typesDeSol/typeDeSolParZH.shp")
    sols = []
    for _, ligne in couche.iterrows():
        sols.append({
            "id_sol": str(ligne["ID_SOL"]),
            "nom": str(ligne.get("ZONE_PEDO") or ligne["ID_SOL"]),
            "profondeur_cm": _nombre(ligne.get("PRO")),
            "argile_pct": _nombre(ligne.get("ARG1")),
            "matiere_organique_pct": _nombre(ligne.get("MO1")),
        })
    return sols


def _nombre(valeur) -> float | None:
    try:
        return round(float(valeur), 3)
    except (TypeError, ValueError):
        return None


def _altitudes(cache_dir: Path) -> dict[int, float]:
    """Altitude renvoyée par Open-Meteo pour chaque point, si le cache est présent."""
    out: dict[int, float] = {}
    for climate in climate_mod.CLIMATE_TYPES:
        cached = cache_dir / f"{climate.id_pdg}_{climate.site}.json"
        if cached.exists():
            out[climate.id_pdg] = float(json.loads(cached.read_text()).get("elevation", 0.0))
    return out


def build(target_root: str | Path | None = None,
          data_dir: str | Path | None = None,
          force: bool = False,
          avec_climat: bool = True,
          avec_sol: bool = False) -> BuiltTerrain:
    """Écrit un terrain portant les strates demandées. N'altère jamais le source.

    Deux strates possibles, indépendantes dans leur mécanisme :

    - **le climat** est géographique. MAELIA retient la zone météo dont l'intersection
      avec l'îlot est la plus grande : faire varier le climat impose donc des îlots
      spatialement séparés, chacun dans son polygone météo.
    - **le sol** est un attribut. L'îlot le désigne par ``ID_SOL`` parmi les sols
      rattachés à sa zone hydrographique : rien ne se déplace, on écrit un autre
      identifiant.

    D'où la construction quand les deux varient : à chaque emplacement climatique, on
    **empile** autant d'îlots que de sols. Ils partagent la géométrie, donc le climat,
    et ne diffèrent que par leur sol — exactement le croisement voulu. L'empilement ne
    gêne pas MAELIA, qui rattache par intersection et par attribut, jamais par
    voisinage.
    """
    root = Path(target_root) if target_root else TERRAINS
    if not (avec_climat or avec_sol):
        raise ValueError("Rien à faire varier : demande le climat, le sol, ou les deux.")
    target = root / NOMS_TERRAIN[(avec_climat, avec_sol)]
    data = Path(data_dir) if data_dir else CLIMATS_OBSERVEE.parent

    if not SOURCE_TERRAIN.exists():
        raise FileNotFoundError(f"Terrain source introuvable : {SOURCE_TERRAIN}")
    series = data / "climats_observee" if (data / "climats_observee").exists() else data / "observee"
    if avec_climat and not series.exists():
        raise FileNotFoundError(
            f"Séries climatiques absentes de {data}. Lance climate.build() d'abord.")

    if target.exists():
        if not force:
            raise FileExistsError(f"{target} existe déjà. Relance avec force=True pour le refaire.")
        shutil.rmtree(target)
    shutil.copytree(SOURCE_TERRAIN, target)

    # Les itinéraires de l'ancien plan n'ont plus de sens ici : les parcelles changent.
    variants = target / "modeleAgricole/agriculteurs/variants_SMT"
    if variants.exists():
        shutil.rmtree(variants)
    variants.mkdir(parents=True)

    src_ilots = gpd.read_file(SOURCE_TERRAIN / SPATIAL / "ilots.shp")
    src_parcelles = gpd.read_file(SOURCE_TERRAIN / SPATIAL / "parcelles.shp")
    modele_ilot = src_ilots.iloc[[0]].copy()
    modele_parcelle = src_parcelles.iloc[[0]].copy()
    exploitation = str(modele_ilot.iloc[0]["ID_EXPL"])

    altitudes = _altitudes(data / "_cache")

    # Emplacements : un par climat, ou celui d'origine si le climat ne varie pas.
    origine = modele_ilot.geometry.iloc[0].centroid
    if avec_climat:
        contour = gpd.read_file(SOURCE_TERRAIN / CONTOUR)
        forme_contour = (contour.geometry.union_all() if hasattr(contour.geometry, "union_all")
                         else contour.geometry.unary_union)
        emplacements = list(zip(climate_mod.CLIMATE_TYPES,
                                _positions_dans_contour(forme_contour,
                                                        len(climate_mod.CLIMATE_TYPES))))
    else:
        emplacements = [(None, (origine.x, origine.y))]

    # Sols : ceux du terrain, ou le seul que l'îtot d'origine désigne.
    sol_origine = str(modele_ilot.iloc[0].get("ID_SOL"))
    sols = sols_disponibles(SOURCE_TERRAIN) if avec_sol else [
        {"id_sol": sol_origine, "nom": sol_origine}]

    ilots, parcelles, assignments = [], [], []
    meteo_polygons = []

    for climate, (x, y) in emplacements:
        dx, dy = x - origine.x, y - origine.y

        if climate is not None:
            centre_x, centre_y = x, y
            half = METEO_SIDE_M / 2
            meteo_polygons.append({
                "ID_PDG": climate.id_pdg,
                "POSX": int(centre_x),
                "POSY": int(centre_y),
                "ALTI_MOY": int(round(altitudes.get(climate.id_pdg, 100.0))),
                "geometry": box(centre_x - half, centre_y - half,
                                centre_x + half, centre_y + half),
            })

        prefixe = (f"clim{climate.code}_{climate.site.lower().replace(' ', '')}"
                   if climate is not None else "site")

        for rang_sol, sol in enumerate(sols):
            # Les îlots d'un même emplacement sont empilés : même géométrie, donc même
            # climat, et un sol différent chacun.
            ilot_id = f"{prefixe}_sol{rang_sol}" if avec_sol else prefixe

            ilot = modele_ilot.copy()
            ilot["ID_ILOT"] = ilot_id
            ilot["ID_EXPL"] = exploitation
            ilot["ID_SOL"] = sol["id_sol"]
            ilot["geometry"] = ilot.geometry.apply(lambda g: translate(g, dx, dy))
            ilots.append(ilot)

            for index in range(PARCELLES_PAR_ILOT):
                parcelle_id = f"{ilot_id}_p{index:02d}"
                parcelle = modele_parcelle.copy()
                parcelle["ID_ILOT"] = ilot_id
                parcelle["ID_PARCELL"] = parcelle_id
                parcelle["geometry"] = parcelle.geometry.apply(lambda g: translate(g, dx, dy))
                parcelles.append(parcelle)
                assignments.append({
                    "parcelle": parcelle_id, "ilot": ilot_id,
                    "climat_code": climate.code if climate else 0,
                    "climat": climate.name if climate else None,
                    "site": climate.site if climate else None,
                    "id_pdg": climate.id_pdg if climate else None,
                    "sol_id": sol["id_sol"], "sol": sol["nom"],
                })

    ilots_gdf = gpd.GeoDataFrame(pd.concat(ilots, ignore_index=True),
                                 geometry="geometry", crs=src_ilots.crs)
    parcelles_gdf = gpd.GeoDataFrame(pd.concat(parcelles, ignore_index=True),
                                     geometry="geometry", crs=src_parcelles.crs)
    meteo_gdf = (gpd.GeoDataFrame(meteo_polygons, geometry="geometry", crs=src_ilots.crs)
                 if meteo_polygons else None)

    for stem, frame in (("ilots", ilots_gdf), ("parcelles", parcelles_gdf)):
        for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            (target / SPATIAL / f"{stem}{suffix}").unlink(missing_ok=True)
        frame.to_file(target / SPATIAL / f"{stem}.shp", driver="ESRI Shapefile", encoding="UTF-8")

    if meteo_polygons:
        for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            (target / METEO / f"polygonesMeteoFrance{suffix}").unlink(missing_ok=True)
        meteo_gdf.to_file(target / METEO / "polygonesMeteoFrance.shp",
                          driver="ESRI Shapefile", encoding="UTF-8")

    parcelle_ids = [a["parcelle"] for a in assignments]
    listing = pd.DataFrame({"ID_PARCELL": parcelle_ids,
                            "CRIT_SPAT_ADHOC": ["une_valeur"] * len(parcelle_ids)})
    listing.to_csv(target / SPATIAL / "parcelles.csv", index=False)
    listing.to_csv(target / SPATIAL / "parcelles_.csv", index=False)

    # Les blocs associent une exploitation à ses parcelles, sur une seule ligne.
    ligne = ";".join([exploitation] + parcelle_ids) + "\n"
    for nom in BLOCS:
        (target / "modeleAgricole" / nom).write_text(ligne, encoding="utf-8")

    if avec_climat:
        # Séries climatiques : on remplace la météo observée, et on retire les
        # scénarios simulés qui ne portent que les anciennes zones.
        shutil.rmtree(target / METEO / "observee", ignore_errors=True)
        shutil.copytree(series, target / METEO / "observee")
        shutil.rmtree(target / METEO / "simulee", ignore_errors=True)
        shutil.copytree(series, target / METEO / "simulee" / SCENARIO_NAME)

    frame = pd.DataFrame(assignments)
    # Nom générique : les strates ne sont plus seulement climatiques. L'ancien nom
    # reste écrit tant que des terrains produits avant coexistent.
    frame.to_csv(target / "affectation_strates.csv", index=False)
    frame.to_csv(target / "affectation_climats.csv", index=False)

    return BuiltTerrain(path=target,
                        ilots=[str(i) for i in ilots_gdf["ID_ILOT"]],
                        parcelles=parcelle_ids,
                        assignments=frame)


def exporter_carte(terrain: str | Path | None = None,
                   sortie: str | Path | None = None) -> Path:
    """Rassemble les couches du terrain en un GeoPackage lisible dans QGIS.

    Les shapefiles de MAELIA ne portent pas le climat : ouverts tels quels, les huit
    îlots sont indiscernables. On y joint donc l'affectation, de sorte qu'une seule
    couche suffise à voir quel climat occupe quel emplacement — et à vérifier de visu
    ce que le fichier de correspondance de GAMA affirme.

    Quatre couches, dans un fichier unique :

      ``ilots``        les îlots, avec les strates qu'ils portent — climat, sol, site
      ``parcelles``    les parcelles, avec les strates héritées de leur îlot
      ``zones_meteo``  les carreaux météo, avec l'altitude retenue
      ``contour``      le contour de la zone d'étude — hors de lui, MAELIA supprime
                       les zones météo, ce qui explique le placement des îlots
    """
    racine = Path(terrain) if terrain else TERRAINS / TARGET_NAME
    cible = Path(sortie) if sortie else racine / "carte_climats.gpkg"
    if cible.exists():
        cible.unlink()

    fichier = racine / "affectation_strates.csv"
    if not fichier.exists():
        fichier = racine / "affectation_climats.csv"
    affectation = pd.read_csv(fichier)
    par_ilot = affectation.drop_duplicates("ilot").set_index("ilot")

    # Les strates présentes varient d'un terrain à l'autre : on joint ce qui existe.
    portees = [c for c in ("climat", "sol", "site", "id_pdg", "sol_id")
               if c in affectation.columns]

    ilots = gpd.read_file(racine / SPATIAL / "ilots.shp")
    for colonne in portees:
        ilots[colonne] = ilots.ID_ILOT.map(par_ilot[colonne])
    ilots["n_parcelles"] = ilots.ID_ILOT.map(affectation.groupby("ilot").size())

    parcelles = gpd.read_file(racine / SPATIAL / "parcelles.shp")
    par_parcelle = affectation.set_index("parcelle")
    for colonne in portees:
        parcelles[colonne] = parcelles.ID_PARCELL.map(par_parcelle[colonne])

    meteo = gpd.read_file(racine / "modeleCommun/meteo/polygonesMeteoFrance.shp")
    if "id_pdg" in affectation.columns and affectation.id_pdg.notna().any():
        par_pdg = affectation.dropna(subset=["id_pdg"]).drop_duplicates("id_pdg")
        par_pdg = par_pdg.set_index(par_pdg.id_pdg.astype(int))
        meteo["climat"] = meteo.ID_PDG.astype(int).map(par_pdg.climat)
        meteo["site"] = meteo.ID_PDG.astype(int).map(par_pdg.site)

    ilots.to_file(cible, layer="ilots", driver="GPKG")
    parcelles.to_file(cible, layer="parcelles", driver="GPKG")
    meteo.to_file(cible, layer="zones_meteo", driver="GPKG")
    contour = racine / CONTOUR
    if contour.exists():
        gpd.read_file(contour).to_file(cible, layer="contour", driver="GPKG")
    return cible


def validate(terrain: Path) -> list[str]:
    """Vérifie que chaque îlot tombe dans un et un seul polygone météo."""
    issues: list[str] = []
    ilots = gpd.read_file(terrain / SPATIAL / "ilots.shp")
    parcelles = gpd.read_file(terrain / SPATIAL / "parcelles.shp")
    meteo = gpd.read_file(terrain / METEO / "polygonesMeteoFrance.shp")

    if len(meteo) != len(climate_mod.CLIMATE_TYPES):
        issues.append(f"{len(meteo)} polygones météo, {len(climate_mod.CLIMATE_TYPES)} attendus")

    for _, ilot in ilots.iterrows():
        touches = meteo[meteo.intersects(ilot.geometry)]
        if len(touches) != 1:
            issues.append(f"îlot {ilot.ID_ILOT} intersecte {len(touches)} polygones météo, 1 attendu")

    # La vérification qui manquait, et qui a coûté un run entier : MAELIA
    # supprime les zones météo n'intersectant pas le contour de la zone d'étude, puis
    # rattache les îlots orphelins à la zone survivante la plus proche. Tous les îlots
    # se retrouvent alors sous le même climat, en silence.
    contour = gpd.read_file(terrain / CONTOUR)
    forme = contour.geometry.union_all() if hasattr(contour.geometry, "union_all") \
        else contour.geometry.unary_union
    dehors = meteo[~meteo.intersects(forme)]
    if len(dehors):
        issues.append(
            f"{len(dehors)} polygone(s) météo hors du contour de la zone d'étude "
            f"(ID_PDG {sorted(dehors.ID_PDG)}) : MAELIA les supprimerait.")
    for _, ilot in ilots.iterrows():
        if not ilot.geometry.intersects(forme):
            issues.append(f"îlot {ilot.ID_ILOT} hors du contour de la zone d'étude")

    counts = parcelles.groupby("ID_ILOT").size()
    if counts.nunique() != 1:
        issues.append(f"parcelles réparties inégalement : {counts.to_dict()}")

    # Le scénario nommé doit exister : c'est lui que l'XML désigne.
    scenario = terrain / METEO / "simulee" / SCENARIO_NAME
    if not scenario.exists() or not list(scenario.glob("*.csv")):
        issues.append(f"scénario climatique absent : {scenario}")

    # Chaque zone météo doit disposer de données pour toutes les campagnes.
    for year_file in sorted((terrain / METEO / "observee").glob("*.csv")):
        zones = set(pd.read_csv(year_file, sep=";").ID_PDG.unique())
        manquantes = set(meteo.ID_PDG) - zones
        if manquantes:
            issues.append(f"{year_file.name} : zones sans données {sorted(manquantes)}")
    return issues
