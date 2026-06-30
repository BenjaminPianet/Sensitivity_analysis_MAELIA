#!/usr/bin/env python3
"""Build a project-local MAELIA terrainSA include directory.

This script does not modify the GAMA/MAELIA workspace. It reads the current
MAELIA includes/terraintest as a template and writes a project-local include
root usable through the GAMA parameter `cheminModeleVersDonnees`:

    Sensitivity_analysis_MAELIA/simulations/gama_includes/terrainSA

The terrain contains one ilot, beauce_5, and 100 cloned parcelles:
beauce_5_sa_000 ... beauce_5_sa_099.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd

try:
    import geopandas as gpd
except Exception as exc:  # pragma: no cover - message for notebooks/users
    raise RuntimeError(
        "geopandas est requis pour construire terrainSA. "
        "Installe les dépendances du projet avec `pip install -r requirements.txt`."
    ) from exc


DEFAULT_MAELIA_ROOT = Path('/Users/benjamin/Workspace_GAMA/MAELIA')
DEFAULT_PROJECT_ROOT = Path('/Users/benjamin/files/Repositories/Sensitivity_analysis_MAELIA')
DEFAULT_PROJECT_INCLUDES_ROOT = DEFAULT_PROJECT_ROOT / 'simulations' / 'gama_includes'

SOURCE_TERRAIN_NAME = 'terraintest'
TARGET_TERRAIN_NAME = 'terrainSA'
SOURCE_ILOT = 'beauce_5'
SOURCE_PARCELLE = 'beauce_5_1'
EXPLOITATION_ID = 'organique_beauce_5'
N_CLONES = 100


def _remove_shapefile_family(directory: Path, stem: str) -> None:
    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.qix']:
        path = directory / f'{stem}{ext}'
        if path.exists():
            path.unlink()


def _clone_ids(n_clones: int = N_CLONES) -> list[str]:
    return [f'beauce_5_sa_{i:03d}' for i in range(n_clones)]


def validate_terrain_sa(terrain_dir: Path, n_clones: int = N_CLONES) -> dict[str, object]:
    spatial_dir = terrain_dir / 'modeleAgricole' / 'ilots' / 'dansZone'
    blocs = terrain_dir / 'modeleAgricole' / 'blocsDonnees.csv'
    variants = terrain_dir / 'modeleAgricole' / 'agriculteurs' / 'variants_SMT'

    required = [spatial_dir / 'ilots.shp', spatial_dir / 'parcelles.shp', blocs]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError('terrainSA incomplet, fichiers manquants: ' + ', '.join(missing))

    ilots = gpd.read_file(spatial_dir / 'ilots.shp')
    parcelles = gpd.read_file(spatial_dir / 'parcelles.shp')
    expected = set(_clone_ids(n_clones))
    actual = set(parcelles['ID_PARCELL'].astype(str))

    if len(ilots) != 1:
        raise RuntimeError(f'terrainSA doit contenir 1 ilot, trouvé {len(ilots)}')
    if str(ilots.iloc[0]['ID_ILOT']) != SOURCE_ILOT:
        raise RuntimeError(f"ilot attendu {SOURCE_ILOT}, trouvé {ilots.iloc[0]['ID_ILOT']}")
    if len(parcelles) != n_clones:
        raise RuntimeError(f'terrainSA doit contenir {n_clones} parcelles, trouvé {len(parcelles)}')
    if actual != expected:
        extra = sorted(actual - expected)[:5]
        missing_ids = sorted(expected - actual)[:5]
        raise RuntimeError(f'IDs clones invalides. extra={extra}, missing={missing_ids}')
    if parcelles['ID_ILOT'].astype(str).nunique() != 1:
        raise RuntimeError('toutes les parcelles doivent pointer vers un seul ID_ILOT')
    if not parcelles['ID_ILOT'].astype(str).eq(SOURCE_ILOT).all():
        raise RuntimeError(f'toutes les parcelles doivent pointer vers {SOURCE_ILOT}')

    text = blocs.read_text(errors='replace').strip()
    parts = [part.strip() for part in text.split(';') if part.strip()]
    if not parts or parts[0] != EXPLOITATION_ID:
        raise RuntimeError(f'blocsDonnees doit commencer par {EXPLOITATION_ID}')
    if set(parts[1:]) != expected:
        raise RuntimeError('blocsDonnees ne contient pas exactement les 100 clones attendus')

    variants.mkdir(parents=True, exist_ok=True)
    return {
        'terrain_dir': str(terrain_dir),
        'includes_root': str(terrain_dir.parent),
        'n_ilots': int(len(ilots)),
        'n_parcelles': int(len(parcelles)),
        'ilot': str(ilots.iloc[0]['ID_ILOT']),
        'exploitation': str(ilots.iloc[0]['ID_EXPL']),
        'soil': str(ilots.iloc[0]['ID_SOL']),
        'zh': str(ilots.iloc[0]['ID_ZH']),
        'first_clone': min(expected),
        'last_clone': max(expected),
    }


def build_terrain_sa(
    maelia_root: Path = DEFAULT_MAELIA_ROOT,
    project_includes_root: Path = DEFAULT_PROJECT_INCLUDES_ROOT,
    force: bool = False,
    n_clones: int = N_CLONES,
) -> Path:
    maelia_root = Path(maelia_root).expanduser()
    project_includes_root = Path(project_includes_root).expanduser()
    source = maelia_root / 'includes' / SOURCE_TERRAIN_NAME
    target = project_includes_root / TARGET_TERRAIN_NAME

    if not source.exists():
        raise FileNotFoundError(f'Terrain source introuvable: {source}')

    if target.exists():
        if not force:
            validate_terrain_sa(target, n_clones=n_clones)
            return target
        shutil.rmtree(target)

    project_includes_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)

    # Le terrain source peut embarquer des tables d'engrais incomplètes.
    # On force les paramètres canoniques de MAELIA pour que l'engrais AN
    # utilisé par le plan SMT soit toujours correctement défini.
    canonical_params = maelia_root / 'models' / 'parametres'
    target_params = target / 'parametres'
    target_params.mkdir(parents=True, exist_ok=True)
    for name in ['Engrais.csv', 'engrais_qte_annuelle_dispo.csv']:
        src_param = canonical_params / name
        if src_param.exists():
            shutil.copy2(src_param, target_params / name)

    src_spatial = source / 'modeleAgricole' / 'ilots' / 'dansZone'
    dst_spatial = target / 'modeleAgricole' / 'ilots' / 'dansZone'

    current_ilots = gpd.read_file(src_spatial / 'ilots.shp')
    current_parcelles = gpd.read_file(src_spatial / 'parcelles.shp')

    ilot = current_ilots[current_ilots['ID_ILOT'].astype(str).eq(SOURCE_ILOT)].copy()
    if len(ilot) != 1:
        raise RuntimeError(f'Expected one {SOURCE_ILOT} ilot, found {len(ilot)}')

    base = current_parcelles[current_parcelles['ID_PARCELL'].astype(str).eq(SOURCE_PARCELLE)].copy()
    if len(base) != 1:
        raise RuntimeError(f'Expected one {SOURCE_PARCELLE} parcelle, found {len(base)}')

    clone_ids = _clone_ids(n_clones)
    rows = []
    for clone_id in clone_ids:
        row = base.copy()
        row.loc[:, 'ID_ILOT'] = SOURCE_ILOT
        row.loc[:, 'ID_PARCELL'] = clone_id
        rows.append(row)

    parcelles = pd.concat(rows, ignore_index=True)
    parcelles = gpd.GeoDataFrame(parcelles, geometry='geometry', crs=current_parcelles.crs)

    for stem in ['ilots', 'parcelles']:
        _remove_shapefile_family(dst_spatial, stem)

    ilot.to_file(dst_spatial / 'ilots.shp', driver='ESRI Shapefile', encoding='UTF-8')
    parcelles.to_file(dst_spatial / 'parcelles.shp', driver='ESRI Shapefile', encoding='UTF-8')

    parcelles_csv = pd.DataFrame({
        'ID_PARCELL': clone_ids,
        'CRIT_SPAT_ADHOC': ['une_valeur'] * len(clone_ids),
    })
    parcelles_csv.to_csv(dst_spatial / 'parcelles.csv', index=False)
    parcelles_csv.to_csv(dst_spatial / 'parcelles_.csv', index=False)

    bloc_line = EXPLOITATION_ID + ';' + ';'.join(clone_ids) + ';\n'
    for name in ['blocsDonnees.csv', 'blocsDonnees_cor.csv', 'blocsDateDose.csv', 'blocsDateDose_cor.csv']:
        (target / 'modeleAgricole' / name).write_text(bloc_line, encoding='utf-8')

    variants_dir = target / 'modeleAgricole' / 'agriculteurs' / 'variants_SMT'
    variants_dir.mkdir(parents=True, exist_ok=True)
    for path in variants_dir.glob('dateDose_smt_*.csv'):
        path.unlink()

    (target / 'README_terrainSA.txt').write_text(
        'terrainSA generated by simulations/build_terrainSA_project.py\n'
        f'Source terrain: {source}\n'
        f'Source ilot: {SOURCE_ILOT}\n'
        f'Source parcelle: {SOURCE_PARCELLE}\n'
        f'Clones: {clone_ids[0]} ... {clone_ids[-1]}\n'
        'The GAMA workspace is not modified; notebooks pass this project-local include root via cheminModeleVersDonnees.\n',
        encoding='utf-8',
    )

    validate_terrain_sa(target, n_clones=n_clones)
    return target


def ensure_terrain_sa(
    maelia_root: Path = DEFAULT_MAELIA_ROOT,
    project_includes_root: Path = DEFAULT_PROJECT_INCLUDES_ROOT,
    force: bool = False,
    n_clones: int = N_CLONES,
) -> Path:
    return build_terrain_sa(
        maelia_root=maelia_root,
        project_includes_root=project_includes_root,
        force=force,
        n_clones=n_clones,
    )


if __name__ == '__main__':
    terrain = ensure_terrain_sa(force=False)
    info = validate_terrain_sa(terrain)
    print('terrainSA ready')
    for key, value in info.items():
        print(f'  {key}: {value}')
