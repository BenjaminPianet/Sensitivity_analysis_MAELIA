"""Assemble les sorties de GAMA en un jeu de données analysable.

C'est le maillon qui referme la boucle. L'application produit un plan, GAMA le
simule et écrit ses sorties dans son propre dossier de journaux ; il reste à relier
chaque point du plan aux sorties de la parcelle sur laquelle il a tourné, et à
déposer le résultat à côté du plan sous le nom que le mode « données » attend.

**Deux pièges, tous deux rencontrés en vrai.**

*Les sorties périmées.* GAMA nomme ses dossiers ``<terrain>_<run>_<horodatage>`` sans
jamais écraser les précédents : au bout de trois exécutions du même plan, trois
dossiers portent le même préfixe. On ne retient donc que ceux **postérieurs à la
génération du plan**, et pour chaque run le plus récent d'entre eux.

*La campagne sans récolte.* Un blé semé en septembre n'est récolté que l'été suivant :
la première campagne n'a pas de rendement. On moyenne donc sur les campagnes
exploitables, sans faire compter un zéro pour une absence.

En ligne de commande :

    python3 -m app_maelia.collecte simulations_app/20260817_120114_essai
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import chemins

# Sorties MAELIA retenues, et la colonne d'où chacune vient.
FLUX = {"N_lixi": "N_lixivie[kgN/ha]", "dCorg": "delta_Corg[kgC/ha]"}
SORTIES = ("N_lixi", "dCorg", "rdt")


@dataclass(frozen=True)
class Collecte:
    """Ce que la collecte a produit, et ce qu'elle a dû écarter."""

    dataset: Path
    n_points: int
    n_runs: int
    dossiers: dict[str, Path]
    manquants: list[str]

    def resume(self) -> str:
        lignes = [f"{self.n_points} points assemblés depuis {self.n_runs} run(s)",
                  f"écrit dans {self.dataset}"]
        if self.manquants:
            lignes.append(f"runs sans sortie : {', '.join(self.manquants)}")
        return " · ".join(lignes)


def _sorties_du_run(racine: Path, decoupage: str, run: str,
                    depuis: float) -> Path | None:
    """Dossier de sortie le plus récent pour ce run, postérieur au plan."""
    candidats = [d for d in racine.glob(f"{decoupage}_{run}_*")
                 if d.is_dir() and (d / "sorties_CN.csv").exists()
                 and d.stat().st_mtime >= depuis]
    return max(candidats, key=lambda d: d.stat().st_mtime) if candidats else None


def _sorties_par_parcelle(dossier: Path) -> pd.DataFrame:
    """Une ligne par parcelle : les trois sorties, moyennées sur les campagnes."""
    cn = pd.read_csv(dossier / "sorties_CN.csv", sep=";")
    ot = pd.read_csv(dossier / "suiviOTParParcelle.csv", sep=";")
    colonne_rdt = next(c for c in ot.columns if "rendement" in c)

    flux = (cn.groupby(["parcelle", "annee"])[list(FLUX.values())].sum().reset_index()
              .rename(columns={v: k for k, v in FLUX.items()}))
    # Une campagne sans récolte ne vaut pas un rendement nul : on l'écarte.
    recolte = (ot[ot[colonne_rdt] > 0].groupby(["parcelle", "annee"])[colonne_rdt]
                 .mean().reset_index().rename(columns={colonne_rdt: "rdt"}))

    annuel = flux.merge(recolte, on=["parcelle", "annee"], how="left")
    return annuel.groupby("parcelle", as_index=False)[list(SORTIES)].mean()


def collecter(plan: str | Path, sorties_racine: str | Path | None = None,
              depuis: float | None = None) -> Collecte:
    """Relie le plan à ses sorties GAMA et écrit ``dataset_metamodel.csv``."""
    dossier_plan = Path(plan).expanduser()
    points_path = dossier_plan / "points.csv"
    if not points_path.exists():
        raise FileNotFoundError(
            f"{points_path} introuvable : ce dossier n'est pas un plan produit par "
            "l'application.")

    points = pd.read_csv(points_path)
    manifeste = json.loads((dossier_plan / "manifest.json").read_text(encoding="utf-8"))
    decoupage = manifeste.get("nom_decoupage") or Path(manifeste["terrain_dir"]).name
    racine = Path(sorties_racine) if sorties_racine else chemins.sorties_gama()
    if not racine.exists():
        raise FileNotFoundError(
            f"Dossier de sorties GAMA introuvable : {racine}. Vérifie le chemin de "
            "l'installation MAELIA dans le panneau « Où sont les fichiers ».")

    # Par défaut, seules les sorties postérieures au plan comptent : sinon on
    # analyserait celles d'une exécution antérieure sans s'en apercevoir.
    if depuis is None:
        depuis = points_path.stat().st_mtime

    morceaux, dossiers, manquants = [], {}, []
    for run in sorted(points.run.unique()):
        dossier = _sorties_du_run(racine, decoupage, str(run), depuis)
        if dossier is None:
            manquants.append(str(run))
            continue
        resultat = _sorties_par_parcelle(dossier)
        resultat["run"] = run
        morceaux.append(resultat)
        dossiers[str(run)] = dossier

    if not morceaux:
        raise ValueError(
            f"Aucune sortie GAMA postérieure au plan dans {racine}. "
            f"Attendu des dossiers « {decoupage}_<run>_<date> ». "
            "GAMA a-t-il bien tourné, et sur ce plan-ci ?")

    resultats = pd.concat(morceaux, ignore_index=True)
    data = points.merge(resultats, on=["run", "parcelle"], how="inner")
    if data.empty:
        raise ValueError(
            "Aucun point du plan ne correspond aux sorties trouvées : les noms de "
            "parcelles diffèrent entre le plan et les sorties GAMA.")

    cible = dossier_plan / "dataset_metamodel.csv"
    data.to_csv(cible, index=False)
    return Collecte(dataset=cible, n_points=len(data), n_runs=len(dossiers),
                    dossiers=dossiers, manquants=manquants)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Assemble les sorties GAMA d'un plan en dataset_metamodel.csv.")
    parser.add_argument("plan", help="dossier du plan, dans simulations_app/")
    parser.add_argument("--sorties", default=None,
                        help="dossier de journaux GAMA (défaut : celui de MAELIA)")
    args = parser.parse_args(argv)

    try:
        resultat = collecter(args.plan, args.sorties)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  {exc}")
        return 1

    print(f"  {resultat.resume()}")
    for run, dossier in sorted(resultat.dossiers.items()):
        print(f"    {run} ← {dossier.name}")
    if resultat.manquants:
        print("  Relance GAMA pour les runs manquants, puis rappelle cette commande.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
