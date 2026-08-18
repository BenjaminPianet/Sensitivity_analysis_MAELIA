"""Séries climatiques des huit types de climat français, au format MAELIA.

`communication/types_climats_france.xlsx` définit une **typologie** — huit types
décrits par leurs propriétés — sans aucune série journalière. MAELIA, lui, a besoin
de `RRmm;Tmin;Tmax;ETP;RGI` par jour et par zone. Ce module comble l'écart en
téléchargeant la réanalyse ERA5 via Open-Meteo pour huit points représentatifs.

Choix explicites, à valider :

- **Points représentatifs.** Un point par type, choisi pour sa position canonique
  dans la typologie. Ce sont mes choix, pas ceux de la source ; ils sont vérifiables
  par les statistiques que `summarise()` recalcule sur les séries téléchargées.
- **Décalage des années.** Les campagnes simulées vont de 2019 à 2029, or la
  réanalyse s'arrête à l'année écoulée. On prend donc onze années réelles complètes
  (2014-2024) que l'on réétiquette en 2019-2029. Le 29 février est ajouté ou retiré
  selon que l'année cible est bissextile ou non — c'est le seul jour retouché.
- **Une ligne par jour.** Les fichiers de référence MAELIA en comptent trois par
  date, mais `zoneMeteo.lectureData` les range dans une map indexée par date : seule
  la dernière est conservée. Écrire une ligne par jour est donc équivalent.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Variables ERA5 et leur correspondance MAELIA.
DAILY_VARS = [
    "temperature_2m_min",             # -> Tmin  (°C)
    "temperature_2m_max",             # -> Tmax  (°C)
    "precipitation_sum",              # -> RRmm  (mm)
    "et0_fao_evapotranspiration",     # -> ETP   (mm)
    "shortwave_radiation_sum",        # -> RGI   (MJ/m²)
]

SOURCE_START_YEAR = 2014
TARGET_START_YEAR = 2019
N_YEARS = 11          # campagnes 2019-2029, cf. final_step du launcher
YEAR_SHIFT = TARGET_START_YEAR - SOURCE_START_YEAR


@dataclass(frozen=True)
class ClimateType:
    """Un type de la typologie, et le point retenu pour l'incarner."""

    code: int
    id_pdg: int          # identifiant de zone météo vu par MAELIA
    name: str
    site: str
    latitude: float
    longitude: float
    rationale: str


CLIMATE_TYPES: tuple[ClimateType, ...] = (
    ClimateType(1, 90001, "Climat de montagne", "Briançon", 44.90, 6.65,
                "Altitude 1300 m : température annuelle basse, nombreux jours de gel fort."),
    ClimateType(2, 90002, "Climat semi-continental et marges montagnardes", "Nancy", 48.69, 6.18,
                "Lorraine : hivers froids sans être montagnards, forte amplitude, position de transition."),
    ClimateType(3, 90003, "Climat océanique dégradé des plaines du Centre et du Nord", "Orléans", 47.90, 1.90,
                "Beauce : le climat du terrain actuel, précipitations faibles, été sec."),
    ClimateType(4, 90004, "Climat océanique altéré", "Poitiers", 46.58, 0.34,
                "Seuil du Poitou : transition entre océanique franc et dégradé."),
    ClimateType(5, 90005, "Climat océanique franc", "Brest", 48.39, -4.49,
                "Finistère : amplitude thermique minimale, pluies abondantes et fréquentes."),
    ClimateType(6, 90006, "Climat méditerranéen altéré", "Valence", 44.93, 4.89,
                "Vallée du Rhône : influence méditerranéenne atténuée, été chaud et sec."),
    ClimateType(7, 90007, "Climat du Bassin du Sud-Ouest", "Toulouse", 43.60, 1.44,
                "Bassin aquitain : température élevée, forte amplitude, pluies estivales orageuses."),
    ClimateType(8, 90008, "Climat méditerranéen franc", "Montpellier", 43.61, 3.88,
                "Littoral languedocien : été aride, fort rapport automne/été des précipitations."),
)

BY_ID = {c.id_pdg: c for c in CLIMATE_TYPES}


def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def fetch_site(climate: ClimateType, timeout: int = 90, retries: int = 5,
               cache_dir: str | Path | None = None) -> dict[str, list]:
    """Série journalière brute d'un point, sur la fenêtre source complète.

    L'API applique une limite de débit ; les réponses sont donc mises en cache sur
    disque et les réessais espacés progressivement, pour qu'une interruption ne
    coûte pas de retélécharger ce qui a déjà abouti.
    """
    cache = Path(cache_dir).expanduser() if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        cached = cache / f"{climate.id_pdg}_{climate.site}.json"
        if cached.exists():
            return json.loads(cached.read_text())["daily"]

    params = {
        "latitude": climate.latitude,
        "longitude": climate.longitude,
        "start_date": f"{SOURCE_START_YEAR}-01-01",
        "end_date": f"{SOURCE_START_YEAR + N_YEARS - 1}-12-31",
        "daily": ",".join(DAILY_VARS),
        "timezone": "UTC",
    }
    url = f"{ARCHIVE_URL}?{urllib.parse.urlencode(params)}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                payload = json.loads(response.read().decode())
            if cache:
                (cache / f"{climate.id_pdg}_{climate.site}.json").write_text(
                    json.dumps(payload), encoding="utf-8")
            return payload["daily"]
        except Exception as exc:  # noqa: BLE001
            last = exc
            # Une limite de débit demande une attente franche, pas un réessai immédiat.
            time.sleep(20 * (attempt + 1) if "429" in str(exc) else 3 * (attempt + 1))
    raise RuntimeError(f"Téléchargement impossible pour {climate.site} : {last}")


def to_maelia_rows(climate: ClimateType, daily: dict[str, list]) -> dict[int, list[str]]:
    """Convertit la série brute en lignes MAELIA, par année cible.

    Le réétiquetage décale l'année de cinq ans. Le 29 février est ajouté par
    duplication du 28 quand l'année cible est bissextile et la source ne l'est pas,
    et retiré dans le cas inverse.
    """
    by_source: dict[str, tuple] = {}
    for index, day in enumerate(daily["time"]):
        by_source[day] = tuple(daily[v][index] for v in DAILY_VARS)

    def value(source_day: date) -> tuple | None:
        return by_source.get(source_day.isoformat())

    rows: dict[int, list[str]] = {}
    for offset in range(N_YEARS):
        target_year = TARGET_START_YEAR + offset
        source_year = target_year - YEAR_SHIFT
        lines: list[str] = []

        current = date(target_year, 1, 1)
        while current.year == target_year:
            if current.month == 2 and current.day == 29 and not _is_leap(source_year):
                # L'année cible est bissextile, pas la source : on duplique le 28.
                sample = value(date(source_year, 2, 28))
            else:
                sample = value(date(source_year, current.month, current.day))
            if sample is not None and all(v is not None for v in sample):
                tmin, tmax, rr, etp, rgi = sample
                lines.append(
                    f"{climate.id_pdg};{current.strftime('%d/%m/%Y')};"
                    f"{rr:.3f};{tmin:.3f};{tmax:.3f};{etp:.3f};{rgi:.3f}"
                )
            current += timedelta(days=1)
        rows[target_year] = lines
    return rows


def build(output_dir: str | Path, climates=CLIMATE_TYPES,
          progress=None, pause: float = 8.0) -> dict[str, object]:
    """Télécharge les huit séries et écrit les fichiers météo au format MAELIA.

    Écrit ``<output_dir>/observee/<année>.csv``, une ligne par jour et par zone,
    toutes zones confondues dans le même fichier — la structure qu'attend
    ``zoneMeteo``. N'écrit rien dans l'installation MAELIA.
    """
    out = Path(output_dir).expanduser()
    (out / "observee").mkdir(parents=True, exist_ok=True)

    per_year: dict[int, list[str]] = {}
    raw: dict[int, dict] = {}
    cache_dir = out / "_cache"
    for index, climate in enumerate(climates):
        if progress:
            progress(f"{climate.site} ({climate.name})")
        if index:
            time.sleep(pause)  # l'API limite le débit ; on l'espace volontairement
        daily = fetch_site(climate, cache_dir=cache_dir)
        raw[climate.id_pdg] = daily
        for year, lines in to_maelia_rows(climate, daily).items():
            per_year.setdefault(year, []).extend(lines)

    header = "ID_PDG;DATE;RRmm;Tmin;Tmax;ETP;RGI"
    for year, lines in sorted(per_year.items()):
        (out / "observee" / f"{year}.csv").write_text(
            header + "\n" + "\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "source": "ERA5 via Open-Meteo (archive-api.open-meteo.com)",
        "licence": "CC-BY 4.0 — Copernicus / ECMWF ERA5",
        "telecharge_le": time.strftime("%Y-%m-%d %H:%M:%S"),
        "annees_source": [SOURCE_START_YEAR, SOURCE_START_YEAR + N_YEARS - 1],
        "annees_cible": [TARGET_START_YEAR, TARGET_START_YEAR + N_YEARS - 1],
        "decalage_annees": YEAR_SHIFT,
        "note_decalage":
            "La réanalyse s'arrête à l'année écoulée : onze années réelles sont "
            "réétiquetées sur les campagnes simulées. Seul le 29 février est retouché.",
        "note_lignes":
            "Une ligne par jour et par zone. Les fichiers de référence MAELIA en "
            "comptent trois par date, mais lectureData n'en conserve que la dernière.",
        "types": [
            {"code": c.code, "id_pdg": c.id_pdg, "nom": c.name, "site": c.site,
             "latitude": c.latitude, "longitude": c.longitude, "motif": c.rationale}
            for c in climates
        ],
    }
    (out / "climats_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output_dir": out, "manifest": manifest, "raw": raw}


def summarise(raw: dict[int, dict]) -> "object":
    """Statistiques annuelles par site, pour confronter les séries à la typologie."""
    import pandas as pd

    rows = []
    for id_pdg, daily in raw.items():
        climate = BY_ID[id_pdg]
        frame = pd.DataFrame({v: daily[v] for v in DAILY_VARS})
        frame["annee"] = pd.to_datetime(daily["time"]).year
        n_years = frame["annee"].nunique()
        tmoy = (frame.temperature_2m_min + frame.temperature_2m_max) / 2
        rows.append({
            "type": climate.code,
            "climat": climate.name,
            "site": climate.site,
            "T_moy": round(float(tmoy.mean()), 1),
            "j_Tmin<-5": round(float((frame.temperature_2m_min < -5).sum() / n_years), 1),
            "j_Tmax>30": round(float((frame.temperature_2m_max > 30).sum() / n_years), 1),
            "pluie_mm_an": round(float(frame.precipitation_sum.sum() / n_years)),
            "amplitude": round(float(
                frame.groupby(pd.to_datetime(daily["time"]).month)
                .apply(lambda g: (g.temperature_2m_min + g.temperature_2m_max).mean() / 2,
                       include_groups=False).agg(lambda s: s.max() - s.min())), 1),
        })
    return pd.DataFrame(rows).sort_values("type").reset_index(drop=True)
