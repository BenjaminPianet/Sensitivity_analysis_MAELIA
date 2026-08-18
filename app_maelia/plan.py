"""Construction d'un espace de conception SMT depuis une spécification.

Remplace le ``build_design_space()`` codé en dur de ``simulations/run_terrainSA_batch.py``
par une boucle pilotée par la spécification. La correspondance est directe :

  continue, fenêtre non dégénérée   -> FloatVariable(fenêtre, convertie par ``scale``)
  ordinale, au moins deux niveaux   -> OrdinalVariable(étiquettes des niveaux retenus)
  catégorielle                      -> CategoricalVariable(domaine complet)
  fenêtre dégénérée                 -> constante : la variable sort du plan

puis un ``declare_decreed_var`` par variable décrétée. Les variables devenues
inconditionnelles parce que la borne basse les active ne sont pas déclarées décrétées.

Les fenêtres sont exprimées dans l'unité agronomique ; ``scale`` porte la conversion
vers la variable de plan (les doses sont saisies en kilos et échantillonnées en log10).

L'espace ADSG sert à décrire la hiérarchie et à la faire vérifier par SMT ; il ne sert
plus à tirer les points. ``sample()`` stratifie par sous-espace et tire un hypercube
latin à l'intérieur — voir sa documentation pour ce que cela change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .space import CATEGORICAL, CONTINUOUS, ORDINAL, SpaceSpec


@dataclass(frozen=True)
class BuiltPlan:
    """Espace SMT construit, avec de quoi le relire sans fouiller l'objet SMT."""

    spec: SpaceSpec
    design_space: object
    design_names: tuple[str, ...]          # variables réellement dans le plan, dans l'ordre
    constants: dict[str, float | str]      # variables figées, hors plan
    unreachable: tuple[str, ...]           # variables qu'aucun niveau retenu n'active
    stratified: tuple[str, ...]            # variables imposées par le terrain, hors tirage
    decreed: tuple[tuple[str, str, tuple[str, ...]], ...]  # (variable, méta, étiquettes)

    def index_of(self, name: str) -> int:
        return self.design_names.index(name)


def _to_plan_scale(value: float, scale: str | None) -> float:
    """Convertit une valeur agronomique en valeur de plan."""
    if scale is None:
        return float(value)
    if scale == "log10":
        if value <= 0:
            raise ValueError(f"échelle log10 impossible pour une valeur ≤ 0 ({value})")
        return math.log10(value)
    raise ValueError(f"échelle inconnue : {scale}")


def to_agronomic_scale(value: float, scale: str | None) -> float:
    """Réciproque : valeur de plan -> valeur agronomique."""
    if scale is None:
        return float(value)
    if scale == "log10":
        return float(10.0 ** value)
    raise ValueError(f"échelle inconnue : {scale}")


def build_design_space(spec: SpaceSpec) -> BuiltPlan:
    """Construit l'espace ADSG hiérarchique décrit par la spécification."""
    from smt.design_space import CategoricalVariable, FloatVariable, OrdinalVariable
    from smt_design_space_ext import AdsgDesignSpaceImpl

    design_variables = []
    design_names: list[str] = []
    constants: dict[str, float | str] = {}

    # Une variable qu'aucun niveau retenu n'active n'existe pas dans cet espace : elle
    # ne doit ni être tirée, ni apparaître comme constante. Le générateur d'itinéraires
    # ne la lira jamais, ses valeurs par défaut suffisent.
    reachable = spec.reachable()
    unreachable = tuple(v.name for v in spec.variables if v.name not in reachable)
    stratified = tuple(v.name for v in spec.variables if v.stratified and v.name in reachable)

    # Les méta-variables viennent en tête, dans l'ordre de la spécification : c'est
    # cet ordre qui fixe les indices attendus par declare_decreed_var.
    for meta in spec.meta_variables:
        if meta.is_frozen:
            only = meta.levels_in_window()[0]
            constants[meta.name] = only.tag
            continue
        tags = [lv.tag for lv in meta.levels_in_window()]
        design_variables.append(OrdinalVariable(tags))
        design_names.append(meta.name)

    for var in spec.variables:
        if var.name in unreachable:
            continue

        if var.stratified:
            # Imposée par le terrain : elle ne fait pas partie des variables tirées.
            # Sa valeur vient de la parcelle sur laquelle le point sera exécuté.
            continue

        if var.kind == CATEGORICAL:
            if len(var.domain) == 1:
                constants[var.name] = var.domain[0]
                continue
            design_variables.append(CategoricalVariable(list(var.domain)))
            design_names.append(var.name)
            continue

        if var.is_frozen:
            # Fenêtre réduite à un point : plus une variable de conception, une constante
            # du générateur d'itinéraires — au même titre que le type d'engrais.
            window = var.effective_window
            constants[var.name] = float(window[0]) if var.kind == CONTINUOUS else window[0]
            continue

        if var.kind == CONTINUOUS:
            lo, hi = var.effective_window
            design_variables.append(
                FloatVariable(_to_plan_scale(lo, var.scale), _to_plan_scale(hi, var.scale))
            )
        else:  # ordinale non-méta
            design_variables.append(OrdinalVariable([str(v) for v in var.effective_window]))
        design_names.append(var.name)

    ds = AdsgDesignSpaceImpl(design_variables=design_variables)

    # Une variable n'est décrétée que si elle est atteignable sans être inconditionnelle.
    decreed_names = spec.decreed()
    declarations: list[tuple[str, str, tuple[str, ...]]] = []
    for name in design_names:
        if name not in decreed_names:
            continue
        for meta in spec.meta_variables:
            if meta.name not in design_names:
                continue  # méta figée : la condition est déjà tranchée
            activating = [lv.tag for lv in meta.levels_in_window() if name in lv.activates]
            if not activating or len(activating) == len(meta.levels_in_window()):
                continue  # non gouvernée par cette méta, ou activée par tous ses niveaux
            ds.declare_decreed_var(
                decreed_var=design_names.index(name),
                meta_var=design_names.index(meta.name),
                meta_value=activating,
            )
            declarations.append((name, meta.name, tuple(activating)))

    return BuiltPlan(
        spec=spec,
        design_space=ds,
        design_names=tuple(design_names),
        constants=constants,
        unreachable=unreachable,
        stratified=stratified,
        decreed=tuple(declarations),
    )


# Critère de l'hypercube latin. « maximin » tire un point au hasard **dans** chaque
# case, puis maximise la distance minimale entre points. Deux variantes ont été
# écartées, chacune pour une raison mesurée sur un plan de 2400 points :
#
#   « ese »           marges équivalentes, pour un coût d'optimisation par
#                     sous-espace sans commune mesure avec celui retenu ici.
#   « centermaximin » place les points au centre de leur case. Les marges y sont à
#                     peine plus régulières (0,0025 contre 0,0031, quand le tirage
#                     i.i.d. donnait 0,0223), mais les valeurs tombent sur une grille
#                     fixe : 200 valeurs distinctes répétées douze fois au lieu de
#                     2400. Une grille identique d'une graine à l'autre interdit
#                     d'explorer l'intervalle entre deux points, et deux plans
#                     fusionnés n'apporteraient aucune valeur nouvelle.
CRITERE_LHS = "maximin"


def _colonnes(plan: BuiltPlan) -> list[tuple[str, object]]:
    """Nature de chaque colonne du plan, dans l'ordre de ``design_names``.

    L'encodage attendu par ``decode_values`` diffère selon la nature : un indice
    entier pour les niveaux, un flottant dans l'échelle du plan pour les continues.
    """
    metas = {m.name: m for m in plan.spec.meta_variables}
    variables = {v.name: v for v in plan.spec.variables}
    colonnes: list[tuple[str, object]] = []
    for nom in plan.design_names:
        if nom in metas:
            colonnes.append(("meta", metas[nom]))
        else:
            var = variables[nom]
            colonnes.append((var.kind if var.kind == CONTINUOUS else "niveaux", var))
    return colonnes


def _active_dans(plan: BuiltPlan, nom: str, sous_espace: dict) -> bool:
    """La variable est-elle active sous cette combinaison de niveaux ?"""
    var = next((v for v in plan.spec.variables if v.name == nom), None)
    if var is None or var.always_active:
        return True
    gouvernantes = [m for m in plan.spec.meta_variables
                    if any(nom in lv.activates for lv in m.levels)]
    if not gouvernantes:
        return True
    return all(
        any(lv.value == sous_espace.get(m.name) and nom in lv.activates for lv in m.levels)
        for m in gouvernantes
    )


def sample(plan: BuiltPlan, n_points: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Plan stratifié par sous-espace, hypercube latin à l'intérieur.

    Le squelette de l'itinéraire — un niveau par méta-variable — est fixé d'abord, et
    chaque combinaison reçoit **exactement** la même part du budget. À l'intérieur
    d'une combinaison la dimension ne varie plus : un hypercube latin y est licite, et
    répartit chaque variable régulièrement sur toute sa plage.

    C'est le seul mode de tirage. Le tirage ADSG qu'il remplace était du Monte-Carlo
    i.i.d. — la boucle ``_sample_valid_x`` appelle *n* fois ``get_random_design_vector``
    sans critère de remplissage, ignore la graine reçue, et laisse au hasard le nombre
    de points par sous-espace. Mesuré sur cet espace : marges 5 fois plus éloignées de
    l'uniforme, effectifs allant de 171 à 222 pour 200 attendus, et aucune
    reproductibilité. Un hypercube latin passé à ``correct_get_acting`` ne corrige
    rien : cette méthode convertit les dimensions discrètes par ``int()``, une
    troncature qui fait disparaître le niveau supérieur de chaque méta-variable.

    Les variables inactives gardent le milieu de leur domaine, valeur qu'écrivait déjà
    l'ADSG — dose 10 kg N, profondeur 13 cm — afin que les plans restent comparables.
    """
    from smt.sampling_methods import LHS

    colonnes = _colonnes(plan)
    noms = list(plan.design_names)
    # Seules les méta-variables non figées structurent le plan : une méta figée est
    # déjà une constante, et ne découpe donc plus l'espace.
    metas = [m for m in plan.spec.meta_variables if m.name in noms]
    sous_espaces = plan.spec.subspaces() or [{}]

    base, reste = divmod(int(n_points), len(sous_espaces))
    if base == 0:
        raise ValueError(
            f"{n_points} points pour {len(sous_espaces)} sous-espaces : il en faut au "
            f"moins un par sous-espace, soit {len(sous_espaces)}.")

    # Une graine indépendante par sous-espace, dérivée plutôt qu'additionnée : avec
    # `seed + rang`, deux plans de graines voisines partageraient la plupart de leurs
    # graines de sous-espace, et donc des tirages entiers là où les formes coïncident.
    graines = [int(g) for g in np.random.SeedSequence(seed).generate_state(
        len(sous_espaces), dtype=np.uint32) % (2 ** 31 - 1)]

    lignes: list[np.ndarray] = []
    actifs: list[np.ndarray] = []
    for rang, sous_espace in enumerate(sous_espaces):
        n = base + (1 if rang < reste else 0)

        acting = np.array([nom in {m.name for m in metas}
                           or _active_dans(plan, nom, sous_espace) for nom in noms])
        # Les colonnes à tirer ici : tout ce qui est actif et n'est pas un niveau imposé.
        tirables = [j for j, (nature, _) in enumerate(colonnes)
                    if nature != "meta" and acting[j]]

        bloc = np.zeros((n, len(noms)))
        if tirables:
            bornes = np.array([_bornes_colonne(colonnes[j]) for j in tirables], dtype=float)
            # Une graine par sous-espace : le plan entier est reproductible.
            # Les critères de type maximin comparent des distances entre points : avec
            # un seul point par sous-espace il n'y a pas de paire, et SMT échoue sur un
            # tableau vide. Un hypercube latin sans critère y suffit, n'ayant rien à
            # optimiser.
            criterion = CRITERE_LHS if n > 1 else "center"
            unites = LHS(xlimits=bornes, criterion=criterion,
                         seed=graines[rang])(n)
            for k, j in enumerate(tirables):
                bloc[:, j] = _encoder(colonnes[j], unites[:, k])

        for j, (nature, objet) in enumerate(colonnes):
            if nature == "meta":
                niveaux = objet.levels_in_window()
                valeur = sous_espace.get(objet.name, niveaux[0].value)
                bloc[:, j] = next(i for i, lv in enumerate(niveaux) if lv.value == valeur)
            elif j not in tirables:
                # Inactive : le milieu du domaine, comme l'ADSG l'écrivait.
                lo, hi = _bornes_colonne(colonnes[j])
                bloc[:, j] = (lo + hi) / 2 if nature == CONTINUOUS else round((lo + hi) / 2)

        lignes.append(bloc)
        actifs.append(np.tile(acting, (n, 1)))

    xt = np.vstack(lignes)
    is_acting = np.vstack(actifs)

    # Mélange indispensable : le point d'indice i est exécuté sur la parcelle i modulo
    # le nombre de parcelles. Laissé en ordre de sous-espace, le plan alignerait
    # exactement les sous-espaces sur les îlots, et donc sur les climats.
    ordre = np.random.default_rng(seed).permutation(len(xt))
    return xt[ordre], is_acting[ordre]


def _niveaux(objet) -> list:
    """Niveaux retenus d'une colonne discrète : le domaine entier si catégorielle."""
    return list(objet.domain if objet.kind == CATEGORICAL else objet.effective_window)


def _bornes_colonne(colonne: tuple[str, object]) -> tuple[float, float]:
    """Bornes de tirage d'une colonne, dans l'encodage attendu par SMT.

    Une colonne discrète à *k* niveaux se tire sur ``[0, k)`` et non sur ``[0, k-1]``,
    de sorte que les *k* intervalles unitaires soient de même largeur. Sur ``[0, k-1]``
    un arrondi donnerait aux deux niveaux extrêmes la moitié du poids des autres, et
    une troncature ferait disparaître le dernier.
    """
    nature, objet = colonne
    if nature == CONTINUOUS:
        lo, hi = objet.effective_window
        return _to_plan_scale(lo, objet.scale), _to_plan_scale(hi, objet.scale)
    return 0.0, float(len(_niveaux(objet)))


def _encoder(colonne: tuple[str, object], valeurs: np.ndarray) -> np.ndarray:
    """Convertit le tirage continu vers l'encodage de la colonne."""
    nature, objet = colonne
    if nature == CONTINUOUS:
        return valeurs
    # La borne haute est exclue par construction ; on la ramène au dernier niveau.
    return np.floor(valeurs).clip(0, len(_niveaux(objet)) - 1)


def decode(plan: BuiltPlan, xt: np.ndarray) -> list[dict]:
    """Décode les points en dictionnaires nom -> valeur agronomique.

    Les méta-variables sont rendues sous leur valeur numérique (celle des données,
    pas l'étiquette SMT), et les échelles sont inversées : une dose échantillonnée
    en log10 ressort en kilos.

    Les variables **figées** ne sont pas tirées mais font partie de l'itinéraire :
    elles sont réinjectées ici. Sans cela, un consommateur en aval lirait une valeur
    absente et retomberait sur un défaut — une méta-variable figée à « 3 apports »
    deviendrait silencieusement « 0 apport ».
    """
    by_tag = {
        meta.name: {lv.tag: lv.value for lv in meta.levels}
        for meta in plan.spec.meta_variables
    }
    # Seules les continues portent une échelle : convertir une catégorielle
    # ferait échouer le décodage sur une valeur non numérique.
    scales = {v.name: v.scale for v in plan.spec.variables if v.kind == CONTINUOUS}

    frozen = {
        name: by_tag[name].get(value, value) if name in by_tag else value
        for name, value in plan.constants.items()
    }

    rows = []
    for raw in plan.design_space.decode_values(xt):
        values = list(raw.values()) if isinstance(raw, dict) else list(raw)
        row = dict(frozen)
        for name, value in zip(plan.design_names, values):
            if name in by_tag:
                row[name] = by_tag[name].get(value, value)
            elif name in scales:
                row[name] = to_agronomic_scale(float(value), scales[name])
            else:
                row[name] = value
        rows.append(row)
    return rows
