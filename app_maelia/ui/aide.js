// Descriptions affichées au survol et au focus clavier.
//
// Le paramétrage d'une analyse de sensibilité manipule des notions qui ne sont pas
// évidentes — variable décrétée, fenêtre, stratification, Q². Plutôt que de supposer
// qu'elles vont de soi, chaque élément principal porte une explication en français,
// visible sans quitter la page.
//
// Un élément est documenté par un attribut `data-aide="clé"` ; la clé renvoie ici.

export const AIDES = {
  // ── Le choix fondateur ─────────────────────────────────────────────────────
  mode:
    "Deux façons de travailler. « Analyser des données existantes » part de "
    + "simulations déjà faites : l'espace exploré est alors imposé par le plan qui "
    + "les a produites, on peut le restreindre mais pas l'élargir. « Construire un "
    + "plan sur mesure » définit un espace libre, dont l'application tirera un plan "
    + "d'expérience à simuler ensuite avec GAMA.",
  donnees:
    "Le dossier doit contenir un fichier dataset_metamodel.csv. S'il contient aussi "
    + "un space_spec.json, celui-ci décrit exactement l'espace qui a produit ces "
    + "simulations et fait foi ; sinon l'application suppose le plan historique à "
    + "quatorze paramètres.",
  dossier:
    "Chemin d'un dossier de votre machine. Laissez vide pour utiliser le jeu "
    + "préchargé sélectionné au-dessus.",
  strates:
    "Deux caractéristiques du milieu, qui ne sont pas tirées au sort comme les autres "
    + "paramètres mais portées par l'îlot auquel appartient chaque parcelle — les "
    + "activer oblige donc l'application à construire un terrain.\n\n"
    + "Le climat est géographique : MAELIA retient la zone météo qui recouvre le plus "
    + "l'îlot, il faut donc huit emplacements séparés, avec les séries journalières "
    + "issues de la réanalyse ERA5.\n\n"
    + "Le sol est un attribut : l'îlot le désigne par son identifiant parmi les sols "
    + "de sa zone hydrographique. Rien ne se déplace, si bien que plusieurs îlots "
    + "peuvent être empilés au même endroit — même climat, sols différents. C'est ce "
    + "qui permet de croiser les deux : huit emplacements de trois îlots.",
  climat:
    "Huit types de climat français, séries météo journalières issues de la réanalyse "
    + "ERA5, placées à huit emplacements assez écartés pour que MAELIA les distingue.",
  sol:
    "Trois types de sol du terrain, contrastés sur les trois traits qui gouvernent le "
    + "bilan : la profondeur, l'argile et la matière organique. Le sol n'est pas lu "
    + "dans la géométrie mais dans l'attribut ID_SOL de l'îlot.",

  // ── Les analyses ───────────────────────────────────────────────────────────
  analyses:
    "Ce qui sera calculé. Les deux dernières sont nettement plus coûteuses que les "
    + "deux premières, HSIC étant quadratique en nombre de points : elles ne sont "
    + "donc pas cochées par défaut.",
  one_factor:
    "Pour chaque paramètre, la part de variance de la sortie que ses variations "
    + "expliquent à elles seules. Un paramètre conditionnel n'est évalué que sur les "
    + "simulations où il est actif.",
  metamodel_comparison:
    "Quatre familles de métamodèles entraînées sur le même partage entraînement/test. "
    + "Si elles s'accordent, la relation paramètres → sortie est robuste ; si elles "
    + "divergent, le résultat dépend du modèle choisi et doit être pris avec prudence. "
    + "La coche marque la famille retenue, sur le Q² pénalisé par le surapprentissage.",
  hsic:
    "Décompose la dépendance non linéaire entre paramètres et sortie, en séparant les "
    + "effets simples des interactions. Deux parts sont données : la contribution "
    + "globale, et la contribution intrinsèque, rapportée aux seules simulations où le "
    + "terme est actif — une variable rarement active peut peser peu globalement tout "
    + "en gouvernant son sous-espace.",
  pdp:
    "Pour chaque sous-espace — une combinaison fixée du nombre d'apports et de "
    + "préparations — on trace l'effet moyen de chaque paramètre. La structure de "
    + "l'itinéraire étant fixée, on y lit l'effet des réglages fins sans que la "
    + "variance due au nombre d'opérations vienne l'écraser.",
  reglages:
    "MAELIA et GAMA sont installés hors de l'application, à un endroit qui change "
    + "d'une machine à l'autre. Le chemin saisi ici l'emporte sur tout le reste et "
    + "prend effet aussitôt, sans redémarrage. Laissé vide, l'application essaie les "
    + "variables d'environnement MAELIA_ROOT et GAMA_HEADLESS, puis les emplacements "
    + "d'installation habituels. Un chemin inexistant est accepté et signalé : on peut "
    + "préparer sa configuration avant d'installer GAMA.",
  // ── Lecture des résultats ──────────────────────────────────────────────────
  couverture:
    "Combien de simulations tombent dans l'espace décrit, et ce qu'on peut en tirer. "
    + "Chaque analyse a son seuil : sous celui-ci elle est refusée plutôt que "
    + "produite à l'aveugle. Restreindre les niveaux coûte peu de points, resserrer "
    + "les bornes continues en fait perdre beaucoup.",
  plan:
    "Le plan d'expérience que produirait la sélection courante. Les variables figées "
    + "en sortent et deviennent des constantes ; les variables qu'aucun niveau retenu "
    + "n'active en sortent aussi. Le tirage fixe d'abord le squelette de l'itinéraire "
    + "— combien d'apports, combien de préparations — et donne à chacune des "
    + "combinaisons exactement la même part du budget, puis répartit régulièrement "
    + "chaque paramètre à l'intérieur. À graine égale, le même plan.",
  lancer: "Lance les analyses cochées sur l'espace décrit par l'arborescence.",
  reouvrir:
    "Réactive toutes les variables et remet chaque borne à son domaine complet.",

  // ── L'arborescence ─────────────────────────────────────────────────────────
  arborescence:
    "L'espace exploré. Les pastilles d'une variable ordinale choisissent les niveaux "
    + "retenus ; les champs d'une variable continue en fixent les bornes. Décocher "
    + "une variable la fige à une valeur unique.",
  inconditionnelle: "Cette variable est active dans toutes les simulations de l'espace.",
  decretee:
    "Cette variable n'existe que dans une partie de l'espace : elle dépend d'un "
    + "niveau d'une variable de décision. Elle n'est analysée que là où elle est active.",
  inatteignable:
    "Aucun niveau retenu n'active cette variable : elle ne fait plus partie de "
    + "l'espace et sort des analyses.",
  categorielle:
    "Variable sans ordre, donc explorée exhaustivement : toutes ses modalités sont "
    + "retenues. Une borne n'aurait pas de sens entre des modalités non ordonnées.",
  stratifiee:
    "Cette variable n'est pas tirée au sort : sa valeur est imposée par la parcelle "
    + "sur laquelle la simulation tourne.",
  resserree: "Les bornes de cette variable ont été réduites par rapport à son domaine.",
};

// ── Infobulle unique, partagée ───────────────────────────────────────────────
let bulle = null;
let minuterie = null;
let cible = null;

function noeud() {
  if (!bulle) {
    bulle = document.createElement("div");
    bulle.className = "aide-bulle";
    bulle.hidden = true;
    document.body.appendChild(bulle);
  }
  return bulle;
}

function montrer(element, immediat) {
  const texte = AIDES[element.dataset.aide];
  if (!texte) return;
  cible = element;
  clearTimeout(minuterie);
  minuterie = setTimeout(() => {
    const b = noeud();
    b.textContent = texte;
    b.hidden = false;
    const r = element.getBoundingClientRect();
    // La bulle suit l'élément, sans déborder de la fenêtre.
    b.style.left = `${Math.max(12, Math.min(r.left, window.innerWidth - 380))}px`;
    b.style.top = `${Math.min(r.bottom + 8, window.innerHeight - 140)}px`;
  }, immediat ? 60 : 450);
}

function cacher() {
  cible = null;
  clearTimeout(minuterie);
  if (bulle) bulle.hidden = true;
}

/** Rend documentés tous les éléments portant `data-aide`, y compris ceux injectés.
 *
 * Les pastilles de l'arborescence ne prennent pas le focus : l'arbre compte déjà
 * une case, deux bornes et plusieurs niveaux par variable, et doubler le nombre
 * d'arrêts de tabulation pour des marqueurs de statut gênerait plus qu'il n'aide.
 * Elles restent explicables au survol, et le texte du groupe reste accessible au
 * clavier depuis l'arbre lui-même.
 */
export function activerAides(racine = document) {
  racine.querySelectorAll("[data-aide]").forEach((el) => {
    if (!AIDES[el.dataset.aide]) return;
    el.classList.add("aidable");
    if (!el.classList.contains("badge") && !el.hasAttribute("tabindex")) {
      el.setAttribute("tabindex", "0");
    }
  });
}

export function installerAides() {
  activerAides();
  document.addEventListener("mouseover", (e) => {
    const el = e.target.closest("[data-aide]");
    if (el && el !== cible) montrer(el, false);
  });
  document.addEventListener("mouseout", (e) => {
    if (cible && !cible.contains(e.relatedTarget)) cacher();
  });
  // Le clavier ouvre sans délai : un utilisateur qui tabule cherche l'explication.
  document.addEventListener("focusin", (e) => {
    const el = e.target.closest("[data-aide]");
    if (el) montrer(el, true);
  });
  document.addEventListener("focusout", cacher);
  document.addEventListener("scroll", cacher, true);
}
