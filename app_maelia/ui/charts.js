// Figures de l'application, en SVG écrit à la main.
//
// Deux formes seulement, choisies par le travail que la donnée doit faire :
//
//   - **barres horizontales** pour comparer des magnitudes (R² par paramètre,
//     effet par modalité). La magnitude est portée par la longueur, donc une seule
//     teinte suffit : colorer les barres selon leur valeur serait redondant.
//   - **courbe d'emphase** pour les PDP : la moyenne en teinte d'accent, le faisceau
//     ICE en gris de retrait. Ce n'est pas deux séries de même statut — l'une est le
//     propos, l'autre le contexte.
//
// Une PDP catégorielle est un **nuage de points**, pas des barres. Les valeurs d'une
// PDP ne franchissent pas nécessairement zéro — les niveaux de carbone organique sont
// tous négatifs — et une barre partant d'un plancher arbitraire ferait lire « petit
// effet » là où la perte est la plus forte. Un point ne revendique aucune ligne de
// base ; seule sa position sur l'axe est lue.
//
// Le texte ne porte jamais la couleur de la donnée : les valeurs et étiquettes
// restent en encre, l'identité vient de la marque colorée à côté.

const ACCENT = "#2a9d8f";        // teinte d'accent de l'app, contraste ≥ 3:1 sur le panneau
const ACCENT_SOMBRE = "#1f756c";
const RETRAIT = "#b9c4c8";       // gris de retrait, pour le contexte

// Ordre d'un terme HSIC : effet simple, ou interaction de deux ou trois variables.
// Une seule teinte pour les effets simples, une seconde famille en deux nuances pour
// les interactions — l'ordre se lit ainsi comme une progression, pas comme trois
// catégories sans rapport. Palette contrôlée sur les six critères usuels : bande de
// clarté, plancher de chroma, séparation en vision déficiente, contraste au fond.
export const COULEURS_ORDRE = { 1: ACCENT, 2: "#c07818", 3: "#8a4f14" };
export const LIBELLES_ORDRE = {
  1: "effet simple", 2: "interaction d'ordre 2", 3: "interaction d'ordre 3",
};
const GRILLE = "#eaeff1";
const SURFACE = "#ffffff";

const ENCRE = "#243238";
const ENCRE_DOUCE = "#6f7b82";

function esc(v) {
  return String(v).replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

const nombre = (v, d = 2) => Number(v).toLocaleString("fr-FR",
  { minimumFractionDigits: d, maximumFractionDigits: d });

// ── Barres horizontales ──────────────────────────────────────────────────────
// Épaisseur plafonnée, extrémité arrondie côté donnée et carrée à la ligne de base,
// 2 px de surface entre deux barres. La valeur est posée à la pointe : c'est une
// étiquette par barre, pas par point, donc elle reste lisible.
export function barresHorizontales(lignes, options = {}) {
  const {
    largeur = 560, hauteurBarre = 18, ecart = 8, margeGauche = 150,
    margeDroite = 56, titre = "", format = (v) => nombre(v, 3),
    max = null, legende = null,
  } = options;

  if (!lignes.length) return `<p class="chart-vide">Aucune donnée à tracer.</p>`;

  const hautMax = max ?? Math.max(...lignes.map((l) => l.valeur), 0);
  const echelle = hautMax > 0 ? hautMax : 1;
  const largeurTrace = largeur - margeGauche - margeDroite;
  const hauteur = lignes.length * (hauteurBarre + ecart) + 26;

  // Graduations rondes, en retrait : elles portent les valeurs non étiquetées.
  const pas = echelle <= 0.25 ? 0.05 : echelle <= 0.6 ? 0.1 : echelle <= 1 ? 0.2
    : Math.pow(10, Math.floor(Math.log10(echelle)));
  const ticks = [];
  for (let v = 0; v <= echelle * 1.0001; v += pas) ticks.push(v);

  const grille = ticks.map((v) => {
    const x = margeGauche + (v / echelle) * largeurTrace;
    return `<line x1="${x}" y1="18" x2="${x}" y2="${hauteur - 8}" stroke="${GRILLE}" stroke-width="1"/>
      <text x="${x}" y="12" text-anchor="middle" font-size="9" fill="${ENCRE_DOUCE}">${format(v)}</text>`;
  }).join("");

  const barres = lignes.map((ligne, i) => {
    const y = 22 + i * (hauteurBarre + ecart);
    const w = Math.max(0, (ligne.valeur / echelle) * largeurTrace);
    // Extrémité arrondie de 4 px côté donnée, carrée à la base : un path, pas un rect.
    const r = Math.min(4, w);
    const d = `M ${margeGauche} ${y} H ${margeGauche + w - r} a ${r} ${r} 0 0 1 ${r} ${r}
               V ${y + hauteurBarre - r} a ${r} ${r} 0 0 1 ${-r} ${r} H ${margeGauche} Z`;
    return `<g class="barre" tabindex="0"
              data-info="${esc(ligne.etiquette)} — ${esc(format(ligne.valeur))}${ligne.detail ? " · " + esc(ligne.detail) : ""}">
        <rect x="0" y="${y - ecart / 2}" width="${largeur}" height="${hauteurBarre + ecart}" fill="transparent"/>
        <text x="${margeGauche - 8}" y="${y + hauteurBarre / 2 + 3}" text-anchor="end"
              font-size="11" fill="${ENCRE}">${esc(ligne.etiquette)}</text>
        <path d="${d}" fill="${ligne.couleur || ACCENT}"/>
        <text x="${margeGauche + w + 6}" y="${y + hauteurBarre / 2 + 3}"
              font-size="10.5" fill="${ENCRE_DOUCE}">${esc(format(ligne.valeur))}</text>
      </g>`;
  }).join("");

  // Dès que deux couleurs coexistent, la légende est obligatoire : la couleur seule
  // ne doit jamais être le seul porteur d'une distinction.
  const cartouche = legende && legende.length > 1
    ? `<div class="chart-legende">${legende.map((e) => `
        <span class="chart-cle"><span class="pastille" style="background:${e.couleur}"></span>${
          esc(e.libelle)}</span>`).join("")}</div>`
    : "";

  return `<figure class="chart">
    ${titre ? `<figcaption>${esc(titre)}</figcaption>` : ""}
    ${cartouche}
    <svg viewBox="0 0 ${largeur} ${hauteur}" role="img" preserveAspectRatio="xMinYMin meet"
         aria-label="${esc(titre)}">
      ${grille}${barres}
    </svg>
  </figure>`;
}

// ── Courbe PDP avec faisceau ICE ─────────────────────────────────────────────
// Emphase : la moyenne porte l'accent en 2 px, les trajectoires individuelles sont
// en gris de retrait et fines. Une catégorielle n'est pas interpolée — il n'y a rien
// entre deux modalités — et ses points ne sont pas reliés.
export function courbePdp(courbe, options = {}) {
  const { largeur = 260, hauteur = 150, titre = "" } = options;
  const marge = { haut: 16, droite: 10, bas: 26, gauche: 42 };
  const l = largeur - marge.gauche - marge.droite;
  const h = hauteur - marge.haut - marge.bas;

  const toutes = [...courbe.pdp, ...(courbe.ice || []).flat()];
  const yMin = Math.min(...toutes), yMax = Math.max(...toutes);
  const etendue = yMax - yMin || 1;
  const Y = (v) => marge.haut + h - ((v - yMin) / etendue) * h;

  const categorielle = courbe.nature === "catégorielle";
  const n = courbe.grille.length;
  const X = (i) => marge.gauche + (n === 1 ? l / 2 : (i / (n - 1)) * l);

  const axes = `
    <line x1="${marge.gauche}" y1="${marge.haut}" x2="${marge.gauche}" y2="${marge.haut + h}"
          stroke="${GRILLE}" stroke-width="1"/>
    <line x1="${marge.gauche}" y1="${marge.haut + h}" x2="${marge.gauche + l}" y2="${marge.haut + h}"
          stroke="${GRILLE}" stroke-width="1"/>
    <text x="${marge.gauche - 5}" y="${Y(yMax) + 4}" text-anchor="end" font-size="9"
          fill="${ENCRE_DOUCE}">${esc(nombre(yMax, 0))}</text>
    <text x="${marge.gauche - 5}" y="${Y(yMin) + 4}" text-anchor="end" font-size="9"
          fill="${ENCRE_DOUCE}">${esc(nombre(yMin, 0))}</text>`;

  let marques;
  if (categorielle) {
    const etiquettes = courbe.modalites || courbe.grille;
    const pas = l / Math.max(1, n);
    marques = courbe.pdp.map((v, i) => {
      const x = marge.gauche + pas * (i + 0.5);
      const y = Y(v);
      // Repère vertical fin : il situe le point sur l'axe sans prétendre mesurer
      // une hauteur depuis zéro. Le point porte la valeur, pas le trait.
      return `<g class="marque" tabindex="0"
                data-info="${esc(etiquettes[i])} — ${esc(nombre(v, 1))}">
          <line x1="${x}" y1="${marge.haut}" x2="${x}" y2="${marge.haut + h}"
                stroke="${GRILLE}" stroke-width="1"/>
          <circle cx="${x}" cy="${y}" r="4" fill="${ACCENT}"
                  stroke="${SURFACE}" stroke-width="2"/>
        </g>`;
    }).join("");
  } else {
    const chemin = (serie) => serie.map((v, i) => `${i ? "L" : "M"} ${X(i)} ${Y(v)}`).join(" ");
    const faisceau = (courbe.ice || []).map(
      (t) => `<path d="${chemin(t)}" fill="none" stroke="${RETRAIT}" stroke-width="1"
                opacity="0.5" stroke-linejoin="round" stroke-linecap="round"/>`).join("");
    // Marqueur d'extrémité : 8 px, avec anneau de surface de 2 px.
    const fin = courbe.pdp.length - 1;
    marques = `${faisceau}
      <path d="${chemin(courbe.pdp)}" fill="none" stroke="${ACCENT}" stroke-width="2"
            stroke-linejoin="round" stroke-linecap="round"/>
      <circle cx="${X(fin)}" cy="${Y(courbe.pdp[fin])}" r="4"
              fill="${ACCENT_SOMBRE}" stroke="${SURFACE}" stroke-width="2"/>
      ${courbe.grille.map((g, i) => `<g class="marque" tabindex="0"
          data-info="${esc(courbe.variable)} = ${esc(nombre(g, 1))} — ${esc(nombre(courbe.pdp[i], 1))}">
          <rect x="${X(i) - l / (2 * n)}" y="${marge.haut}" width="${l / n}" height="${h}"
                fill="transparent"/></g>`).join("")}`;
  }

  const bornes = categorielle
    ? `${esc((courbe.modalites || [])[0] || "")} → ${esc((courbe.modalites || []).slice(-1)[0] || "")}`
    : `${esc(nombre(courbe.grille[0], 0))} → ${esc(nombre(courbe.grille[n - 1], 0))}`;

  return `<figure class="chart chart-pdp">
    <figcaption>${esc(titre || courbe.variable)}
      <span class="chart-note">amplitude ${esc(nombre(courbe.amplitude, 1))}</span></figcaption>
    <svg viewBox="0 0 ${largeur} ${hauteur}" role="img" preserveAspectRatio="xMinYMin meet"
         aria-label="PDP de ${esc(courbe.variable)}">
      ${axes}${marques}
      <text x="${marge.gauche}" y="${hauteur - 8}" font-size="9" fill="${ENCRE_DOUCE}">${bornes}</text>
    </svg>
  </figure>`;
}

// Légende de l'emphase : deux rôles, donc elle est nécessaire — mais elle nomme un
// propos et un contexte, pas deux séries de même statut.
export function legendePdp() {
  return `<p class="chart-legende">
    <span class="cle"><svg width="18" height="8"><line x1="1" y1="4" x2="17" y2="4"
      stroke="${ACCENT}" stroke-width="2" stroke-linecap="round"/></svg> moyenne PDP</span>
    <span class="cle"><svg width="18" height="8"><line x1="1" y1="2" x2="17" y2="6"
      stroke="${RETRAIT}" stroke-width="1"/><line x1="1" y1="6" x2="17" y2="3"
      stroke="${RETRAIT}" stroke-width="1"/></svg> trajectoires individuelles (ICE)</span>
  </p>`;
}

// ── Infobulle partagée ───────────────────────────────────────────────────────
// Une figure HTML est interactive par nature : survol et focus clavier révèlent la
// valeur exacte, ce qui dispense d'étiqueter chaque point.
export function activerInfobulles(racine) {
  let bulle = document.getElementById("chart-tooltip");
  if (!bulle) {
    bulle = document.createElement("div");
    bulle.id = "chart-tooltip";
    bulle.className = "chart-tooltip";
    bulle.hidden = true;
    document.body.appendChild(bulle);
  }
  const montrer = (cible) => {
    const info = cible.getAttribute("data-info");
    if (!info) return;
    bulle.textContent = info;
    bulle.hidden = false;
    const r = cible.getBoundingClientRect();
    bulle.style.left = `${Math.min(r.left, window.innerWidth - 240)}px`;
    bulle.style.top = `${Math.max(8, r.top - 30)}px`;
  };
  const cacher = () => { bulle.hidden = true; };

  racine.addEventListener("mouseover", (e) => {
    const c = e.target.closest("[data-info]");
    if (c) montrer(c);
  });
  racine.addEventListener("mouseout", cacher);
  racine.addEventListener("focusin", (e) => {
    const c = e.target.closest("[data-info]");
    if (c) montrer(c);
  });
  racine.addEventListener("focusout", cacher);
}
