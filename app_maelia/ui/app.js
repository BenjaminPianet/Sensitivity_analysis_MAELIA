import { activerInfobulles, barresHorizontales, courbePdp, legendePdp,
         COULEURS_ORDRE, LIBELLES_ORDRE } from "/ui/static/charts.js?v=18";
import { activerAides, installerAides } from "/ui/static/aide.js?v=18";

// Interface v2 : arborescence tri-état pour construire l'espace hiérarchique.
// Ni framework ni étape de build. L'état vit dans `windows` ; tout le reste (statut
// des variables, grisage, couverture) en est dérivé, comme côté Python.

const treeEl = document.getElementById("tree");
const resultsEl = document.getElementById("results");
const runBtn = document.getElementById("run-btn");
const resetBtn = document.getElementById("reset-btn");

let spec = null;
// L'état de la demande : le mode, la source des données, le climat. Trois valeurs
// suffisent à décrire ce que l'utilisateur veut — le reste s'en déduit.
let mode = "donnees";
let dossierDonnees = "";
let avecClimat = false;
let avecSol = false;
// windows : nom -> fenêtre. Continue [min, max], ordinale [niveaux retenus].
let windows = {};
// Valeur de gel d'une variable décochée, mémorisée pour la restituer si on la recoche.
let frozenAt = {};
let coverageTimer = null;

// Analyses demandées. Rien n'est lancé implicitement : les analyses coûteuses
// doivent être cochées, pour que l'utilisateur sache ce qu'il déclenche.
function analysesChoisies() {
  return [...document.querySelectorAll('.analyses input[type="checkbox"]:checked')]
    .map((c) => c.value);
}

const fmt = (n) => Number(n).toLocaleString("fr-FR");

// Identifiant sans accent d'un statut, pour la classe CSS et la clé d'aide.
const nombreCourt = (v) => (typeof v === "number" ? v.toFixed(3) : "—");

const sansAccent = (mot) => mot.normalize("NFD").replace(/[\u0300-\u036f]/g, "");

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}

// ── Dérivation, miroir de space.py ───────────────────────────────────────────
// Refaite côté navigateur pour que le grisage soit instantané : le serveur reste
// la référence, mais on n'attend pas son retour pour redessiner l'arbre.
function selectedLevels(meta) {
  return windows[meta.name] || meta.window;
}

function statusOf(variable) {
  if (variable.always_active) return "inconditionnelle";
  const governing = Object.entries(variable.activated_by || {});
  if (!governing.length) return "inconditionnelle";

  let unconditional = true;
  for (const [metaName, activatingLevels] of governing) {
    const meta = spec.meta_variables.find((m) => m.name === metaName);
    const retained = selectedLevels(meta);
    const activated = retained.filter((lv) => activatingLevels.includes(lv));
    if (!activated.length) return "inatteignable";
    if (activated.length < retained.length) unconditional = false;
  }
  return unconditional ? "inconditionnelle" : "décrétée";
}

function isFrozen(variable) {
  if (variable.kind === "categorical") return false;
  const w = windows[variable.name] || variable.window;
  return w.length === 2 && w[0] === w[1];
}

// ── Rendu de l'arborescence ──────────────────────────────────────────────────
function variableRow(variable) {
  const status = statusOf(variable);
  const unreachable = status === "inatteignable";

  // Une catégorielle n'a ni bornes ni gel : elle est explorée exhaustivement, et son
  // domaine n'est pas réglable. On montre ses modalités, sans contrôle.
  if (variable.kind === "categorical") {
    const modalites = (variable.domain || []).map(escapeHtml).join(" · ");
    return `
      <li role="treeitem" class="node var ${unreachable ? "unreachable" : ""}">
        <label class="node-label">
          <input type="checkbox" checked disabled aria-label="${escapeHtml(variable.label)}">
          <span class="node-name">${escapeHtml(variable.label)}</span>
        </label>
        <span class="frozen-note">${variable.domain.length} modalités, toutes</span>
        <span class="badges">
          <span class="badge categorielle" data-aide="categorielle">catégorielle</span>
          ${variable.stratified
            ? '<span class="badge narrowed" data-aide="stratifiee">imposée par le terrain</span>'
            : ""}
          <span class="domain">${modalites}</span>
        </span>
      </li>`;
  }

  const frozen = isFrozen(variable);
  const w = windows[variable.name] || variable.window;
  const [dlo, dhi] = variable.domain;
  const narrowed = !unreachable && !frozen && (w[0] !== dlo || w[1] !== dhi);
  const unit = variable.unit ? ` <span class="unit">${escapeHtml(variable.unit)}</span>` : "";

  const controls = unreachable
    ? `<span class="frozen-note">hors de l'espace</span>`
    : frozen
      ? `<label class="frozen-input">figée à
           <input type="number" step="any" data-var="${variable.name}" data-role="frozen"
                  value="${w[0]}" min="${dlo}" max="${dhi}"></label>`
      : `<span class="range">
           <input type="number" step="any" data-var="${variable.name}" data-role="min"
                  value="${w[0]}" min="${dlo}" max="${dhi}" aria-label="borne basse ${escapeHtml(variable.label)}">
           <span class="arrows">⟷</span>
           <input type="number" step="any" data-var="${variable.name}" data-role="max"
                  value="${w[1]}" min="${dlo}" max="${dhi}" aria-label="borne haute ${escapeHtml(variable.label)}">
         </span>`;

  return `
    <li role="treeitem" class="node var ${unreachable ? "unreachable" : ""}">
      <label class="node-label">
        <input type="checkbox" data-var="${variable.name}" data-role="explore"
               ${frozen || unreachable ? "" : "checked"} ${unreachable ? "disabled" : ""}>
        <span class="node-name">${escapeHtml(variable.label)}${unit}</span>
      </label>
      ${controls}
      <span class="badges">
        <span class="badge status-${sansAccent(status)}"
              data-aide="${sansAccent(status)}">${status}</span>
        ${narrowed ? '<span class="badge narrowed" data-aide="resserree">resserrée</span>' : ""}
        <span class="domain">sur [${dlo}, ${dhi}]</span>
      </span>
    </li>`;
}

function metaNode(meta) {
  const retained = selectedLevels(meta);
  const all = meta.domain;
  const state = retained.length === all.length ? "checked"
              : retained.length === 0 ? "" : "indeterminate";

  // Les variables gouvernées par cette méta, montrées une seule fois.
  const governed = spec.variables.filter(
    (v) => !v.always_active && (v.activated_by || {})[meta.name] !== undefined
  );

  const levelPills = all.map((lv) => {
    const level = meta.levels.find((l) => l.value === lv);
    const on = retained.includes(lv);
    return `<button type="button" class="level ${on ? "on" : ""}"
              data-meta="${meta.name}" data-level="${lv}"
              aria-pressed="${on}">${escapeHtml(level.label)}</button>`;
  }).join("");

  const lo = retained.length ? Math.min(...retained) : "—";
  const hi = retained.length ? Math.max(...retained) : "—";

  return `
    <li role="treeitem" class="node group">
      <div class="group-head">
        <label class="node-label">
          <input type="checkbox" data-meta="${meta.name}" data-role="meta-all" ${state === "checked" ? "checked" : ""}>
          <span class="node-name">${escapeHtml(meta.label)}</span>
        </label>
        <span class="badges">
          <span class="badge ordinal">ordinale</span>
          <span class="domain">niveaux [${lo} ⟷ ${hi}] sur [${Math.min(...all)}, ${Math.max(...all)}]</span>
        </span>
      </div>
      <div class="levels">${levelPills}</div>
      ${governed.length ? `<ul role="group" class="children">${governed.map(variableRow).join("")}</ul>` : ""}
    </li>`;
}

function renderTree() {
  const always = spec.variables.filter((v) => v.always_active);
  const free = spec.variables.filter(
    (v) => !v.always_active && Object.keys(v.activated_by || {}).length === 0
  );

  treeEl.innerHTML = `
    <ul role="group" class="root" data-aide="arborescence">
      <li role="treeitem" class="node group">
        <div class="group-head">
          <span class="node-name always">Toujours explorées</span>
        </div>
        <ul role="group" class="children">${[...always, ...free].map(variableRow).join("")}</ul>
      </li>
      ${spec.meta_variables.map(metaNode).join("")}
    </ul>`;

  // L'état « partiellement coché » n'existe qu'en propriété DOM, pas en attribut.
  spec.meta_variables.forEach((meta) => {
    const box = treeEl.querySelector(`input[data-meta="${meta.name}"][data-role="meta-all"]`);
    if (!box) return;
    const retained = selectedLevels(meta);
    box.indeterminate = retained.length > 0 && retained.length < meta.domain.length;
  });
}

// ── Interactions ─────────────────────────────────────────────────────────────
function midpoint(variable) {
  const w = windows[variable.name] || variable.window;
  return Math.round(((w[0] + w[1]) / 2) * 1000) / 1000;
}

treeEl.addEventListener("click", (event) => {
  const level = event.target.closest(".level");
  if (!level) return;
  const meta = spec.meta_variables.find((m) => m.name === level.dataset.meta);
  const value = Number(level.dataset.level);
  const retained = new Set(selectedLevels(meta));
  if (retained.has(value)) retained.delete(value);
  else retained.add(value);
  if (retained.size === 0) return; // un espace sans aucun niveau n'a pas de sens
  windows[meta.name] = [...retained].sort((a, b) => a - b);
  refresh();
});

treeEl.addEventListener("change", (event) => {
  const el = event.target;

  if (el.dataset.role === "meta-all") {
    const meta = spec.meta_variables.find((m) => m.name === el.dataset.meta);
    windows[meta.name] = el.checked ? [...meta.domain] : [meta.domain[0]];
    refresh();
    return;
  }

  const variable = spec.variables.find((v) => v.name === el.dataset.var);
  if (!variable) return;

  if (el.dataset.role === "explore") {
    if (el.checked) {
      // On restitue la fenêtre pleine, ou la dernière fenêtre non dégénérée connue.
      windows[variable.name] = frozenAt[variable.name] || [...variable.domain];
    } else {
      const w = windows[variable.name] || variable.window;
      if (w[0] !== w[1]) frozenAt[variable.name] = [...w];
      const v = midpoint(variable);
      windows[variable.name] = [v, v];
    }
  } else if (el.dataset.role === "frozen") {
    const v = Number(el.value);
    windows[variable.name] = [v, v];
  } else {
    const w = [...(windows[variable.name] || variable.window)];
    w[el.dataset.role === "min" ? 0 : 1] = Number(el.value);
    if (w[0] > w[1]) w[el.dataset.role === "min" ? 1 : 0] = w[el.dataset.role === "min" ? 0 : 1];
    windows[variable.name] = w;
  }
  refresh();
});

resetBtn.addEventListener("click", () => {
  windows = {};
  frozenAt = {};
  refresh();
});

runBtn.addEventListener("click", runAnalysis);

// ── Couverture, rafraîchie à chaque modification ─────────────────────────────
// Ce que l'application doit savoir pour répondre : rassemblé en un seul endroit,
// pour qu'aucun appel n'oublie le mode ou le dossier.
function demande() {
  return { mode, dossier_donnees: dossierDonnees || null,
           avec_climat: avecClimat, avec_sol: avecSol };
}

function refresh() {
  renderTree();
  activerAides(treeEl);
  clearTimeout(coverageTimer);
  coverageTimer = setTimeout(fetchCoverage, 150);
}

async function fetchCoverage() {
  try {
    const resp = await fetch("/coverage", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...demande(), windows }),
    });
    const body = await resp.json();
    if (!resp.ok) throw new Error(body.detail || "Couverture indisponible.");
    renderCoverage(body);
  } catch (error) {
    document.getElementById("cov-sub").textContent = error.message;
  }
}

function renderCoverage(cov) {
  document.getElementById("cov-points").textContent = fmt(cov.n_points);
  document.getElementById("cov-sub").textContent =
    `sur ${fmt(cov.n_available)} simulations, réparties en ${cov.subspaces.length} sous-espace(s)`;

  document.getElementById("cov-stats").innerHTML = `
    <div class="stat"><span>Analysées</span><strong>${cov.reachable.length}</strong></div>
    <div class="stat"><span>Inconditionnelles</span><strong>${cov.unconditional.length}</strong></div>
    <div class="stat"><span>Décrétées</span><strong>${cov.decreed.length}</strong></div>
    <div class="stat"><span>Hors espace</span><strong>${cov.unreachable.length}</strong></div>`;

  document.getElementById("verdicts").innerHTML = cov.verdicts.map((v) => `
    <div class="verdict ${v.ok ? "ok" : "ko"}">
      <span class="dot"></span>
      <span class="verdict-label">${escapeHtml(v.label)}</span>
      <span class="verdict-reason">${escapeHtml(v.ok ? "calculable" : v.reason)}</span>
    </div>`).join("");

  const plan = cov.plan;
  // Le plan ne s'affiche qu'en mode sur mesure : analyser des simulations déjà
  // faites n'appelle pas à en générer de nouvelles.
  document.getElementById("plan-panel").hidden = mode !== "sur_mesure";
  const constants = Object.entries(plan.constants);
  document.getElementById("plan-body").innerHTML = `
    <div class="plan-grid">
      <div class="stat"><span>Variables du plan</span><strong>${plan.design_names.length}</strong></div>
      <div class="stat"><span>Décrétées</span><strong>${plan.n_decreed}</strong></div>
      <div class="stat"><span>Constantes</span><strong>${constants.length}</strong></div>
      <div class="stat"><span>Sous-espaces</span><strong>${plan.n_subspaces}</strong></div>
    </div>
    <p class="plan-names">${plan.design_names.map((n) => `<code>${escapeHtml(n)}</code>`).join(" ")}</p>
    ${constants.length ? `<p class="hint">Figées : ${constants.map(([k, v]) =>
        `<code>${escapeHtml(k)} = ${escapeHtml(v)}</code>`).join(", ")}</p>` : ""}`;

  const blocked = cov.verdicts.filter((v) => !v.ok);
  runBtn.disabled = blocked.length === cov.verdicts.length;
  runBtn.textContent = runBtn.disabled ? "Trop peu de simulations" : "Lancer l'analyse";
}

// ── Analyse ──────────────────────────────────────────────────────────────────
async function runAnalysis() {
  const choisies = analysesChoisies();
  if (!choisies.length) {
    resultsEl.innerHTML = `<div class="empty-state"><h3>Aucune analyse sélectionnée</h3>
      <p>Coche au moins une analyse dans la barre latérale.</p></div>`;
    return;
  }
  runBtn.disabled = true;
  runBtn.textContent = "Analyse en cours…";
  const lourdes = choisies.filter((a) => a === "hsic" || a === "pdp");
  resultsEl.innerHTML = `<div class="empty-state"><h3>Analyse en cours</h3>
    <p>${choisies.length} analyse(s) demandée(s)${lourdes.length
      ? `, dont ${lourdes.length} coûteuse(s) — cela peut prendre un moment.` : "."}</p></div>`;
  try {
    const resp = await fetch("/analyse", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...demande(), windows, analyses: analysesChoisies(),
      }),
    });
    const body = await resp.json();
    if (!resp.ok) throw new Error(body.detail || "Analyse impossible.");
    renderResults(body);
  } catch (error) {
    resultsEl.innerHTML = `<div class="empty-state error"><h3>Analyse impossible</h3>
      <p>${escapeHtml(error.message)}</p></div>`;
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "Lancer l'analyse";
  }
}

// Matrice paramètre × modalité : une ligne par paramètre, une colonne par strate.
// Le rang est affiché à côté du R², car c'est son instabilité qui révèle une
// interaction entre le paramètre et la variable de regroupement.
function pdpSection(pdp) {
  if (!pdp || pdp.status !== "ok" || !pdp.sous_espaces) return "";
  const calcules = pdp.sous_espaces.filter((s) => s.status === "ok" && s.courbes.length);
  if (!calcules.length) return "";

  const vignettes = calcules.map((s) => `
    <div class="pdp-carte">
      <p class="pdp-titre">${escapeHtml(s.titre)}
        <span class="chart-note">${fmt(s.n_points)} pts · Q² ${s.q2.toFixed(2)}</span></p>
      ${courbePdp(s.courbes[0], { titre: s.courbes[0].variable })}
    </div>`).join("");

  const verdict = pdp.dominant_stable
    ? `<p class="hint">La même variable domine les ${pdp.n_calcules} sous-espaces
         (<code>${escapeHtml(pdp.dominants[0] || "")}</code>).</p>`
    : `<p class="hint">La variable dominante change selon le sous-espace
         (${pdp.dominants.map((d) => `<code>${escapeHtml(d)}</code>`).join(", ")}).</p>`;

  return `<h4 class="strata-title">Effet dominant par sous-espace</h4>
    ${verdict}${legendePdp()}
    <div class="pdp-grille">${vignettes}</div>
    ${pdp.n_ecartes ? `<p class="hint">${pdp.n_ecartes} sous-espace(s) écarté(s),
       effectif insuffisant.</p>` : ""}`;
}

function hsicSection(hsic) {
  // HSIC répond à une question que le R² ne pose pas : la part de dépendance qui
  // tient aux interactions plutôt qu'aux effets simples. On montre donc d'abord
  // cette répartition, puis les termes eux-mêmes.
  if (!hsic || hsic.status !== "ok" || !(hsic.termes || []).length) return "";

  const parts = [
    ["effets simples", hsic.part_ordre_1],
    ["interactions d'ordre 2", hsic.part_ordre_2],
    ["interactions d'ordre 3", hsic.part_ordre_3],
  ].filter(([, v]) => typeof v === "number");

  const lignes = hsic.termes.slice(0, 12).map((t) => ({
    etiquette: t.variables,
    valeur: t.part_globale,
    // L'ordre d'un terme se lit à sa couleur : un effet simple et une interaction
    // ne se comparent pas de la même façon.
    couleur: COULEURS_ORDRE[t.ordre] || COULEURS_ORDRE[3],
    // La part intrinsèque rapporte le terme aux seules simulations où il est actif :
    // une variable rarement active peut peser peu globalement et gouverner son
    // sous-espace. Les deux chiffres se lisent ensemble, jamais l'un sans l'autre.
    detail: `ordre ${t.ordre} · active ${Math.round(t.frequence_active * 100)} % du temps`
            + ` · part intrinsèque ${nombreCourt(t.part_intrinseque)}`,
  }));

  return `<h4 class="strata-title">Dépendance non linéaire (HSIC-ANOVA)</h4>
    <p class="hint">${parts.map(([nom, v]) =>
        `${nom} : <strong>${Math.round(v * 100)} %</strong>`).join(" · ")}
      — sur ${fmt(hsic.n_points)} points et ${hsic.n_termes} termes retenus.</p>
    ${barresHorizontales(lignes, {
      titre: "Part de dépendance par terme", max: null,
      legende: [...new Set(hsic.termes.slice(0, 12).map((t) => t.ordre))].sort()
        .map((o) => ({ couleur: COULEURS_ORDRE[o] || COULEURS_ORDRE[3],
                       libelle: LIBELLES_ORDRE[o] || `ordre ${o}` })),
    })}`;
}

function comparaisonSection(comp) {
  // Quatre familles entraînées sur le même partage. Ce qui importe n'est pas le
  // classement mais l'écart : s'il est faible, la relation ne dépend pas du modèle.
  if (!comp || comp.status !== "ok" || !(comp.scores || []).length) return "";

  const scores = comp.scores.filter((s) => s.status === "ok");
  if (!scores.length) return "";
  // Quatre lignes de trois nombres proches : un tableau les compare mieux que des
  // barres, dont les longueurs seraient indiscernables. Il se lit seul : la coche
  // marque la famille retenue, l'écart au R² dit le surapprentissage.
  const rangs = scores.map((s) => `
    <tr class="${s.model === comp.model_name ? "retenu" : ""}">
      <td>${escapeHtml(s.model)}${s.model === comp.model_name ? " ✓" : ""}</td>
      <td class="num">${nombreCourt(s.Q2_test)}</td>
      <td class="num muted">${nombreCourt(s.R2_train)}</td>
      <td class="num muted">${nombreCourt(s.overfit_gap)}</td>
    </tr>`).join("");

  return `<h4 class="strata-title" data-aide="metamodel_comparison">Comparaison des métamodèles</h4>
    <table class="table">
      <thead><tr><th>Famille</th><th class="num">Q² (test)</th>
        <th class="num">R² (entraînement)</th><th class="num">écart</th></tr></thead>
      <tbody>${rangs}</tbody></table>`;
}

function renderResults(body) {
  const blocks = Object.entries(body.targets).map(([target, entry]) => {
    const rows = (entry.one_factor || []).map((r) => `
      <tr>
        <td>${escapeHtml(r.parametre)}</td>
        <td class="num">${r.r2.toFixed(3)}</td>
        <td class="num muted">${fmt(r.n_points)}</td>
        <td><span class="badge status-${sansAccent(r.statut)}">${escapeHtml(r.statut)}</span></td>
      </tr>`).join("");

    const classement = (entry.one_factor || []).slice(0, 10).map((r) => ({
      etiquette: r.parametre, valeur: r.r2,
      detail: `${fmt(r.n_points)} points · ${r.statut}`,
    }));
    const figureClassement = classement.length
      ? barresHorizontales(classement, {
          titre: `Part de variance expliquée par paramètre — ${target}`, max: 1 })
      : "";

    return `
      <section class="panel">
        <h3>${escapeHtml(target)}</h3>
        ${figureClassement}
        ${comparaisonSection(entry.metamodel_comparison)}
        ${hsicSection(entry.hsic)}
        ${pdpSection(entry.pdp)}
        ${rows ? `<details class="table-view"><summary>Voir les valeurs</summary><table class="table">
          <thead><tr><th>Paramètre</th><th class="num">R² descriptif</th>
            <th class="num">points actifs</th><th>statut</th></tr></thead>
          <tbody>${rows}</tbody></table></details>`
        : '<p class="hint">R² par paramètre non calculé — couverture insuffisante.</p>'}
      </section>`;
  }).join("");

  const anomalies = (body.anomalies || []).length
    ? `<section class="panel warn"><h3>Écarts au domaine déclaré</h3>
        <ul>${body.anomalies.map((a) => `<li>${escapeHtml(a)}</li>`).join("")}</ul></section>`
    : "";

  resultsEl.innerHTML = anomalies + blocks;
  activerInfobulles(resultsEl);
}

// ── Génération du plan ───────────────────────────────────────────────────────
// Deux temps délibérés : on vérifie les calendriers (rapide, rien n'est écrit),
// puis seulement on génère. Le bouton de génération reste inerte tant que la
// vérification n'a pas eu lieu.
const pointsInput = document.getElementById("gen-points");
const pointsField = document.getElementById("gen-points-field");
const checkBtn = document.getElementById("check-btn");
const genBtn = document.getElementById("gen-btn");
const genBody = document.getElementById("gen-body");
const costHint = document.getElementById("gen-cost");

// Le nombre de runs est la seule mesure du coût affichée : c'est un fait, là où
// une durée dépendrait de la machine, du terrain et du nombre de campagnes.
function updateCost() {
  const points = Number(pointsInput.value) || 0;
  const runs = Math.max(1, Math.ceil(points / 100));
  const tip = `${runs} run${runs > 1 ? "s" : ""} GAMA de 100 parcelles`;

  costHint.textContent = `${runs} run${runs > 1 ? "s" : ""}`;
  costHint.title = tip;
  pointsField.title = tip;
  checkBtn.title = `Contrôle les calendriers en mémoire, sans rien écrire. ${tip}`;
  genBtn.title = `Écrit les fichiers dateDose et XML dans un dossier de run. ${tip}`;
}

pointsInput.addEventListener("input", () => {
  updateCost();
  // Changer le nombre de points invalide la vérification précédente.
  genBtn.disabled = true;
});

function planBody() {
  return { ...demande(), windows, n_points: Number(pointsInput.value) };
}

checkBtn.addEventListener("click", async () => {
  checkBtn.disabled = true;
  checkBtn.textContent = "Vérification…";
  try {
    const resp = await fetch("/plan/check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(planBody()),
    });
    const body = await resp.json();
    if (!resp.ok) throw new Error(body.detail || "Vérification impossible.");
    renderCheck(body);
    genBtn.disabled = body.n_failed > 0;
  } catch (error) {
    genBody.innerHTML = `<p class="hint error-text">${escapeHtml(error.message)}</p>`;
    genBtn.disabled = true;
  } finally {
    checkBtn.disabled = false;
    checkBtn.textContent = "Vérifier les calendriers";
  }
});

function renderCheck(body) {
  const blocking = body.n_failed > 0;
  const lines = body.summary.length
    ? `<ul class="warn-list">${body.summary.map((l) => `<li>${escapeHtml(l)}</li>`).join("")}</ul>`
    : `<p class="hint">Aucun décalage ni apport confondu détecté.</p>`;

  genBody.innerHTML = `
    <div class="verdict-block ${blocking ? "ko" : "ok"}">
      <strong>${blocking
        ? `${body.n_failed}/${body.n_points} calendriers infaisables — génération bloquée`
        : `${body.n_ok}/${body.n_points} calendriers valides`}</strong>
      ${lines}
      ${blocking ? "" : `<p class="hint">Les points ci-dessus n'empêchent pas la génération,
        mais ils décrivent des itinéraires qui s'écartent de ce que la sélection demande.</p>`}
    </div>`;
}

genBtn.addEventListener("click", async () => {
  genBtn.disabled = true;
  genBtn.textContent = "Génération…";
  try {
    const resp = await fetch("/plan/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(planBody()),
    });
    const body = await resp.json();
    if (!resp.ok) throw new Error(body.detail || "Génération impossible.");
    renderGenerated(body);
  } catch (error) {
    genBody.innerHTML = `<p class="hint error-text">${escapeHtml(error.message)}</p>`;
  } finally {
    genBtn.textContent = "Générer les fichiers";
    genBtn.disabled = true; // il faut re-vérifier avant de regénérer
  }
});

// Une commande que le serveur ne renvoie pas ne doit jamais s'afficher « undefined ».
// Le cas se produit quand le serveur tourne encore avec une version antérieure du
// code Python : les fichiers statiques sont relus à chaque requête, le module non.
function commande(valeur) {
  return valeur
    ? `<pre>${escapeHtml(valeur)}</pre>`
    : `<p class="hint avertissement">Le serveur n'a pas renvoyé cette commande. Il
         tourne probablement avec une version antérieure du code : arrêtez-le et
         relancez-le, de préférence avec <code>--reload</code>.</p>`;
}

function renderGenerated(body) {
  const v = body.validation;
  const warnings = v.summary.length
    ? `<ul class="warn-list">${v.summary.map((l) => `<li>${escapeHtml(l)}</li>`).join("")}</ul>`
    : "";

  genBody.innerHTML = `
    <div class="verdict-block ok">
      <strong>${fmt(body.n_points)} points · ${body.n_runs} fichiers dateDose et XML</strong>
      <p class="hint">Écrit dans <code>${escapeHtml(body.output_dir)}</code> —
        rien n'a été modifié dans l'installation MAELIA.</p>
      ${warnings}
    </div>
    <div class="commands">
      <h4>1 · Installer les itinéraires dans le terrain</h4>
      <p class="hint">Cette commande écrit dans MAELIA et peut écraser les fichiers d'un plan
        précédent. À exécuter en connaissance de cause.</p>
      ${commande(body.commands.install)}
      <h4 title="${body.n_runs} run(s) GAMA de 100 parcelles">2 · Lancer GAMA</h4>
      ${body.commands.gama_headless_trouve
        ? ""
        : `<p class="hint">gama-headless.sh n'a pas été trouvé automatiquement :
             remplace le chemin dans la commande.</p>`}
      ${commande(body.commands.gama)}
      <h4>3 · Rassembler les sorties de GAMA</h4>
      <p class="hint">Relie chaque point du plan aux sorties de sa parcelle et écrit le
        jeu de données à côté du plan. Seules les sorties postérieures au plan sont
        retenues : GAMA n'écrase jamais ses dossiers.</p>
      ${commande(body.commands.collecte)}

      <h4>4 · Analyser</h4>
      <p class="hint">Revenez en haut, choisissez « Analyser des données existantes » et
        saisissez le dossier du plan. Sa spécification y est déjà : l'analyse portera
        exactement sur l'espace qui a servi à tirer ce plan.</p>
      <pre>${escapeHtml(body.output_dir)}</pre>
    </div>`;
}

// ── Démarrage ────────────────────────────────────────────────────────────────
// ── Choix de l'espace de référence ───────────────────────────────────────────
async function loadSpec() {
  const q = new URLSearchParams({
    mode, dossier_donnees: dossierDonnees || "", avec_climat: String(avecClimat),
  });
  spec = await (await fetch(`/spec?${q}`)).json();
  document.getElementById("spec-name").textContent = spec.name;
  dossierEtat.textContent = mode === "donnees"
    ? `Espace lu depuis : ${spec.origin} · ${fmt(spec.n_available)} simulations`
    : "";
  // Changer d'espace invalide les fenêtres : leurs bornes ne s'appliquent plus.
  windows = {};
  frozenAt = {};
  refresh();
}

// ── Mode, source des données, climat ─────────────────────────────────────────
const blocDonnees = document.getElementById("bloc-donnees");
const blocSurMesure = document.getElementById("bloc-sur-mesure");
const jeuSelect = document.getElementById("jeu-select");
const dossierInput = document.getElementById("dossier-input");
const dossierEtat = document.getElementById("dossier-etat");
const climatCheck = document.getElementById("climat-check");

function appliquerMode() {
  blocDonnees.hidden = mode !== "donnees";
  blocSurMesure.hidden = mode !== "sur_mesure";
  // Le plan ne se génère que sur mesure : analyser des données existantes ne
  // produit pas de nouvelles simulations.
  document.getElementById("plan-panel").hidden = mode !== "sur_mesure";
}

document.querySelectorAll('input[name="mode"]').forEach((r) => {
  r.addEventListener("change", () => {
    mode = r.value;
    appliquerMode();
    loadSpec();
  });
});

jeuSelect.addEventListener("change", () => {
  dossierDonnees = jeuSelect.value;
  dossierInput.value = "";
  loadSpec();
});

dossierInput.addEventListener("change", () => {
  dossierDonnees = dossierInput.value.trim();
  loadSpec();
});

const solCheck = document.getElementById("sol-check");

climatCheck.addEventListener("change", () => {
  avecClimat = climatCheck.checked;
  annoncerTerrain();
  loadSpec();
});

solCheck.addEventListener("change", () => {
  avecSol = solCheck.checked;
  annoncerTerrain();
  loadSpec();
});

// Ce que les strates coûtent en géométrie, dit avant plutôt que découvert après.
function annoncerTerrain() {
  const emplacements = avecClimat ? 8 : 1;
  const sols = avecSol ? 3 : 1;
  const ilots = emplacements * sols;
  document.getElementById("strates-cout").textContent = ilots > 1
    ? `Terrain à construire : ${ilots} îlots (${emplacements} emplacement${
        emplacements > 1 ? "s" : ""} × ${sols} sol${sols > 1 ? "s" : ""}), `
      + `${ilots * 12} parcelles par exécution de GAMA.`
    : "";
}

(async function start() {
  const catalogue = await (await fetch("/modes")).json();

  jeuSelect.innerHTML = catalogue.jeux_precharges.map((j) =>
    `<option value="${escapeHtml(j.cle)}">${escapeHtml(j.nom)} — ${fmt(j.n_points)} points${
      j.avec_climat ? " · avec climat" : ""}</option>`).join("");
  dossierDonnees = catalogue.jeux_precharges[0]?.cle || "";

  document.getElementById("climat-liste").textContent =
    "Types explorés : " + catalogue.climats.map((c) => c.libelle).join(" · ");
  document.getElementById("sol-liste").textContent =
    catalogue.sols.map((s) => s.libelle).join(" · ");

  renderChemins(catalogue.chemins);
  document.getElementById("reglage-enregistrer")
    .addEventListener("click", () => enregistrerReglages(false));
  document.getElementById("reglage-effacer")
    .addEventListener("click", () => enregistrerReglages(true));
  chargerReglages();
  appliquerMode();
  installerAides();
  await loadSpec();
  updateCost();
})();

// ── Chemins vers MAELIA et GAMA ──────────────────────────────────────────────
// Ils vivent hors de l'application et changent d'une machine à l'autre. Le réglage
// est relu à chaque appel côté serveur : il prend effet sans redémarrage.
async function chargerReglages() {
  const body = await (await fetch("/reglages")).json();
  document.getElementById("reglage-maelia").value = body.reglages.maelia_root || "";
  document.getElementById("reglage-gama").value = body.reglages.gama_headless || "";
  // Le champ vide n'est pas un champ vide de sens : il montre ce que l'application
  // utilise à défaut. Le chemin de GAMA ne varie guère d'une installation à l'autre,
  // et le saisir ne sert que dans le cas exceptionnel où il diffère.
  document.getElementById("reglage-gama").placeholder =
    body.chemins.gama.chemin || body.candidats.gama[0];
  document.getElementById("reglage-maelia").placeholder =
    body.chemins.maelia.chemin || body.candidats.maelia[0];
  rendreEtatReglage("etat-maelia", body.maelia, body.chemins.maelia);
  rendreEtatReglage("etat-gama", body.gama, body.chemins.gama);
  document.getElementById("reglage-fichier").textContent =
    `Enregistré dans ${body.fichier}. Laissez vide pour retomber sur les variables `
    + `d'environnement, puis sur les emplacements d'installation habituels.`;
  renderChemins(body.chemins);
}

function rendreEtatReglage(id, diagnostic, etat) {
  const el = document.getElementById(id);
  const bon = diagnostic.existe !== false;
  el.className = `reglage-etat ${bon ? "ok" : "ko"}`;
  el.textContent = diagnostic.existe === null
    ? `non réglé — ${etat.chemin ? `trouvé via ${etat.source}` : "introuvable"}`
    : `${diagnostic.message}${bon && etat.source ? ` · ${etat.source}` : ""}`;
}

async function enregistrerReglages(effacer) {
  const corps = effacer
    ? { maelia_root: "", gama_headless: "" }
    : { maelia_root: document.getElementById("reglage-maelia").value.trim(),
        gama_headless: document.getElementById("reglage-gama").value.trim() };
  const body = await (await fetch("/reglages", {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify(corps),
  })).json();
  document.getElementById("reglage-maelia").value = body.reglages.maelia_root || "";
  document.getElementById("reglage-gama").value = body.reglages.gama_headless || "";
  rendreEtatReglage("etat-maelia", body.maelia, body.chemins.maelia);
  rendreEtatReglage("etat-gama", body.gama, body.chemins.gama);
  renderChemins(body.chemins);
}

// Où l'application range ses fichiers : montré, pas deviné. Quand GAMA ne démarre
// pas, la première question est toujours « où cherche-t-il ? ».
function renderChemins(etat) {
  // Une seule ancre absolue — le dossier de l'application — et tout le reste écrit
  // relativement à elle. Un chemin complet est illisible et change d'une machine à
  // l'autre, là où la structure, elle, ne bouge jamais.
  const ligne = (nom, entree, ok = true) => `
    <div class="chemin-ligne">
      <span class="chemin-nom">${escapeHtml(nom)}</span>
      <code class="${ok ? "" : "manquant"}" title="${escapeHtml(entree.chemin ?? "")}"
        >${escapeHtml(entree.affiche ?? "introuvable")}</code>
    </div>`;
  document.getElementById("chemins-body").innerHTML = `
    <div class="chemin-ligne ancre">
      <span class="chemin-nom">Dossier de l'application</span>
      <code>${escapeHtml(etat.application)}</code>
    </div>
    <p class="hint">Les quatre suivants lui sont relatifs : l'application n'écrit que
      dans son propre dossier.</p>
    ${ligne("Espaces livrés", etat.espaces)}
    ${ligne("Plans produits", etat.simulations)}
    ${ligne("Résultats", etat.resultats)}
    ${ligne("Terrains construits", etat.terrains)}
    <p class="hint">Ces deux-là vivent dehors. Le chemin complet apparaît au survol.</p>
    ${ligne("Installation MAELIA", etat.maelia, etat.maelia.present)}
    ${ligne("Script gama-headless", etat.gama, etat.gama.present)}`;
}
