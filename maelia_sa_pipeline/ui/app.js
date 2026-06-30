const form = document.getElementById("analysis-form");
const runButton = document.getElementById("run-button");
const statusBand = document.getElementById("status-band");
const statusIndicator = document.getElementById("status-indicator");
const summaryGrid = document.getElementById("summary-grid");
const targetTabs = document.getElementById("target-tabs");
const resultsPanel = document.getElementById("results-panel");


let currentManifest = null;
let currentTarget = null;
let tooltipTimer = null;
let tooltipNode = null;
let currentHelpTarget = null;

const targetLabels = {
  N_lixi: "Azote lixivié",
  dCorg: "Carbone organique",
  rdt: "Rendement",
};

const analysisLabels = {
  anova_1factor: "ANOVA 1 facteur",
  anova_2factor: "ANOVA 2 facteurs",
  hsic_anova: "HSIC-ANOVA",
  metamodel: "Métamodèles",
  sobol_indices: "Sobol S1/ST",
  pdp_ice: "PDP/ICE",
  decision_tree: "Arbres de régression",
};

const figureLabels = {
  anova_1factor_png: "ANOVA à un facteur",
  anova_2factor_interaction_png: "Interactions à deux facteurs",
  metamodel_performance_png: "Performance du métamodèle",
  pce_sobol_png: "Sobol ordre 1 et total",
  hsic_anova_order_png: "HSIC-ANOVA",
  decision_tree_regions_png: "Seuils et régions sensibles",
  decision_tree_png: "Arbre de régression",
  pdp_ice_pngs: "PDP/ICE finales",
};

const featureOrder = [
  "n_ferti", "has_prepa", "nb_prepa",
  "Date_Semis", "Delta_PREPA_Semis", "Profondeur_Semis",
  "Profondeur_Prepa_1", "Profondeur_Prepa_2",
  "Date_F1", "Date_F2", "Date_F3", "Date_Recolte",
  "Dose_F1", "Dose_F2", "Dose_F3",
];

const helpTexts = {
  log_dir: "Dossier où GAMA a écrit les logs d'une série de simulations MAELIA. Il doit contenir les fichiers de sortie utilisés pour calculer N_lixi, dCorg et rdt. Le dataset de paramètres doit être dans ce dossier ou indiqué séparément.",
  dataset_path: "Chemin vers dataset_metamodel.csv, exporté par le notebook de simulation. Ce fichier contient la matrice du plan SMT, donc les paramètres testés pour chaque simulation. Les logs seuls ne suffisent pas.",
  n_bins: "Nombre de classes utilisées pour transformer les paramètres continus en groupes avant l'ANOVA/Kruskal. Plus de classes donne une lecture plus fine, mais exige plus de points par classe.",
  sobol_n_mc: "Paramètre de compatibilité conservé par l'API. Le bloc Sobol affiché par l'app calcule désormais S1 et ST via un PCE creux entraîné sur les points SMT faisables.",
  tree_max_depth: "Profondeur maximale des arbres de décision. Une profondeur faible donne des seuils lisibles; une profondeur élevée capture plus de détails mais devient plus difficile à interpréter.",
  random_state: "Graine aléatoire utilisée pour rendre reproductibles les séparations train/test, l'entraînement et les échantillonnages associés.",
  targets: "Sorties MAELIA analysées. Chaque sortie produit ses propres scores, figures, indices et régions sensibles.",
  analyses_menu: "Menu des blocs à exécuter. L'app est limitée aux analyses retenues pour le workflow terrainSA : ANOVA 1/2 facteurs, HSIC-ANOVA, métamodèles, Sobol S1/ST, arbres de régression et PDP/ICE.",
  analysis_anova_1factor: "Classe les paramètres selon leur effet descriptif individuel sur chaque sortie.",
  analysis_anova_2factor: "Calcule les interactions entre couples de paramètres et génère une matrice de R² d'interaction.",
  analysis_sobol_indices: "Calcule les indices de Sobol d'ordre 1 et d'ordre total via un PCE creux entraîné sur un sous-échantillon reproductible des points SMT faisables.",
  analysis_hsic_anova: "Décompose la dépendance non linéaire entre paramètres et sortie avec HSIC-ANOVA. La figure résume la part portée par les effets simples, les interactions à deux paramètres et les interactions plus complexes.",
  analysis_decision_tree: "Entraîne un RegressionTree interprétable pour identifier des seuils et des régions locales de réponse.",
  
  analysis_metamodel: "Compare et sélectionne un métamodèle prédictif parmi les candidats disponibles. Ce modèle sert notamment aux PDP/ICE.",
  analysis_pdp_ice: "Trace les courbes PDP/ICE finales pour les paramètres temporels principaux. Ce bloc entraîne automatiquement le métamodèle prédictif.",
  
  run_button: "Lance la pipeline sélectionnée : chargement logs + dataset, contrôle des colonnes, ANOVA, HSIC-ANOVA, métamodèles, Sobol S1/ST, seuils ou PDP/ICE selon les cases cochées.",
  summary_rows: "Nombre de simulations exploitables dans le dataset après chargement et alignement avec les logs.",
  summary_features: "Nombre de paramètres agronomiques utilisés comme variables d'entrée de l'analyse. Le plan actuel compact en comporte 15.",
  summary_targets: "Nombre de sorties MAELIA demandées pour l'analyse en cours.",
  summary_run: "Identifiant unique du calcul web. Les figures, CSV et rapports sont sauvegardés dans analysis/web_runs/<run_id>/.",
  summary_analyses: "Nombre de blocs d'analyse effectivement demandés pour ce run.",
  target_N_lixi: "Azote lixivié. Sortie environnementale représentant les pertes d'azote par lixiviation, généralement interprétées comme une pression sur l'eau et le sol.",
  target_dCorg: "Variation de carbone organique du sol. Sortie décrivant l'évolution simulée du stock de carbone organique; elle dépend fortement de la fenêtre culturale et de la dynamique du sol.",
  target_rdt: "Rendement. Sortie agronomique de production, utile pour comparer les effets techniques sur la performance de la culture.",
  metric_R2_train: "R² d'entraînement. Part de variance expliquée par le modèle sur les données utilisées pour l'entraîner. Un bon R² seul ne suffit pas: il faut le comparer au Q² de test.",
  metric_Q2_test: "Q² de test. R² calculé sur des simulations non vues pendant l'entraînement. C'est l'indicateur principal de généralisation du métamodèle ou de l'arbre.",
  metric_selected_model: "Métamodèle sélectionné automatiquement pour cette sortie. Les candidats sont comparés sur le même dataset final dynamique et classés principalement par Q² de test, avec une petite pénalité de surapprentissage.",
  metric_pce_model: "Métamodèle polynomial du chaos creux utilisé pour calculer Sobol S1/ST sur l'espace SMT faisable, sans générer de recombinaisons Saltelli invalides.",
  metric_pce_R2_train: "R² d'entraînement du PCE creux. Il indique à quel point le polynôme approxime les points utilisés pour son ajustement.",
  metric_pce_Q2_test: "Q² de test du PCE creux. C'est le meilleur signal pour savoir si les indices Sobol S1/ST reposent sur une approximation fiable.",
  metric_tree_R2_train: "R² de l'arbre sur l'ensemble d'entraînement. Il mesure à quel point l'arbre interprétable capture la structure des données d'apprentissage.",
  metric_tree_Q2_test: "Q² de l'arbre sur l'ensemble de test. Il indique si les seuils affichés par l'arbre restent prédictifs sur des simulations non vues.",
  anova_1factor_png: "ANOVA/Kruskal à un facteur. La figure classe les paramètres selon leur R² descriptif, c'est-à-dire la part de variance expliquée par les groupes de ce paramètre.",
  anova_2factor_interaction_png: "Matrice d'interaction à deux facteurs. Elle affiche seulement le R² d'interaction, donc ce qui reste quand les effets additifs des deux paramètres sont retirés.",
  metamodel_performance_png: "Performance du métamodèle. Elle compare les prédictions aux valeurs observées et sépare la qualité d'entraînement de la qualité de test.",
  pce_sobol_png: "Indices de Sobol d'ordre 1 et total calculés depuis le PCE creux web. Cette figure évite les recombinaisons Saltelli incompatibles avec les contraintes SMT.",
  hsic_anova_order_png: "Décomposition HSIC-ANOVA par ordre. L’ordre 1 correspond aux effets simples, l’ordre 2 aux interactions deux à deux, et les ordres supérieurs aux dépendances plus combinatoires. Ce n’est pas un R² ni un indice de Sobol, mais une contribution au HSIC global.",
  
  decision_tree_regions_png: "Régions sensibles. Chaque barre correspond à une feuille de l'arbre, donc à un ensemble de simulations partageant les mêmes règles de seuil.",
  decision_tree_png: "Arbre de décision complet. Il expose les seuils successifs utilisés pour séparer les simulations en régimes locaux.",
  
  pdp_ice_pngs: "PDP/ICE finales calculées avec le métamodèle sur l'état final des expériences. L'app affiche systématiquement Date de semis, Date de récolte et Décalage préparation-semis.",
  pdp_ice_feature: "Paramètre temporel continu retenu pour comparer les trois sorties sur les mêmes axes d'interprétation. La figure montre comment la sortie finale prédite évolue quand ce paramètre varie et que les autres restent distribués comme dans le plan SMT.",
  action_report: "Ouvre le rapport HTML complet sauvegardé pour ce run. Il regroupe toutes les figures générées.",
  action_regions_csv: "Ouvre le CSV des régions sensibles. Il contient les règles de seuil, la taille de chaque région et la moyenne de sortie associée.",
  action_rules_txt: "Ouvre les règles textuelles brutes de l'arbre de décision.",
  feature_n_ferti: "Nombre d'événements de fertilisation azotée activés dans l'itinéraire. Le produit est fixé à AN; seules les dates et doses des apports varient.",
  feature_has_prepa: "Indique si une préparation du sol est activée avant le semis. Cette variable active le nombre de préparations, le délai préparation-semis et les profondeurs de travail du sol.",
  feature_nb_prepa: "Nombre d'opérations de préparation du sol effectuées sur l'unique date de préparation. Le plan autorise une ou deux profondeurs de travail.",
  feature_Date_Semis: "Date de semis exprimée en jour de campagne, avec 1 = 1er août. C'est un moteur majeur des fenêtres culturales.",
  feature_Delta_PREPA_Semis: "Délai entre préparation du sol et semis. La valeur est négative: -25 signifie environ 25 jours avant le semis.",
  feature_Profondeur_Semis: "Profondeur du semis en centimètres. Dans MAELIA, elle agit comme un petit travail du sol associé au semis.",
  feature_Profondeur_Prepa_1: "Profondeur de la première opération de préparation du sol en centimètres. Elle remplace les anciens noms de préparation, qui n'étaient que des alias.",
  feature_Profondeur_Prepa_2: "Profondeur de la deuxième opération de préparation du sol, active seulement lorsque deux préparations sont prévues.",
  feature_Date_F1: "Date du premier apport d'azote minéral AN, exprimée en jour de campagne.",
  feature_Date_F2: "Date du deuxième apport d'azote minéral AN, exprimée en jour de campagne et active si F2 existe.",
  feature_Date_F3: "Date du troisième apport d'azote minéral AN, exprimée en jour de campagne et active si F3 existe.",
  feature_Date_Recolte: "Date de récolte exprimée en jour de campagne. Elle définit la fin de la fenêtre culturale simulée.",
  feature_Dose_F1: "Dose d'azote de l'apport F1, avec le fertilisant fixé à AN.",
  feature_Dose_F2: "Dose d'azote de l'apport F2, active si le deuxième apport existe.",
  feature_Dose_F3: "Dose d'azote de l'apport F3, active si le troisième apport existe.",
};

const references = [
  "Borgonovo et al. (2022) : protocole d'analyse de sensibilité pour modèles agent-based.",
  "ten Broeke et al. (2016) : choix de méthodes de sensibilité selon la question posée.",
  "Thiele et al. (2014) : guide pratique pour estimation de paramètres et sensibilité en ABM.",
  "Sobol' (2001) et Saltelli (2002) : indices de sensibilité variance-based et indices totaux.",
  "Shapley (1953) et Song et al. (2016) : allocation de variance par effets de Shapley.",
  "Breiman et al. (1984) : arbres de classification et régression pour règles de seuil.",
  "Gretton et al. (2005) et Da Veiga (2015) : HSIC et décomposition HSIC-ANOVA par noyaux.",
  "Geurts et al. (2006), Chen et Guestrin (2016), Rasmussen et Williams (2006) : ExtraTrees, XGBoost et Gaussian Processes pour métamodélisation.",
];

function setStatus(kind, eyebrow, title, text) {
  const className = kind === "running" ? "running" : kind === "done" ? "done" : kind === "error" ? "error" : "";
  statusIndicator.className = `status-indicator ${className}`.trim();
  statusIndicator.textContent = kind === "running" ? "Analyse en cours" : kind === "done" ? "Terminé" : kind === "error" ? "À corriger" : "En attente";
  statusBand.querySelector(".eyebrow").textContent = eyebrow;
  statusBand.querySelector("h2").textContent = title;
  if (text) statusBand.title = text;
}

function showLoading() {
  resultsPanel.innerHTML = `
    <div class="loading-state">
      <div>
        <div class="spinner"></div>
        <h3>Analyse en cours</h3>
        <p>La pipeline entraîne les métamodèles, estime les indices et prépare les figures. Selon la taille du dataset, cela peut prendre un peu de temps.</p>
      </div>
    </div>`;
  decorateHelpables(resultsPanel);
}

function showError(message) {
  resultsPanel.innerHTML = `
    <div class="error-state">
      <div>
        <div class="empty-visual"></div>
        <h3>Impossible de lancer l'analyse</h3>
        <p>${escapeHtml(message)}</p>
      </div>
    </div>`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function helpSpan(label, key) {
  return `<span class="helpable" data-help-key="${escapeHtml(key)}">${escapeHtml(label)}</span>`;
}

function relativeAssetUrl(path) {
  if (!currentManifest || !path) return "";
  const outputDir = currentManifest.output_dir.replace(/\/$/, "");
  let relative = path;
  if (path.startsWith(outputDir)) {
    relative = path.slice(outputDir.length).replace(/^\//, "");
  }
  return `/analyses/${encodeURIComponent(currentManifest.run_id)}/${relative.split("/").map(encodeURIComponent).join("/")}`;
}

function collectPayload() {
  const data = new FormData(form);
  const targets = [...form.querySelectorAll('input[name="targets"]:checked')].map((item) => item.value);
  const analyses = [...form.querySelectorAll('input[name="analyses"]:checked')].map((item) => item.value);
  return {
    log_dir: data.get("log_dir").trim(),
    dataset_path: data.get("dataset_path").trim() || null,
    targets: targets.length ? targets : null,
    analyses: analyses.length ? analyses : [],
    n_bins: Number(data.get("n_bins")),
    sobol_n_mc: Number(data.get("sobol_n_mc")),
    tree_max_depth: Number(data.get("tree_max_depth")),
    random_state: Number(data.get("random_state")),
  };
}

function renderSummary(manifest) {
  const targetCount = Object.keys(manifest.targets || {}).length;
  const analysisCount = (manifest.analyses || []).length;
  summaryGrid.hidden = false;
  summaryGrid.innerHTML = `
    <div class="metric"><span>${helpSpan("Simulations", "summary_rows")}</span><strong>${manifest.n_rows.toLocaleString("fr-FR")}</strong></div>
    <div class="metric"><span>${helpSpan("Paramètres", "summary_features")}</span><strong>${manifest.n_features}</strong></div>
    <div class="metric"><span>${helpSpan("Sorties", "summary_targets")}</span><strong>${targetCount}</strong></div>
    <div class="metric"><span>${helpSpan("Analyses", "summary_analyses")}</span><strong>${analysisCount}</strong></div>
    <div class="metric"><span>${helpSpan("Run", "summary_run")}</span><strong>${escapeHtml(manifest.run_id.slice(-8))}</strong></div>`;
  decorateHelpables(summaryGrid);
}

function renderTabs(manifest) {
  const targets = Object.keys(manifest.targets || {});
  targetTabs.hidden = targets.length === 0;
  targetTabs.innerHTML = targets.map((target) => `
    <button class="tab ${target === currentTarget ? "active" : ""}" data-target="${escapeHtml(target)}" type="button">
      ${helpSpan(targetLabels[target] || target, `target_${target}`)}
    </button>`).join("");
  decorateHelpables(targetTabs);
  targetTabs.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      currentTarget = tab.dataset.target;
      renderTarget(currentTarget);
      renderTabs(currentManifest);
    });
  });
}

function scoreCard(label, value, className, helpKey) {
  const numeric = Number(value);
  const rendered = Number.isFinite(numeric) ? numeric.toFixed(2) : "-";
  return `<div class="score ${className}"><span>${helpSpan(label, helpKey)}</span><strong>${rendered}</strong></div>`;
}

function textScoreCard(label, value, className, helpKey) {
  return `<div class="score ${className}"><span>${helpSpan(label, helpKey)}</span><strong>${escapeHtml(value || "-")}</strong></div>`;
}

function figurePanelFromPath(path, label, helpKey, wide = false, extraClass = "") {
  if (!path) return "";
  return `
    <section class="figure-panel ${wide ? "wide" : ""} ${extraClass}">
      <h3>${helpSpan(label, helpKey)}</h3>
      <img src="${relativeAssetUrl(path)}" alt="${escapeHtml(label)}">
    </section>`;
}

function figurePanel(artifacts, key, wide = false, extraClass = "") {
  return figurePanelFromPath(artifacts[key], figureLabels[key], key, wide, extraClass);
}

function actionLink(path, label, helpKey, primary = false) {
  if (!path) return "";
  return `<a class="link-button ${primary ? "primary" : ""} helpable" data-help-key="${escapeHtml(helpKey)}" href="${relativeAssetUrl(path)}" target="_blank" rel="noreferrer">${escapeHtml(label)}</a>`;
}

function reportLink() {
  if (!currentManifest) return "";
  return `<a class="link-button primary helpable" data-help-key="action_report" href="/analyses/${encodeURIComponent(currentManifest.run_id)}/report" target="_blank" rel="noreferrer">Ouvrir le rapport complet</a>`;
}

function pdpIcePanels(artifacts) {
  const items = artifacts.pdp_ice_pngs || [];
  if (!items.length) return "";
  return `
    <section class="subsection-title wide">
      <div>
        <p>${helpSpan("Lecture finale", "pdp_ice_pngs")}</p>
        <h3>PDP/ICE principaux</h3>
      </div>
    </section>
    ${items.map((item) => figurePanelFromPath(
      item.path,
      `PDP/ICE — ${item.feature_label || item.feature || "paramètre"}`,
      "pdp_ice_feature",
      false,
      "pdp-panel"
    )).join("")}`;
}

function renderTarget(target) {
  if (!currentManifest || !currentManifest.targets[target]) return;
  const artifacts = currentManifest.targets[target];
  const metrics = artifacts.metamodel_metrics || null;
  const treeMetrics = artifacts.decision_tree_metrics || null;
  const pceMetrics = artifacts.pce_metrics || null;
  const modelScores = metrics ? `
      <div class="score-row">
        ${textScoreCard("Métamodèle retenu", metrics.model_name, "model", "metric_selected_model")}
        ${scoreCard("R² entraînement métamodèle", metrics.R2_train, "train", "metric_R2_train")}
        ${scoreCard("Q² test métamodèle", metrics.Q2_test, "test", "metric_Q2_test")}
      </div>` : "";
  const treeScores = treeMetrics ? `
      <div class="score-row">
        ${scoreCard("R² entraînement arbre", treeMetrics.R2_train, "train", "metric_tree_R2_train")}
        ${scoreCard("Q² test arbre", treeMetrics.Q2_test, "test", "metric_tree_Q2_test")}
      </div>` : "";
  const pceScores = pceMetrics ? `
      <div class="score-row pce-score-row">
        ${textScoreCard("Sobol S1/ST", pceMetrics.model_name || "SparsePCE", "model pce", "metric_pce_model")}
        ${scoreCard("R² entraînement PCE", pceMetrics.R2_train, "train pce", "metric_pce_R2_train")}
        ${scoreCard("Q² test PCE", pceMetrics.Q2_test, "test pce", "metric_pce_Q2_test")}
      </div>` : "";
  resultsPanel.innerHTML = `
    <div class="target-view">
      ${modelScores}
      ${pceScores}
      ${treeScores}
      <div class="report-actions">
        ${reportLink()}
        ${actionLink(artifacts.decision_tree_regions_csv, "Voir les régions CSV", "action_regions_csv")}
        ${actionLink(artifacts.decision_tree_rules_txt, "Voir les règles", "action_rules_txt")}
      </div>
      <div class="figure-grid">
        ${figurePanel(artifacts, "metamodel_performance_png")}
        ${figurePanel(artifacts, "pce_sobol_png", false, "pce-panel")}
        ${figurePanel(artifacts, "hsic_anova_order_png", false, "hsic-panel")}
        ${pdpIcePanels(artifacts)}
        ${figurePanel(artifacts, "anova_1factor_png")}
        ${figurePanel(artifacts, "anova_2factor_interaction_png")}
        ${figurePanel(artifacts, "decision_tree_regions_png", true)}
        ${figurePanel(artifacts, "decision_tree_png", true)}
      </div>
    </div>`;
  decorateHelpables(resultsPanel);
}

async function runAnalysis(event) {
  event.preventDefault();
  const payload = collectPayload();
  currentManifest = null;
  currentTarget = null;
  summaryGrid.hidden = true;
  targetTabs.hidden = true;
  runButton.disabled = true;
  runButton.innerHTML = '<span class="button-icon">…</span> Analyse en cours';
  setStatus("running", "Calcul", "La pipeline prépare les analyses et les visualisations.");
  showLoading();

  try {
    const response = await fetch("/analyses", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "Erreur inconnue pendant l'analyse.");
    currentManifest = body;
    currentTarget = Object.keys(body.targets || {})[0];
    setStatus("done", "Résultats", `Analyse terminée : ${body.n_rows.toLocaleString("fr-FR")} simulations traitées.`);
    renderSummary(body);
    renderTabs(body);
    renderTarget(currentTarget);
  } catch (error) {
    setStatus("error", "Erreur", "L'analyse n'a pas pu être produite.");
    showError(error.message);
  } finally {
    runButton.disabled = false;
    runButton.innerHTML = '<span class="button-icon">▶</span> Lancer l\'analyse';
  }
}

function ensureTooltip() {
  if (!tooltipNode) {
    tooltipNode = document.createElement("div");
    tooltipNode.className = "help-tooltip";
    tooltipNode.hidden = true;
    document.body.appendChild(tooltipNode);
  }
  return tooltipNode;
}

function resolveHelpTarget(node) {
  const direct = node.closest?.(".helpable[data-help-key]");
  if (direct) return direct;
  const labelledField = node.closest?.("label, .metric, .score, .figure-panel, .report-actions");
  return labelledField?.querySelector?.(".helpable[data-help-key]") || null;
}

function decorateHelpables(root = document) {
  root.querySelectorAll(".helpable[data-help-key]").forEach((item) => {
    const text = helpTexts[item.dataset.helpKey];
    if (!text) return;
    item.setAttribute("tabindex", "0");
    item.setAttribute("aria-label", `${item.textContent.trim()} - ${text}`);
    item.removeAttribute("title");
  });
}

function showTooltip(target, immediate = false) {
  const key = target.dataset.helpKey;
  const text = helpTexts[key];
  if (!text) return;
  currentHelpTarget = target;
  clearTimeout(tooltipTimer);
  tooltipTimer = setTimeout(() => {
    const tooltip = ensureTooltip();
    tooltip.textContent = text;
    tooltip.hidden = false;
    const rect = target.getBoundingClientRect();
    const maxLeft = window.innerWidth - 390;
    const top = rect.bottom + 10;
    tooltip.style.left = `${Math.max(16, Math.min(rect.left, maxLeft))}px`;
    tooltip.style.top = `${Math.max(16, Math.min(top, window.innerHeight - 120))}px`;
  }, immediate ? 120 : 2000);
}

function hideTooltip() {
  currentHelpTarget = null;
  clearTimeout(tooltipTimer);
  if (tooltipNode) tooltipNode.hidden = true;
}

function setupHelpTooltips() {
  decorateHelpables();
  document.addEventListener("mouseover", (event) => {
    const target = resolveHelpTarget(event.target);
    if (target && target !== currentHelpTarget) showTooltip(target, false);
  });
  document.addEventListener("mouseout", (event) => {
    if (!currentHelpTarget) return;
    const related = event.relatedTarget;
    if (related && (currentHelpTarget.contains(related) || resolveHelpTarget(related) === currentHelpTarget)) return;
    hideTooltip();
  });
  document.addEventListener("click", (event) => {
    const target = resolveHelpTarget(event.target);
    if (target) showTooltip(target, true);
  });
  document.addEventListener("focusin", (event) => {
    const target = resolveHelpTarget(event.target);
    if (target) showTooltip(target, true);
  });
  document.addEventListener("focusout", hideTooltip);
  document.addEventListener("scroll", hideTooltip, true);
}


runButton.dataset.helpKey = "run_button";
runButton.classList.add("helpable");
runButton.addEventListener("click", runAnalysis);
form.addEventListener("submit", runAnalysis);
setupHelpTooltips();
