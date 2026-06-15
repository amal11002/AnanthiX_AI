// ============================================================
// AnanthiX AI — Frontend v2 (Sidebar Layout)
// ============================================================

const I18N = {
  fr: {
    ctrlLang: "Langue",
    ctrlModel: "Moteur de diagnostic",
    ctrlGradcam: "Zones analysées",
    ctrlGradcamHint: "Afficher sur l'image",
    navWorkspace: "Diagnostic",
    navAbout: "À propos",
    dropTitle: "Déposez une image de feuille",
    dropSub: "ou cliquez pour parcourir vos fichiers",
    dropHint: "PNG · JPG · 10 Mo max · 38 maladies · 14 espèces",
    loaderText: "Analyse en cours...",
    panelInputTitle: "Image analysée",
    panelDiagTitle: "Résultat du diagnostic",
    lblDisease: "MALADIE DÉTECTÉE",
    lblPlant: "PLANTE",
    lblSeverity: "GRAVITÉ",
    lblModel: "Moteur",
    metaFormat: "FORMAT",
    metaSize: "TAILLE",
    metaTime: "HEURE",
    confHigh: "Diagnostic fiable",
    confMed: "Diagnostic à vérifier",
    confLow: "Résultat non concluant",
    confDetailsLabel: "Voir le détail",
    confLabelStatic: "Score de confiance",
    rankSummary: "Autres hypothèses",
    gradcamLabel: "Zones analysées",
    gradcamHint: "Survolez pour voir les zones",
    btnNewLabel: "Analyser une autre image",
    btnPdfLabel: "Télécharger le rapport",
    tabOverview: "APERÇU",
    tabSymptoms: "SYMPTÔMES",
    tabTreatment: "TRAITEMENT",
    tabPrevention: "PRÉVENTION",
    healthyTitle: "Plante saine",
    healthyOverview: "Aucune maladie détectée. La plante présente les caractéristiques d'un feuillage en bonne santé.",
    uncertainTitle: "Résultat non concluant",
    uncertainCauses: "Causes possibles",
    uncertainActions: "Actions recommandées",
    causes: ["Qualité d'image insuffisante", "Espèce non couverte par le modèle", "Symptômes peu visibles"],
    actions: ["Capturer une image plus nette", "Améliorer l'éclairage", "Solliciter un expert agronome"],
    sevHealthy: "Saine",
    sevMild: "Légère",
    sevModerate: "Modérée",
    sevCritical: "Critique",
    aboutEyebrow: "À PROPOS",
    aboutTitle: "À propos d'AnanthiX AI",
    aboutLead: "Projet de recherche appliquée sur le diagnostic des maladies des plantes par deep learning.",
    aboutH1: "Mission", aboutP1: "AnanthiX AI fournit un diagnostic assisté par IA des maladies des plantes pour les professionnels agricoles, les agronomes et les producteurs. La plateforme combine une classification haute précision avec un raisonnement visuel explicable.",
    aboutH2: "Méthodologie", aboutP2: "Le moteur est un ResNet-50 fine-tuné sur PlantVillage (54 303 images, 38 classes, 14 espèces) et enrichi avec PlantDoc (2 670 images terrain, ×10).",
    aboutH3: "Explicabilité", aboutP3: "Chaque prédiction est accompagnée d'une visualisation des zones analysées et d'un seuil de confiance à trois niveaux.",
    aboutH5: "Limites", aboutP5: "38 classes sur 14 espèces. Précision terrain : 70 %. Validation agronomique requise pour toute décision de traitement.",
    aboutH6: "Projet", aboutP6: "Projet académique — UQAC, Atelier pratique en IA I. Auteure : Amal Ouedraogo. Encadrement : Prof. Julien Maitre, Ph.D.",
  },
  en: {
    ctrlLang: "Language",
    ctrlModel: "Diagnostic engine",
    ctrlGradcam: "Analysis zones",
    ctrlGradcamHint: "Show on image",
    navWorkspace: "Diagnosis",
    navAbout: "About",
    dropTitle: "Drop a leaf image",
    dropSub: "or click to browse your files",
    dropHint: "PNG · JPG · 10 MB max · 38 diseases · 14 species",
    loaderText: "Analysing...",
    panelInputTitle: "Analysed image",
    panelDiagTitle: "Diagnosis result",
    lblDisease: "DETECTED DISEASE",
    lblPlant: "PLANT",
    lblSeverity: "SEVERITY",
    lblModel: "Engine",
    metaFormat: "FORMAT",
    metaSize: "SIZE",
    metaTime: "TIME",
    confHigh: "Reliable diagnosis",
    confMed: "Verify with an expert",
    confLow: "Inconclusive result",
    confDetailsLabel: "See details",
    confLabelStatic: "Confidence score",
    rankSummary: "Other hypotheses",
    gradcamLabel: "Analysis zones",
    gradcamHint: "Hover to see zones",
    btnNewLabel: "Analyse another image",
    btnPdfLabel: "Download report",
    tabOverview: "OVERVIEW",
    tabSymptoms: "SYMPTOMS",
    tabTreatment: "TREATMENT",
    tabPrevention: "PREVENTION",
    healthyTitle: "Healthy plant",
    healthyOverview: "No disease detected. The plant shows characteristics of healthy foliage.",
    uncertainTitle: "Inconclusive result",
    uncertainCauses: "Possible causes",
    uncertainActions: "Recommended actions",
    causes: ["Insufficient image quality", "Plant species not covered by the model", "Insufficient visible symptoms"],
    actions: ["Capture a clearer image", "Improve lighting conditions", "Seek expert agronomic validation"],
    sevHealthy: "Healthy",
    sevMild: "Mild",
    sevModerate: "Moderate",
    sevCritical: "Critical",
    aboutEyebrow: "ABOUT",
    aboutTitle: "About AnanthiX AI",
    aboutLead: "An applied research project on plant disease diagnosis through deep learning.",
    aboutH1: "Mission", aboutP1: "AnanthiX AI provides AI-assisted plant disease diagnosis for agricultural professionals, agronomists, and commercial growers.",
    aboutH2: "Methodology", aboutP2: "The diagnostic engine is a fine-tuned ResNet-50 trained on PlantVillage (54,303 images, 38 classes, 14 species) and enriched with PlantDoc (2,670 field images, ×10).",
    aboutH3: "Explainability", aboutP3: "Each prediction is paired with an analysis zone visualisation and a three-tier confidence threshold.",
    aboutH5: "Limitations", aboutP5: "38 classes across 14 species. Field accuracy: 70%. Expert validation required for treatment decisions.",
    aboutH6: "Project", aboutP6: "Academic project — UQAC, Applied AI Workshop I. Author: Amal Ouedraogo. Supervisor: Prof. Julien Maitre, Ph.D.",
  },
};

const MODEL_INFO = {
  v2:       { short: "V2",       tag: "RECOMMANDÉ" },
  baseline: { short: "BASELINE", tag: "RÉFÉRENCE" },
};

const SEVERITY_MAP = {
  "saine":    { cls: "ok",   key: "sevHealthy" },
  "légère":   { cls: "mild", key: "sevMild" },
  "modérée":  { cls: "warn", key: "sevModerate" },
  "critique": { cls: "crit", key: "sevCritical" },
};

let state = { lang: "fr", model: "v2", gradcam: true, currentFile: null, gradcamPinned: false };
let lastApiResponse = null;

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

// ── i18n ──────────────────────────────────────────────────────
function setText(sel, val) { const el = $(sel); if (el) el.textContent = val; }

function setLang(lang) {
  state.lang = lang;
  const t = I18N[lang];
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const key = el.dataset.i18n;
    if (t[key] !== undefined) el.textContent = t[key];
  });
  setText("#lang-value", lang.toUpperCase());
  setText("#dropzone-title", t.dropTitle);
  setText("#dropzone-sub", t.dropSub);
  setText("#dropzone-hint", t.dropHint);
  setText("#loader-text", t.loaderText);
  setText("#panel-input-title", t.panelInputTitle);
  setText("#panel-diag-title", t.panelDiagTitle);
  setText("#lbl-disease", t.lblDisease);
  setText("#lbl-plant", t.lblPlant);
  setText("#lbl-severity", t.lblSeverity);
  setText("#lbl-model", t.lblModel);
  setText("#meta-format-label", t.metaFormat);
  setText("#meta-size-label", t.metaSize);
  setText("#meta-time-label", t.metaTime);
  setText("#conf-label-static", t.confLabelStatic);
  setText("#conf-details-label", t.confDetailsLabel);
  setText("#rank-summary", t.rankSummary);
  setText("#gradcam-btn-label", t.gradcamLabel);
  setText("#image-stage-hint", t.gradcamHint);
  setText("#btn-new-label", t.btnNewLabel);
  setText("#btn-pdf-label", t.btnPdfLabel);
  if ($("#workspace") && $("#workspace").classList.contains("visible") && lastApiResponse) {
    renderResults(lastApiResponse);
  }
}

function setModel(modelKey) {
  state.model = modelKey;
  $("#model-value").textContent = MODEL_INFO[modelKey].short;
}

// ── Tabs (sidebar links) ──────────────────────────────────────
$$(".sidebar-link").forEach(link => {
  link.addEventListener("click", e => {
    e.preventDefault();
    const target = link.dataset.tab;
    $$(".sidebar-link").forEach(l => l.classList.toggle("active", l.dataset.tab === target));
    $$(".tab-page").forEach(p => p.classList.toggle("active", p.id === `page-${target}`));
  });
});

// ── Mini selects ──────────────────────────────────────────────
function bindMiniSelect(id, onChange) {
  const root = $(id);
  if (!root) return;
  root.querySelector(".mini-select-btn").addEventListener("click", e => {
    e.stopPropagation();
    $$(".mini-select").forEach(s => s !== root && s.classList.remove("open"));
    root.classList.toggle("open");
  });
  root.querySelectorAll(".mini-select-item").forEach(item => {
    item.addEventListener("click", () => {
      root.querySelectorAll(".mini-select-item").forEach(i => i.classList.remove("selected"));
      item.classList.add("selected");
      root.classList.remove("open");
      onChange(item.dataset.value);
    });
  });
}
document.addEventListener("click", () => $$(".mini-select").forEach(s => s.classList.remove("open")));
bindMiniSelect("#lang-select", setLang);
bindMiniSelect("#model-select", setModel);

// ── Grad-CAM toggle ───────────────────────────────────────────
const gradcamToggle = $("#gradcam-toggle");
if (gradcamToggle) {
  gradcamToggle.addEventListener("change", e => {
    state.gradcam = e.target.checked;
    const btn = $("#gradcam-btn");
    if (!state.gradcam) {
      state.gradcamPinned = false;
      $("#image-stage")?.classList.remove("pinned");
      if (btn) btn.style.display = "none";
    } else if (lastApiResponse) {
      if (btn) btn.style.display = "";
    }
  });
}

const gradcamBtn = $("#gradcam-btn");
if (gradcamBtn) {
  gradcamBtn.addEventListener("click", () => {
    state.gradcamPinned = !state.gradcamPinned;
    $("#image-stage")?.classList.toggle("pinned", state.gradcamPinned);
    gradcamBtn.classList.toggle("active", state.gradcamPinned);
  });
}

// ── File upload ───────────────────────────────────────────────
const dropzone = $("#dropzone");
const fileInput = $("#file-input");

if (dropzone) {
  dropzone.addEventListener("dragover", e => { e.preventDefault(); dropzone.classList.add("drag-over"); });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));
  dropzone.addEventListener("drop", e => {
    e.preventDefault(); dropzone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
}
if (fileInput) fileInput.addEventListener("change", e => { if (e.target.files.length) handleFile(e.target.files[0]); });

function fmtBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(2) + " MB";
}

async function handleFile(file) {
  if (!file.type.startsWith("image/")) return;
  state.currentFile = file;

  $("#dropzone-state").style.display = "none";
  $("#workspace").classList.remove("visible");
  $("#loader").classList.add("active");

  const formData = new FormData();
  formData.append("image", file);
  formData.append("model", state.model);
  formData.append("gradcam", state.gradcam ? "true" : "false");

  try {
    const res = await fetch("/api/predict", { method: "POST", body: formData });
    if (!res.ok) throw new Error("API error " + res.status);
    const data = await res.json();
    lastApiResponse = data;
    renderResults(data);
  } catch (err) {
    console.error(err);
    alert("Erreur : " + err.message);
    $("#dropzone-state").style.display = "";
  } finally {
    $("#loader").classList.remove("active");
  }
}

// ── Render ────────────────────────────────────────────────────
function getConfidenceInfo(prob) {
  const t = I18N[state.lang];
  if (prob >= 0.85) return { label: t.confHigh, cls: "high", color: "#5DB870" };
  if (prob >= 0.50) return { label: t.confMed,  cls: "medium", color: "#C8A84B" };
  return { label: t.confLow, cls: "low", color: "#E07070" };
}

function renderResults(data) {
  const t = I18N[state.lang];
  const top = data.predictions[0];
  const lang = state.lang;

  // Images
  $("#img-input").src = data.input_image;
  const overlayEl = $("#img-gradcam");
  const stage = $("#image-stage");
  const gradBtn = $("#gradcam-btn");
  if (data.gradcam_image) {
    overlayEl.src = data.gradcam_image;
    stage.classList.remove("pinned");
    state.gradcamPinned = false;
    gradBtn.classList.remove("active");
    gradBtn.style.display = "";
  } else {
    overlayEl.removeAttribute("src");
    gradBtn.style.display = "none";
  }

  // Metadata
  if (state.currentFile) {
    $("#meta-format").textContent = state.currentFile.type.split("/")[1].toUpperCase();
    $("#meta-size").textContent = fmtBytes(state.currentFile.size);
    $("#meta-time").textContent = new Date().toLocaleTimeString(lang === "fr" ? "fr-FR" : "en-US", { hour: "2-digit", minute: "2-digit" });
  }

  // Diagnosis fields
  const card = data.card;
  if (card) {
    $("#val-disease").textContent = card[`disease_${lang}`];
    $("#val-plant").textContent   = card[`plant_${lang}`];
    const sev = SEVERITY_MAP[card.severity] || { cls: "mild", key: "sevMild" };
    const sevEl = $("#val-severity");
    sevEl.className = `severity-chip ${sev.cls}`;
    sevEl.innerHTML = `<span class="severity-dot"></span><span>${t[sev.key]}</span>`;
  } else {
    $("#val-disease").textContent = top.class;
    $("#val-plant").textContent   = "—";
    $("#val-severity").innerHTML  = "—";
  }
  $("#val-model").textContent = MODEL_INFO[data.model_used]?.short || data.model_used;

  // Confidence badge (simplifié)
  const ci = getConfidenceInfo(top.prob);
  const badge = $("#conf-badge");
  badge.textContent = ci.label;
  badge.className = `conf-badge ${ci.cls}`;
  $("#conf-fill").style.background = ci.color;
  $("#conf-fill").style.width = "0%";
  setTimeout(() => { $("#conf-fill").style.width = (top.prob * 100) + "%"; }, 50);
  $("#conf-value").textContent = (top.prob * 100).toFixed(2) + "%";

  // Top 3
  const tbody = $("#ranking-body");
  tbody.innerHTML = data.predictions.map((p, i) => `
    <tr>
      <td class="rank">${i + 1}</td>
      <td class="cls">${p.class}</td>
      <td class="bar-cell"><div class="bar"><div class="bar-fill" style="width:${p.prob*100}%"></div></div></td>
      <td class="pct">${(p.prob*100).toFixed(1)}%</td>
    </tr>
  `).join("");

  // Disease tabs
  const wrap = $("#disease-tabs-wrap");
  if (top.prob < 0.50) {
    wrap.innerHTML = renderUncertain();
  } else if (card) {
    wrap.innerHTML = renderDiseaseTabs(card);
    bindReportTabs();
  } else {
    wrap.innerHTML = "";
  }

  // Bouton PDF
  const btnPdf = $("#btn-pdf");
  btnPdf.style.display = "flex";
  btnPdf.onclick = () => downloadPdf(data);

  $("#workspace").classList.add("visible");
}

function renderUncertain() {
  const t = I18N[state.lang];
  return `
    <div class="uncertain-card">
      <h3><span class="pulse"></span>${t.uncertainTitle}</h3>
      <div class="uncertain-block">
        <div class="label">${t.uncertainCauses}</div>
        <ul>${t.causes.map(c => `<li>${c}</li>`).join("")}</ul>
      </div>
      <div class="uncertain-block">
        <div class="label">${t.uncertainActions}</div>
        <ul>${t.actions.map(a => `<li>${a}</li>`).join("")}</ul>
      </div>
    </div>`;
}

function renderDiseaseTabs(card) {
  const t = I18N[state.lang];
  const lang = state.lang;
  if (card.is_healthy) {
    return `<div class="report-overview"><h3>${t.healthyTitle}</h3><p>${t.healthyOverview}</p></div>`;
  }
  return `
    <div class="report-tabs">
      <button class="report-tab active" data-rtab="overview">${t.tabOverview}</button>
      <button class="report-tab" data-rtab="symptoms">${t.tabSymptoms}</button>
      <button class="report-tab" data-rtab="treatment">${t.tabTreatment}</button>
      <button class="report-tab" data-rtab="prevention">${t.tabPrevention}</button>
    </div>
    <div class="report-content active" data-rcontent="overview">
      <div class="report-overview">
        <h3>${card[`disease_${lang}`]}</h3>
        <p>${card[`plant_${lang}`]} · ${(card[`symptoms_${lang}`] || []).slice(0,1).join("")}</p>
      </div>
    </div>
    <div class="report-content" data-rcontent="symptoms">
      <ul class="report-list">${(card[`symptoms_${lang}`]||[]).map(s=>`<li>${s}</li>`).join("")}</ul>
    </div>
    <div class="report-content" data-rcontent="treatment">
      <ul class="report-list">${(card[`treatment_${lang}`]||[]).map(s=>`<li>${s}</li>`).join("")}</ul>
    </div>
    <div class="report-content" data-rcontent="prevention">
      <ul class="report-list">${(card[`prevention_${lang}`]||[]).map(s=>`<li>${s}</li>`).join("")}</ul>
    </div>`;
}

function bindReportTabs() {
  $$(".report-tab").forEach(btn => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.rtab;
      $$(".report-tab").forEach(b => b.classList.toggle("active", b.dataset.rtab === target));
      $$(".report-content").forEach(c => c.classList.toggle("active", c.dataset.rcontent === target));
    });
  });
}

// ── PDF download (appel serveur) ──────────────────────────────
async function downloadPdf(data) {
  try {
    const res = await fetch("/api/report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data, lang: state.lang })
    });
    if (!res.ok) throw new Error("PDF error");
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "rapport_ananthix.pdf";
    document.body.appendChild(a); a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (err) {
    console.error(err);
    alert("Erreur lors de la génération du rapport.");
  }
}

// ── Init ──────────────────────────────────────────────────────
setLang("fr");
setModel("v2");