import { PresentationFile, FileBlob } from "@oai/artifact-tool";

const ROOT = "/Users/corentinplumet/Documents/RL_Marl2grid";
const BUILD = `${ROOT}/tmp/thesis_defense_build`;
const FIG = `${ROOT}/latex/figures`;
const STARTER = `${BUILD}/epfl_inspect/template-starter.pptx`;
const OUTPUT = `${ROOT}/master_thesis_defense_EPFL.pptx`;

const P = {
  graphBusbar: `${FIG}/graph_busbar_baseline.png`,
  graphLine: `${FIG}/graph_heterogeneous_line_nodes.png`,
  graphEquipment: `${FIG}/graph_heterogeneous_equipment.png`,
  sparse: `${FIG}/sparse_control_fulltest_tradeoff.png`,
  actor: `${BUILD}/graph_actor_pipeline.png`,
  scorer: `${BUILD}/candidate_action_scoring_pipeline.png`,
  placeholder: `${BUILD}/encoder_transfer_placeholder.png`,
  sourceResult: `${FIG}/nl_cas_hl_fulltest.png`,
  featureRaw: `${FIG}/wcci_feature_distributions.png`,
  featureFixed: `${FIG}/wcci_feature_corrected.png`,
  targetControls: `${FIG}/wcci_target_controls.png`,
  zeroShot: `${FIG}/wcci_zero_shot_determinants.png`,
  architecture: `${FIG}/wcci_zero_shot_architecture_effects.png`,
  adaptation: `${FIG}/wcci_adaptation_summary.png`,
  gate: `${FIG}/wcci_intervention_gate_effects.png`,
  coverage: `${FIG}/wcci_appendix_coverage.png`,
  chronics: `${FIG}/wcci_appendix_chronics.png`,
  depthWidth: `${FIG}/gs_s3dw_depth_width_heatmap.png`,
  readout: `${FIG}/gs_s3p_bar.png`,
  structure: `${FIG}/gs_s2_structure_bar.png`,
  direction: `${FIG}/gs_hmd_direction_seed_heatmap.png`,
  matrix: `${FIG}/wcci_appendix_zero_shot_matrix.png`,
  targetBoard: `${FIG}/wcci_appendix_target_board.png`,
  actionHead: `${FIG}/wcci_action_head.png`,
  actionPreprocess: `${FIG}/candidate_action_preprocessing_fulltest.png`,
};

const presentation = await PresentationFile.importPptx(await FileBlob.load(STARTER));

function named(items, name) {
  const norm = (value) => String(value).replace(/\u00a0/g, " ").replace(/\s+/g, " ").trim();
  return items.filter((item) => norm(item.name) === norm(name));
}

function setText(slide, name, value, index = 0) {
  const item = named(slide.shapes.items, name)[index];
  if (!item) throw new Error(`Missing text shape '${name}' on slide ${slide.index ?? "?"}`);
  item.text = value;
  return item;
}

function setTexts(slide, name, values) {
  const items = named(slide.shapes.items, name);
  if (items.length < values.length) {
    throw new Error(`Expected ${values.length} shapes '${name}', found ${items.length}`);
  }
  values.forEach((value, i) => { items[i].text = value; });
}

function deleteNamed(slide, ...names) {
  for (const item of [...slide.shapes.items, ...slide.images.items]) {
    if (names.includes(item.name)) item.delete();
  }
}

function setChrome(slide, number) {
  for (const item of [...slide.shapes.items]) {
    if (/date/i.test(item.name)) item.text = "MASTER THESIS";
    if (/pied de page/i.test(item.name)) item.delete();
    if (/numéro de diapositive/i.test(item.name)) item.text = String(number);
  }
}

async function replaceImage(slide, name, path, index = 0, alt = "Thesis figure") {
  const image = named(slide.images.items, name)[index];
  if (!image) throw new Error(`Missing image '${name}' on slide ${slide.index ?? "?"}`);
  const frame = image.frame;
  const borderRadius = image.borderRadius;
  const rotation = image.rotation;
  const flipHorizontal = image.flipHorizontal;
  const flipVertical = image.flipVertical;
  const lockAspectRatio = image.lockAspectRatio;
  image.delete();
  const source = await FileBlob.load(path);
  const replacement = slide.images.add({
    name,
    blob: source.data,
    contentType: source.mime,
    alt,
    fit: "contain",
    position: frame,
    crop: { left: 0, top: 0, right: 0, bottom: 0 },
  });
  replacement.borderRadius = borderRadius;
  replacement.rotation = rotation;
  replacement.flipHorizontal = flipHorizontal;
  replacement.flipVertical = flipVertical;
  replacement.lockAspectRatio = lockAspectRatio;
  return replacement;
}

function setImageFrame(slide, name, frame, index = 0) {
  const image = named(slide.images.items, name)[index] ?? slide.images.items.at(-1);
  if (!image) throw new Error(`Missing image '${name}' for frame update`);
  image.frame = frame;
}

function setNotes(slide, talk, sources) {
  slide.speakerNotes.clear();
  slide.speakerNotes.textFrame.setText(`${talk}\n\n[Sources]\n${sources.map((s) => `- ${s}`).join("\n")}`);
  slide.speakerNotes.setVisible(true);
}

const S = presentation.slides.items;

// 1 — Title
setText(S[0], "Espace réservé du texte 3", "MASTER THESIS · AUGUST 2026");
setText(S[0], "Titre 2", "Graph policies for\npower-grid topology control");
setText(S[0], "Espace réservé du texte 4", "EPFL × NUS\n\nMaster thesis\n2026");
setText(S[0], "Espace réservé du texte 5", "PRESENTED BY\n\nCorentin Plumet\nMaster thesis");
setNotes(S[0], "Open with the thesis question: can one decentralised graph policy cross from bus14 to a much larger power grid? State that the answer depends on the complete policy interface, not only the encoder.", [
  "latex/main.tex",
  "latex/chapters/01_introduction.tex",
]);

// 2 — Roadmap
setText(S[1], "Titre 2", "THESIS STORY");
setText(S[1], "Espace réservé du texte 3", "Why topology control?\nA policy that can cross grids\nEvidence: bus14 → WCCI\nWhat transfers — and what does not");
await replaceImage(S[1], "Espace réservé pour une image  8", P.graphLine, 0, "Heterogeneous power-grid graph representation");
setImageFrame(S[1], "Espace réservé pour une image  8", { left: 647, top: 2, width: 631, height: 716 });
setNotes(S[1], "Give the audience the non-chronological story: problem, complete transferable interface, source validation, target evidence, limitations.", [
  "latex/chapters/01_introduction.tex",
  "latex/chapters/03_marl_graph_control.tex",
]);

// 3 — Motivation
setText(S[2], "Espace réservé du contenu 1", "• Reconfigure busbars to redirect power flows\n• Uses switching equipment already installed\n• Decisions interact over time and cooldowns\n\n178 unitary actions on bus14\n≈66,965 on WCCI");
setText(S[2], "Titre 2", "Topology control is cheap to actuate — hard to search");
await replaceImage(S[2], "Picture 1", P.graphBusbar, 0, "Busbar graph representation");
await replaceImage(S[2], "Picture 9", P.graphLine, 0, "Line-node heterogeneous grid graph");
setNotes(S[2], "Explain the action: move elements between two busbars at a substation. The switching equipment exists; the difficulty is combinatorial, sequential, and safety-critical.", [
  "latex/chapters/02_background.tex",
  "latex/chapters/07_wcci_transfer.tex",
]);

// 4 — State of the art
setText(S[3], "Espace réservé du contenu 1", "Single-agent & hierarchical\n• Strong benchmark results\n• Reduced or factorised action sets\n• Often centralised at execution");
setText(S[3], "Espace réservé du contenu 2", "Heuristics & MARL\n• Safety rules decide when to act\n• Regional actors match grid operation\n• Partial observability remains");
setText(S[3], "Espace réservé du contenu 4", "Graph policies\n• Parameters can ignore grid size\n• Structure enters message passing\n• Most action heads still use fixed indices");
setText(S[3], "Titre 3", "THE GAP: PORTABLE INPUTS ARE NOT ENOUGH IF THE ACTION INTERFACE STAYS GRID-SPECIFIC");
deleteNamed(S[3], "ZoneTexte 11");
await replaceImage(S[3], "Image 9", P.graphLine, 0, "Graph representation of the grid");
setNotes(S[3], "Position the work at the intersection of MARL, graph policies, and action reduction. The under-addressed problem is cross-grid transfer when candidate action identities and counts change.", [
  "latex/chapters/02_background.tex",
  "latex/chapters/03_marl_graph_control.tex",
]);

// 5 — Objectives
setText(S[4], "Espace réservé du contenu 7", "1. LOCAL\nDECENTRALISED EXECUTION");
setText(S[4], "Espace réservé du contenu 2", "2. SHARED\nGRAPH STATE ENCODER");
setText(S[4], "Espace réservé du contenu 1", "3. PORTABLE\nCANDIDATE SCORER");
setText(S[4], "Titre 3", "PORTABLE");
deleteNamed(S[4], "!!!Image");
await replaceImage(S[4], "Image 8", P.graphBusbar, 0, "Decentralised busbar graph");
await replaceImage(S[4], "Image 9", P.graphLine, 0, "Shared heterogeneous graph encoder");
await replaceImage(S[4], "Image 10", P.graphEquipment, 0, "Equipment-aware graph representation");
setNotes(S[4], "Frame these as engineering requirements. A transferable encoder solves only the middle requirement; the action scorer is required for a different number of candidates.", [
  "latex/chapters/03_marl_graph_control.tex",
  "latex/chapters/06_candidate_action_scoring.tex",
]);

// 6 — Experimental frame
setText(S[5], "Titre 3", "TWO GRIDS,\nONE POLICY");
await replaceImage(S[5], "Espace réservé pour une image  8", P.coverage, 0, "Action-space coverage across WCCI candidate caps");
setImageFrame(S[5], "Espace réservé pour une image  8", { left: 0, top: 0, width: 534, height: 720 });
setTexts(S[5], "Espace réservé du contenu 2", [
  "14\nSOURCE",
  "36\nTARGET",
  "3 → 4\nAGENTS",
  "201 / 50\nCHRONICS",
]);
for (const metric of named(S[5].shapes.items, "Espace réservé du contenu 2")) {
  metric.text.style = { fontSize: 28, bold: true, color: "#3A3838" };
}
setText(S[5], "ZoneTexte 1", "Transfer uses the same 50 WCCI chronics; 24 are difficult because do nothing fails.");
deleteNamed(S[5], "Image 9", "Picture 2", "Image 11", "Image 12");
setNotes(S[5], "The source grid is used for design screens and source pretraining. WCCI is both larger and operationally different. The fixed cohort makes transfer comparisons auditable.", [
  "latex/chapters/05_experiments_results.tex",
  "latex/chapters/07_wcci_transfer.tex",
]);

const mainTechSlides = [
  [6, "Safe control means knowing when not to act", "• Survival alone rewards needless switching\n• A local ρ rule blocks actions while the region is safe\n• Local-safe adaptive budgets reach 98.4% survival with 0.98 idle-action usage", P.sparse, "Source-grid survival versus intervention trade-off", "Introduce the safety layer as a decision-time mechanism. It suppresses unnecessary actions, but later evidence will show that its value is policy-dependent."],
  [7, "A shared local graph replaces the fixed input vector", "• Each actor receives its regional graph plus one-hop context\n• GINE shares weights across all agents\n• MAPPO critic uses global state only during training", P.actor, "Graph actor and critic pipeline", "Walk left-to-right: local observation, graph construction, shared GINE actor, decentralised action. The global critic is training-only."],
  [8, "Candidate scoring removes the fixed action head", "• Gather the node rows touched by candidate j\n• Pool them into an action context cⱼ\n• A shared scorer emits one logit per candidate — on any grid", P.scorer, "Candidate action scoring pipeline", "The scorer is permutation-equivariant over candidates and does not allocate one output parameter per action index. This is the key action-side transfer mechanism."],
  [9, "The transferable actor first has to solve the source grid", "• All leading Screen-F variants exceed 98%\n• Best full-test survival: 99.52%\n• Source-grid accuracy does not predict target-grid learning", P.sourceResult, "Source-grid candidate scorer full-test survival", "Establish competence before transfer. Emphasise that several source variants are nearly tied, so target transfer is needed to distinguish them."],
  [10, "Physical inputs must mean the same thing on both grids", "• Absolute power scales shift with grid capacity\n• Absolute voltage angle is reference-dependent\n• Use p / max Pmax and angle differences\n• Freeze source statistics for strict zero-shot", P.featureFixed, "Corrected feature distributions across source and target grids", "Explain the two portable transformations. These remove avoidable scale and reference choices, but do not eliminate every distribution shift."],
  [11, "Training directly on WCCI is surprisingly weak", "At k = 64, overall family means:\n• Flat MLP: 32.6%\n• Fixed-head GNN: 38.4%\n• Candidate scorer: 47.2%\n\nAll remain below 56.0% do nothing overall.", P.targetControls, "WCCI scratch-control comparison", "Do not hide the awkward baseline: overall survival is dominated by easy chronics. The candidate scorer is best among learned families, yet scratch learning remains unstable."],
  [12, "Encoder-only transfer: the key diagnostic is still missing", "RESULTS REQUIRED BEFORE DEFENCE\n\nCompare four source encoders under:\n• target scratch\n• random frozen encoder\n• NL frozen transfer\n• NLS frozen transfer", P.placeholder, "Placeholder for the unfinished encoder-only transfer experiment", "Be explicit that this is a planned result, not evidence. If the study is not complete, remove this main slide or present only the protocol question."],
  [13, "Zero-shot works best when the decision set stays small", "With local ρ = 0.95:\n• k = 32: 62.5% overall\n• 21.9% on difficult cases\n• 54.6% of feasible gain recovered\n\nAt k = 256, recovered gain falls to 2.9%.", P.zeroShot, "Determinants of WCCI zero-shot transfer", "Larger candidate sets improve feasibility but make ranking harder. The policy has a finite ranking capacity: k=32 is the clearest zero-shot operating point."],
  [14, "Typed candidate pooling is the clearest transfer gain", "Matched difficult-cohort effects:\n• +5.4 pp ungated\n• +8.7 pp with local gate\n• Physical scaling ≈ +1 pp\n\nThe dedicated idle head is harmful when ungated.", P.architecture, "Matched architecture effects in zero-shot transfer", "Separate the effects. Typed mean pooling is the most repeatable architectural gain. Feature scaling helps modestly. The idle mechanism interacts negatively without gating."],
  [15, "Source-initialised fine-tuning makes transfer reliable", "Median difficult-cohort survival:\n• 5.7% zero-shot\n• 9.6% scratch\n• 18.1% fine-tuned\n\nFine-tuning wins 7/8 matched comparisons vs scratch.", P.adaptation, "Adaptation summary on difficult WCCI chronics", "Finish the evidence section with the robust route: source initialise and fine-tune. Keep the claim matched—same architecture and action cap."],
];

for (const [idx, title, body, imagePath, alt, talk] of mainTechSlides) {
  setText(S[idx], "Title 1", title);
  setText(S[idx], "Espace réservé du contenu 7", body);
  await replaceImage(S[idx], "Picture 6", imagePath, 0, alt);
  setNotes(S[idx], talk, [
    idx <= 10 ? "latex/chapters/05_experiments_results.tex" : "latex/chapters/07_wcci_transfer.tex",
    idx === 12 ? "latex/chapters/09_encoder_transfer_bridge.tex" : `latex/figures/${imagePath.split("/").pop()}`,
  ]);
}

// 17 — Interpretation
setText(S[16], "Espace réservé du contenu 1", "WHAT TRANSFERRED\n• Shared GINE encoder\n• Candidate-wise scorer\n• Typed action context\n• Small reduced action sets");
setText(S[16], "Espace réservé du contenu 2", "WHAT DID NOT\n• Fixed-index heads\n• Raw physical scales\n• Larger k without ranking capacity\n• Universal safety rule");
setText(S[16], "Espace réservé du contenu 4", "EVIDENCE LIMITS\n• One training seed\n• 50 WCCI chronics\n• Mixed target budgets\n• No completed encoder-only endpoint");
setText(S[16], "Titre 3", "TRANSFER NEEDS PORTABLE INTERFACES");
deleteNamed(S[16], "ZoneTexte 11");
await replaceImage(S[16], "Image 9", P.graphLine, 0, "Portable grid graph representation");
setNotes(S[16], "Synthesize the evidence and its limits. The strongest conclusion concerns compatible representations and decision interfaces; it does not establish universal graph generalisation.", [
  "latex/chapters/08_discussion_outlook.tex",
  "latex/chapters/09_encoder_transfer_bridge.tex",
]);

// 18 — Summary
setText(S[17], "Titre 2", "Three conclusions\n\n1. Make state and actions portable\n\n2. Validate safety per policy\n\n3. Fine-tune conservatively on the target grid");
setNotes(S[17], "Land the talk with the three reusable lessons. The next step is a completed, seed-matched encoder-only study and broader target-grid validation.", [
  "latex/chapters/08_discussion_outlook.tex",
]);

// 19 — Questions
setText(S[18], "Espace réservé du texte 3", "MASTER THESIS");
setText(S[18], "Titre 2", "Questions?");
setText(S[18], "Espace réservé du texte 5", "PRESENTED BY\n\nCorentin Plumet\nEPFL × NUS");
setNotes(S[18], "Pause here. Use the supplementary slides as return targets for methods, baselines, and chronic-level questions.", ["latex/main.tex"]);

// 20 — Supplement divider
setText(S[19], "Titre 2", "SUPPLEMENTARY");
setText(S[19], "Espace réservé du texte 3", "Protocol & baselines\nGraph-design screens\nTransfer diagnostics");
setNotes(S[19], "Backup slides start here; they are not part of the 20-minute sequence.", ["latex/main.tex"]);

// 21 — Protocol
setText(S[20], "Titre 3", "WCCI\nEVALUATION");
await replaceImage(S[20], "Espace réservé pour une image  8", P.coverage, 0, "WCCI action-cap feasibility coverage");
setImageFrame(S[20], "Espace réservé pour une image  8", { left: 0, top: 0, width: 534, height: 720 });
setTexts(S[20], "Espace réservé du contenu 2", [
  "50\nCHRONICS",
  "26\nEASY",
  "24\nDIFFICULT",
  "8.34%\nBASELINE",
]);
for (const metric of named(S[20].shapes.items, "Espace réservé du contenu 2")) {
  metric.text.style = { fontSize: 28, bold: true, color: "#3A3838" };
}
setText(S[20], "ZoneTexte 1", "Maintenance is disabled to remove a hazard absent from bus14; on the full WCCI test it changes do-nothing survival from 27.3% to 57.3%.");
deleteNamed(S[20], "Image 9", "Picture 2", "Image 11", "Image 12");
setNotes(S[20], "Use this slide to defend the cohort split and maintenance choice. Report difficult-cohort results whenever possible because overall survival saturates on easy chronics.", [
  "latex/chapters/07_wcci_transfer.tex",
]);

// 22 — Action spaces: retain the EPFL frame, replace the busy metric collage.
setText(S[21], "Titre 1", "More candidates help greedy — not the policy");
for (const item of [...S[21].shapes.items]) {
  if (!["Titre 1", "Espace réservé de la date 2", "Espace réservé du pied de page 3", "Espace réservé du numéro de diapositive 4"].includes(item.name)) item.delete();
}
for (const item of [...S[21].images.items]) item.delete();
const actionCards = [
  { x: 55, fill: "#E5E5E5", color: "#222222", text: "FULL LIST\n\n≈66,965\n\nactions" },
  { x: 350, fill: "#3D74B7", color: "#FFFFFF", text: "k = 32\n\n128\ncandidates\n\n33.1% greedy" },
  { x: 645, fill: "#12868B", color: "#FFFFFF", text: "k = 64\n\n256\ncandidates\n\n67.9% greedy" },
  { x: 940, fill: "#FF0000", color: "#FFFFFF", text: "k = 256\n\n716\ncandidates\n\n94.9% greedy" },
];
for (const card of actionCards) {
  const box = S[21].shapes.add({
    geometry: "roundRect",
    position: { left: card.x, top: 210, width: 255, height: 350 },
    fill: card.fill,
    line: { style: "solid", fill: card.fill, width: 1 },
    borderRadius: "rounded-xl",
  });
  box.text = card.text;
  box.text.style = { fontSize: 26, bold: true, color: card.color, alignment: "center", verticalAlignment: "middle" };
}
const actionTakeaway = S[21].shapes.add({
  geometry: "textbox",
  position: { left: 55, top: 590, width: 1140, height: 55 },
  fill: "none",
  line: { style: "solid", fill: "none", width: 0 },
});
actionTakeaway.text = "Feasible actions expand faster than the learned policy can rank them.";
actionTakeaway.text.style = { fontSize: 20, bold: true, color: "#222222", alignment: "center" };
setNotes(S[21], "The reduced list is decentralised, so total candidates exceed k. Greedy feasibility rises with k; policy performance does not, which diagnoses ranking difficulty rather than action absence.", [
  "latex/chapters/07_wcci_transfer.tex",
  "latex/figures/wcci_appendix_coverage.png",
]);

const backupTech = [
  [22, "Sparse-control mechanisms occupy different trade-offs", "Best source-grid learned control:\n• Local-safe AIB: 98.4% survival\n• 0.98 mean idle-action usage\n\nThe local rule improves sparse behaviour at evaluation time.", P.sparse, "Detailed sparse-control trade-off", "Answer questions about intervention frequency, survival, and the different heuristic variants."],
  [23, "Two GINE layers and width 128 were the robust default", "• K = 2, h = 128 crosses 90% earliest\n• Highest time-weighted training mean\n• Lower late-training variance\n• Peak gaps remain within one-seed noise", P.depthWidth, "GINE depth-width screen", "The selection criterion was not only the single best checkpoint; it combined learning speed, training mean, and stability."],
];
for (const [idx, title, body, imagePath, alt, talk] of backupTech) {
  setText(S[idx], "Title 1", title);
  setText(S[idx], "Espace réservé du contenu 7", body);
  await replaceImage(S[idx], "Picture 6", imagePath, 0, alt);
  setNotes(S[idx], talk, ["latex/chapters/05_experiments_results.tex", `latex/figures/${imagePath.split("/").pop()}`]);
}

// 25 — readout + structure
setText(S[24], "Titre 2", "Readout > structural hubs");
setText(S[24], "Espace réservé du contenu 1", "SCREEN C\nEnergized/max readout\n99.39% full-test survival");
setText(S[24], "Espace réservé du contenu 3", "SCREEN D\nSimple unaugmented graph retained\nVirtual nodes: 18–77%");
deleteNamed(S[24], "Picture 4", "Image 9");
await replaceImage(S[24], "Picture 2", P.readout, 0, "Readout aggregation comparison");
await replaceImage(S[24], "Image 8", P.structure, 0, "Graph-structure augmentation comparison");
setNotes(S[24], "The energized/max readout was a clear gain. Most explicit structural hubs destabilised training or reduced survival, so the simpler graph was retained.", [
  "latex/chapters/05_experiments_results.tex",
  "latex/figures/gs_s3p_bar.png",
  "latex/figures/gs_s2_structure_bar.png",
]);

// 26 — direction + scoring
setText(S[25], "Titre 2", "Direction + action context");
setText(S[25], "Espace réservé du contenu 1", "SCREEN E\nBest physical direction:\ngenerator → busbar\nload ← busbar\n99.12%");
setText(S[25], "Espace réservé du contenu 3", "SCREEN F\nLeading candidate scorers\n98–99.5% on bus14");
deleteNamed(S[25], "Picture 4", "Image 9");
await replaceImage(S[25], "Picture 2", P.direction, 0, "Heterogeneous graph message-direction screen");
await replaceImage(S[25], "Image 8", P.sourceResult, 0, "Candidate scorer source-grid comparison");
setNotes(S[25], "Direction encodes physical asymmetry and changes performance. Candidate scoring then provides the grid-size-independent action interface.", [
  "latex/chapters/05_experiments_results.tex",
  "latex/chapters/06_candidate_action_scoring.tex",
]);

// 27 — features
setText(S[26], "Titre 2", "Scale physical inputs");
setText(S[26], "Espace réservé du contenu 1", "BEFORE\nPower and angle channels differ in scale, offset, and tails");
setText(S[26], "Espace réservé du contenu 3", "AFTER\np / max Pmax and angle differences remove avoidable mismatch");
deleteNamed(S[26], "Picture 4", "Image 9");
await replaceImage(S[26], "Picture 2", P.featureRaw, 0, "Raw source-target feature distributions");
await replaceImage(S[26], "Image 8", P.featureFixed, 0, "Corrected source-target feature distributions");
setNotes(S[26], "Use the before-and-after view to explain why portability is a semantics problem, not only numerical standardisation.", [
  "latex/chapters/07_wcci_transfer.tex",
  "latex/figures/wcci_feature_distributions.png",
  "latex/figures/wcci_feature_corrected.png",
]);

// 28 — zero-shot matrices
setText(S[27], "Titre 2", "Zero-shot interactions");
setText(S[27], "Espace réservé du contenu 1", "PLAIN ZERO-SHOT\nPooling, preprocessing and idle-head choices interact with k");
setText(S[27], "Espace réservé du contenu 3", "ACTION-SPACE COVERAGE\nLarger k expands feasibility faster than the policy can rank it");
deleteNamed(S[27], "Picture 4", "Image 9");
await replaceImage(S[27], "Picture 2", P.matrix, 0, "Zero-shot transfer matrix");
await replaceImage(S[27], "Image 8", P.coverage, 0, "Candidate coverage and greedy feasibility");
setNotes(S[27], "This slide is for detailed architecture questions. Avoid reading every cell; point to interactions and the mismatch between feasibility and policy ranking.", [
  "latex/chapters/07_wcci_transfer.tex",
  "latex/figures/wcci_appendix_zero_shot_matrix.png",
]);

const finalBackupTech = [
  [28, "Decision-time changes help only in matched policy contexts", "Matched difficult-cohort change:\n• Gmax-delta: +4.6 pp ungated\n• Adaptive budget: +6.0 pp ungated\n• Local gate: +7.2 pp for plain zero-shot\n• But −2.2 pp for scratch scorer", P.gate, "Matched effect of decision-time interventions", "The gate is not universally safe. It helps zero-shot policies but can suppress useful actions for policies trained to act without it."],
  [29, "Aggregate gains hide chronic-specific failures", "• Some difficult chronics remain near do nothing\n• Greedy shows useful actions exist\n• Ranking and long-horizon credit remain the bottlenecks", P.chronics, "Per-chronic WCCI performance", "Use this slide to show that mean gains are uneven. The greedy reference separates lack of feasible actions from failure to rank them."],
];
for (const [idx, title, body, imagePath, alt, talk] of finalBackupTech) {
  setText(S[idx], "Title 1", title);
  setText(S[idx], "Espace réservé du contenu 7", body);
  await replaceImage(S[idx], "Picture 6", imagePath, 0, alt);
  setNotes(S[idx], talk, ["latex/chapters/07_wcci_transfer.tex", `latex/figures/${imagePath.split("/").pop()}`]);
}

// 31 — missing encoder outputs
setText(S[30], "Espace réservé du contenu 1", "PROTOCOL\n• action cap + data path\n• target step budget\n• source/target seed set");
setText(S[30], "Espace réservé du contenu 2", "OPTIMISATION\n• head optimiser + LR\n• checkpoint selection\n• gate condition");
setText(S[30], "Espace réservé du contenu 4", "OUTPUTS\n• learning curves\n• difficult endpoint table\n• 4 × 4 route/representation plot");
setText(S[30], "Titre 3", "COMPLETE THE MATCHED STUDY");
deleteNamed(S[30], "ZoneTexte 11");
await replaceImage(S[30], "Image 9", P.placeholder, 0, "Checklist for missing encoder-only results");
setNotes(S[30], "This is the exact completion checklist. The thesis chapter currently contains pending protocol fields and placeholder results, so no numerical claim should be made yet.", [
  "latex/chapters/09_encoder_transfer_bridge.tex",
]);

// 32 — reproducibility
setText(S[31], "Espace réservé du contenu 1", "SOURCE SCREENS\n15M environment steps\n201 held-out chronics\nbest periodic checkpoint");
setText(S[31], "Espace réservé du contenu 2", "WCCI TRANSFER\nfixed 50 chronics\n24 difficult / 26 easy\nseed 0");
setText(S[31], "Espace réservé du contenu 4", "COMPARISON RULES\nsame k and architecture\nreport overall + difficult\nshow do nothing + greedy");
setText(S[31], "Titre 3", "COMPARE LIKE WITH LIKE");
deleteNamed(S[31], "ZoneTexte 11");
await replaceImage(S[31], "Image 9", P.targetBoard, 0, "WCCI target training result board");
setNotes(S[31], "Use this slide to answer protocol and reproducibility questions. State clearly where evidence is single-seed and where intervals are across architectures rather than training seeds.", [
  "latex/chapters/05_experiments_results.tex",
  "latex/chapters/07_wcci_transfer.tex",
]);

// 33 — definitions
setText(S[32], "Espace réservé du contenu 1", "MEAN SURVIVAL\nsurvived steps / episode length, averaged over chronics");
setText(S[32], "Espace réservé du contenu 2", "LOCAL ρ GATE\nblock an agent action when its local maximum line loading < 0.95");
setText(S[32], "Espace réservé du contenu 4", "FEASIBLE GAIN\nprogress from do nothing to the same-k greedy reference");
setText(S[32], "Titre 3", "THREE DIFFERENT QUESTIONS");
deleteNamed(S[32], "ZoneTexte 11");
await replaceImage(S[32], "Image 9", P.actionHead, 0, "Action-head comparison on WCCI");
setNotes(S[32], "Definitions for metric questions. Feasible gain is normalised by the do-nothing-to-greedy interval at the same candidate cap.", [
  "latex/chapters/04_sparse_control.tex",
  "latex/chapters/07_wcci_transfer.tex",
]);

// 34 — return question slide
setText(S[33], "Espace réservé du texte 3", "MASTER THESIS");
setText(S[33], "Titre 2", "Questions?");
setText(S[33], "Espace réservé du texte 5", "PRESENTED BY\n\nCorentin Plumet\nEPFL × NUS");
setNotes(S[33], "Return target after a supplementary answer.", ["latex/main.tex"]);

// Standardise page chrome after slide-local edits.
for (let i = 0; i < S.length; i += 1) setChrome(S[i], i + 1);
// This half-image roadmap layout stores its date/page placeholders above the canvas.
deleteNamed(S[1], "Espace réservé de la date 4", "Espace réservé du numéro de diapositive 6");

const out = await PresentationFile.exportPptx(presentation);
await out.save(OUTPUT);
console.log(OUTPUT);
