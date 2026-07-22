import fs from "node:fs/promises";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT = "/Users/corentinplumet/Documents/RL_Marl2grid/outputs";

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

const pres = Presentation.create({ slideSize: { width: 1280, height: 720 } });
let slide = pres.slides.add();
slide.background.fill = "#FFFFFF";

function textBox(name, text, x, y, w, h, style = {}) {
  const s = slide.shapes.add({
    geometry: "textbox",
    name,
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  s.text = text;
  s.text.style = {
    fontSize: style.fontSize ?? 16,
    bold: style.bold ?? false,
    color: style.color ?? "#1B2633",
    alignment: style.alignment ?? "left",
    verticalAlignment: style.verticalAlignment ?? "middle",
    fontFamily: "Aptos",
  };
  return s;
}

function block(name, title, body, x, y, w, h, fill, stroke, accent, titleSize = 17, bodySize = 13) {
  const s = slide.shapes.add({
    geometry: "roundRect",
    name,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: stroke, width: 1.5 },
    borderRadius: 12,
    shadow: "shadow-sm",
  });
  const bar = slide.shapes.add({
    geometry: "roundRect",
    name: `${name}-accent`,
    position: { left: x, top: y, width: 7, height: h },
    fill: accent,
    line: { style: "solid", fill: accent, width: 0 },
    borderRadius: 8,
  });
  textBox(`${name}-title`, title, x + 17, y + 9, w - 26, 27, {
    fontSize: titleSize, bold: true, color: "#122230",
  });
  textBox(`${name}-body`, body, x + 17, y + 37, w - 27, h - 43, {
    fontSize: bodySize, color: "#415466", verticalAlignment: "top",
  });
  return s;
}

function connect(a, b, fromSide = "right", toSide = "left", color = "#607386", dashed = false, kind = "straight") {
  const connector = slide.shapes.connect(a, b, {
    kind,
    fromSide,
    toSide,
    line: { style: dashed ? "dashed" : "solid", fill: color, width: 2 },
    tail: { type: "triangle", width: "sm", length: "sm" },
  });
  connector.bringToFront();
  return connector;
}

// Title and framing labels.
textBox("title", "Four local GINE actors choose topology actions; one global critic guides MAPPO", 48, 27, 1184, 51, {
  fontSize: 30, bold: true, color: "#111820",
});
textBox("subtitle", "Current WCCI-36 configuration · decentralized execution · centralized training", 50, 80, 930, 26, {
  fontSize: 15, color: "#5A6875",
});

const actorBand = slide.shapes.add({
  geometry: "roundRect",
  name: "actor-band",
  position: { left: 32, top: 120, width: 1216, height: 315 },
  fill: "#F7FAFD",
  line: { style: "solid", fill: "#D7E3EC", width: 1 },
  borderRadius: 14,
});
actorBand.sendToBack();
textBox("actor-label", "DECENTRALIZED ACTOR  πᵢ", 48, 128, 260, 25, {
  fontSize: 13, bold: true, color: "#2B6F9C",
});
textBox("actor-repeat", "× 4 agents  ·  same architecture  ·  weights not shared", 844, 127, 370, 24, {
  fontSize: 13, bold: true, color: "#2B6F9C", alignment: "right",
});

// Create actor nodes before the connectors are created; connectors are automatically sent behind shapes.
const graph = block(
  "local-graph",
  "Local busbar graph  Gᵢ",
  "Nodes: local busbars\nEdges: feasible line connections\nTopology: active-edge mask",
  50, 185, 170, 148,
  "#EAF3FA", "#78A5C2", "#2F7CA8", 17, 13
);

const nodes = block(
  "node-tensor",
  "Node tensor",
  "6 physical features\n+ 8-D substation ID\n+ 8-D busbar ID\n\n22 features / node",
  257, 165, 164, 123,
  "#F0F6FB", "#8FB3CA", "#4384AA", 16, 12
);
const edges = block(
  "edge-tensor",
  "Edge tensor",
  "6 line features\nstatus · ρ · overload\ncooldown · maintenance",
  257, 303, 164, 102,
  "#F0F6FB", "#8FB3CA", "#4384AA", 16, 12
);

const gine = block(
  "gine-encoder",
  "GINE graph encoder",
  "Node MLP: 22 → 64\nReLU · LayerNorm\n\n1 edge-aware message-passing layer\nhidden dimension: 64",
  462, 181, 190, 169,
  "#E3F0F8", "#5E99BC", "#1E6F9E", 18, 13
);

const pool = block(
  "pool-readout",
  "Graph readout",
  "Mean pooling\nover local nodes\n\nLinear 64 → 64 · ReLU",
  693, 195, 157, 140,
  "#EAF3FA", "#78A5C2", "#2F7CA8", 17, 13
);

const actor = block(
  "actor-mlp",
  "Actor MLP",
  "64 → 128 → 128\nReLU activations",
  891, 211, 139, 108,
  "#E6EEF9", "#788FB9", "#4A69A3", 18, 13
);

const policy = block(
  "policy-head",
  "Categorical policy",
  "128 → Kᵢ action logits\n\ntrain: sample aᵢ\neval: argmax aᵢ",
  1069, 190, 158, 150,
  "#E8EAF7", "#8888B5", "#6262A5", 17, 13
);

connect(graph, nodes, "right", "left", "#6C8799", false, "elbow");
connect(graph, edges, "right", "left", "#6C8799", false, "elbow");
connect(nodes, gine, "right", "left", "#4B7792");
connect(edges, gine, "right", "left", "#4B7792");
connect(gine, pool, "right", "left", "#3C718F");
connect(pool, actor, "right", "left", "#3C718F");
connect(actor, policy, "right", "left", "#536A98");

textBox("graph-dim-note", "nᵢ = 2 × controlled substations", 48, 349, 176, 38, {
  fontSize: 11, color: "#657684", alignment: "center",
});

// Critic / learning band.
const trainBand = slide.shapes.add({
  geometry: "roundRect",
  name: "training-band",
  position: { left: 32, top: 458, width: 1216, height: 220 },
  fill: "#FCFAF7",
  line: { style: "solid", fill: "#E8DDD0", width: 1 },
  borderRadius: 14,
});
trainBand.sendToBack();
textBox("critic-label", "CENTRALIZED TRAINING", 48, 466, 260, 25, {
  fontSize: 13, bold: true, color: "#A0642A",
});

const jointObs = block(
  "joint-observation",
  "Joint flat observation",
  "[o₀ ∥ o₁ ∥ o₂ ∥ o₃]\n77 + 150 + 139 + 377\n= 743 features",
  52, 520, 185, 116,
  "#FFF4E7", "#D2A268", "#C07A2E", 16, 13
);
const critic = block(
  "critic-mlp",
  "Centralized critic",
  "MLP 743 → 128 → 128 → 1\nReLU activations",
  286, 528, 203, 100,
  "#FFF0DF", "#CE9553", "#B66D23", 17, 13
);
const value = block(
  "state-value",
  "State value  V(s)",
  "Single scalar estimate",
  538, 535, 159, 85,
  "#FFF4E9", "#D3A36B", "#BD762C", 17, 13
);
const mappo = block(
  "mappo-update",
  "MAPPO update",
  "GAE advantages + returns\nClipped policy loss\nEntropy bonus + value loss",
  760, 516, 207, 124,
  "#F2EAF8", "#A68ABD", "#80579E", 17, 13
);
const environment = block(
  "grid-environment",
  "WCCI-36 environment",
  "Joint action (a₀…a₃)\nAC power flow\nreward · next state",
  1035, 510, 188, 135,
  "#EAF5EC", "#79A884", "#3E8250", 17, 13
);

connect(jointObs, critic, "right", "left", "#A87843");
connect(critic, value, "right", "left", "#A87843");
connect(value, mappo, "right", "left", "#987348");
connect(mappo, critic, "bottom", "bottom", "#895E9E", true, "elbow");
connect(policy, environment, "bottom", "top", "#546A92", false, "elbow");
connect(environment, mappo, "left", "right", "#638A6C");
connect(mappo, actor, "top", "bottom", "#80579E", true, "elbow");

textBox("actor-update-label", "policy gradients", 844, 440, 112, 22, {
  fontSize: 11, bold: true, color: "#80579E", alignment: "center",
});
textBox("critic-update-label", "value gradients", 300, 645, 118, 20, {
  fontSize: 11, bold: true, color: "#80579E", alignment: "center",
});

// ---------------------------------------------------------------------------
// Slide 2 — topology actions and chronic split
// ---------------------------------------------------------------------------
slide = pres.slides.add();
slide.background.fill = "#FFFFFF";

textBox(
  "slide2-title",
  "A fixed chronic split separates policy learning from model selection",
  48, 27, 1160, 48,
  { fontSize: 30, bold: true, color: "#111820" }
);
textBox(
  "slide2-subtitle",
  "Agents act every five simulated minutes; held-out chronics never enter MAPPO training",
  50, 79, 1020, 26,
  { fontSize: 15, color: "#5A6875" }
);

const actionBand2 = slide.shapes.add({
  geometry: "roundRect",
  name: "action-space-band",
  position: { left: 32, top: 120, width: 362, height: 548 },
  fill: "#F7FAFD",
  line: { style: "solid", fill: "#D7E3EC", width: 1 },
  borderRadius: 14,
});
actionBand2.sendToBack();

const splitBand2 = slide.shapes.add({
  geometry: "roundRect",
  name: "chronic-split-band",
  position: { left: 414, top: 120, width: 834, height: 548 },
  fill: "#FCFCFB",
  line: { style: "solid", fill: "#E3E5E7", width: 1 },
  borderRadius: 14,
});
splitBand2.sendToBack();

textBox("action-space-label", "TOPOLOGY ACTION SPACE", 52, 133, 270, 24, {
  fontSize: 13, bold: true, color: "#2B6F9C",
});
textBox("action-space-heading", "Each agent selects one local action", 52, 166, 306, 58, {
  fontSize: 23, bold: true, color: "#172735", verticalAlignment: "top",
});
textBox("action-space-caption", "The four agents’ selections are combined into one joint grid-control decision.", 52, 221, 302, 48, {
  fontSize: 14, color: "#607080", verticalAlignment: "top",
});

function actionRow(name, symbol, title, body, y) {
  const marker = slide.shapes.add({
    geometry: "ellipse",
    name: `${name}-marker`,
    position: { left: 55, top: y + 7, width: 46, height: 46 },
    fill: "#E7F1F8",
    line: { style: "solid", fill: "#6F9EBB", width: 1.5 },
  });
  marker.text = symbol;
  marker.text.style = {
    fontSize: symbol.length > 1 ? 13 : 19,
    bold: true,
    color: "#276F9A",
    alignment: "center",
    verticalAlignment: "middle",
    fontFamily: "Aptos",
  };
  textBox(`${name}-title`, title, 119, y, 228, 29, {
    fontSize: 17, bold: true, color: "#1B2C39",
  });
  textBox(`${name}-body`, body, 119, y + 31, 228, 37, {
    fontSize: 13, color: "#607080", verticalAlignment: "top",
  });
  return marker;
}

actionRow("noop-action", "0", "Do nothing", "Preserve the current topology", 290);
actionRow("line-action", "↔", "Switch a transmission line", "Connect or disconnect a line", 382);
actionRow("busbar-action", "1|2", "Reassign substation elements", "Move components between busbars", 474);

const cadence = slide.shapes.add({
  geometry: "roundRect",
  name: "decision-cadence",
  position: { left: 52, top: 580, width: 306, height: 62 },
  fill: "#EAF3FA",
  line: { style: "solid", fill: "#91B4CA", width: 1 },
  borderRadius: 10,
});
cadence.text = "Every 5 minutes  ·  shared grid-level reward";
cadence.text.style = {
  fontSize: 15,
  bold: true,
  color: "#285F7F",
  alignment: "center",
  verticalAlignment: "middle",
  fontFamily: "Aptos",
};

textBox("chronic-split-label", "CHRONIC SPLIT & EVALUATION", 435, 133, 310, 24, {
  fontSize: 13, bold: true, color: "#4F5F6A",
});

function flowNode(name, title, body, x, y, w, h, fill, stroke, titleColor, bodyColor = "#566573") {
  const node = slide.shapes.add({
    geometry: "roundRect",
    name,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: stroke, width: 1.5 },
    borderRadius: 11,
    shadow: "shadow-sm",
  });
  textBox(`${name}-title`, title, x + 15, y + 8, w - 30, body ? 27 : h - 16, {
    fontSize: 17,
    bold: true,
    color: titleColor,
    alignment: "center",
  });
  if (body) {
    textBox(`${name}-body`, body, x + 14, y + 36, w - 28, h - 43, {
      fontSize: 12,
      color: bodyColor,
      alignment: "center",
      verticalAlignment: "top",
    });
  }
  return node;
}

const chronics = flowNode(
  "available-chronics",
  "Available chronics",
  "fixed shuffle · split seed 0",
  675, 158, 310, 72,
  "#F4F6F7", "#9CA8AF", "#263641"
);

textBox("train-column-label", "LEARNING", 487, 251, 150, 22, {
  fontSize: 12, bold: true, color: "#BA6B2E", alignment: "center",
});
textBox("test-column-label", "MODEL SELECTION", 992, 251, 180, 22, {
  fontSize: 12, bold: true, color: "#16837F", alignment: "center",
});

const trainSplit = flowNode(
  "train-split",
  "80% training chronics",
  "Only these chronics generate policy updates",
  456, 278, 270, 76,
  "#FFF2E8", "#E9A16D", "#A75A24"
);
const trainLearn = flowNode(
  "mappo-learning",
  "MAPPO rollouts & updates",
  "72 parallel environments in the current WCCI runs",
  456, 386, 270, 82,
  "#FFF5ED", "#E9A16D", "#A75A24"
);
const trainEval = flowNode(
  "train-diagnostic",
  "Train-split evaluation",
  "Diagnostic only · does not choose the checkpoint",
  456, 506, 270, 82,
  "#FFF9F4", "#EAB28A", "#A75A24"
);

const testSplit = flowNode(
  "test-split",
  "20% held-out test chronics",
  "Never used for gradient updates",
  888, 278, 316, 76,
  "#EAF8F7", "#49A8A4", "#146E6B"
);
const periodicTest = flowNode(
  "periodic-test-eval",
  "Deterministic test evaluation",
  "Greedy actions on a held-out subset during training",
  888, 376, 316, 76,
  "#EFF9F8", "#49A8A4", "#146E6B"
);
const checkpoint = flowNode(
  "checkpoint-selection",
  "Best checkpoint",
  "Selected by mean held-out test survival",
  888, 474, 316, 76,
  "#EAF8F7", "#49A8A4", "#146E6B"
);
const finalEval = flowNode(
  "final-test-eval",
  "Final full-split evaluation",
  "Every chronic in the complete held-out test split",
  888, 572, 316, 76,
  "#E4F5F3", "#319894", "#105F5C"
);

connect(chronics, trainSplit, "bottom", "top", "#D6874D", false, "elbow");
connect(chronics, testSplit, "bottom", "top", "#288F8B", false, "elbow");
connect(trainSplit, trainLearn, "bottom", "top", "#D6874D");
connect(trainLearn, trainEval, "bottom", "top", "#D6874D", true);
connect(testSplit, periodicTest, "bottom", "top", "#288F8B");
connect(periodicTest, checkpoint, "bottom", "top", "#288F8B");
connect(checkpoint, finalEval, "bottom", "top", "#288F8B");

// ---------------------------------------------------------------------------
// Slide 3 — IEEE-14 reference MAPPO hyperparameters
// ---------------------------------------------------------------------------
slide = pres.slides.add();
slide.background.fill = "#FFFFFF";

textBox(
  "slide3-title",
  "The IEEE-14 reference baseline trains for 25M steps with 80 PPO epochs",
  48, 27, 1160, 48,
  { fontSize: 30, bold: true, color: "#111820" }
);
textBox(
  "slide3-subtitle",
  "Original MAPPO hyperparameters reproduced before architectural changes",
  50, 79, 1000, 26,
  { fontSize: 15, color: "#5A6875" }
);

const leftTableFrame = slide.shapes.add({
  geometry: "roundRect",
  name: "training-network-table-frame",
  position: { left: 40, top: 126, width: 554, height: 518 },
  fill: "#FBFCFD",
  line: { style: "solid", fill: "#DDE4E8", width: 1 },
  borderRadius: 14,
});
leftTableFrame.sendToBack();

const rightTableFrame = slide.shapes.add({
  geometry: "roundRect",
  name: "ppo-table-frame",
  position: { left: 612, top: 126, width: 628, height: 518 },
  fill: "#FBFCFD",
  line: { style: "solid", fill: "#DDE4E8", width: 1 },
  borderRadius: 14,
});
rightTableFrame.sendToBack();

const leftValues = [
  ["Hyperparameter", "Value"],
  ["TRAINING", ""],
  ["Total timesteps", "25M"],
  ["Rollout steps per environment", "2,000"],
  ["Total rollout batch", "20,000"],
  ["Evaluation frequency", "80,000"],
  ["NETWORK", ""],
  ["Actor architecture", "MLP · 256 × 3"],
  ["Critic architecture", "MLP · 256 × 3"],
  ["Activation", "ReLU"],
];

const rightValues = [
  ["Hyperparameter", "Value"],
  ["PPO OPTIMIZATION", ""],
  ["Actor learning rate", "3 × 10⁻⁵"],
  ["Critic learning rate", "3 × 10⁻⁵"],
  ["Learning-rate schedule", "Linear decay"],
  ["Discount factor  γ", "0.99"],
  ["GAE parameter  λ", "0.95"],
  ["Update epochs", "80"],
  ["Minibatches", "4"],
  ["Clip coefficient  ε", "0.20"],
  ["Target KL", "0.02"],
  ["Maximum gradient norm", "10.0"],
  ["Entropy coefficient", "0.01"],
  ["Value-loss coefficient", "0.50"],
];

const leftTable = slide.tables.add({
  rows: leftValues.length,
  columns: 2,
  left: 56,
  top: 143,
  width: 522,
  height: 478,
  columnWidths: [335, 187],
  values: leftValues,
});
leftTable.merge({ startRow: 1, endRow: 1, startColumn: 0, endColumn: 1 });
leftTable.merge({ startRow: 6, endRow: 6, startColumn: 0, endColumn: 1 });
leftTable.borders.assign({ style: "solid", fill: "#DDE4E8", width: 0.4 });

const rightTable = slide.tables.add({
  rows: rightValues.length,
  columns: 2,
  left: 628,
  top: 143,
  width: 596,
  height: 478,
  columnWidths: [382, 214],
  values: rightValues,
});
rightTable.merge({ startRow: 1, endRow: 1, startColumn: 0, endColumn: 1 });
rightTable.borders.assign({ style: "solid", fill: "#DDE4E8", width: 0.4 });

function styleTable(table, rowCount) {
  for (let r = 0; r < rowCount; r += 1) {
    for (let c = 0; c < 2; c += 1) {
      const cell = table.getCell(r, c);
      cell.fill = r === 0 ? "#F1F4F6" : "#FFFFFF";
      cell.text.style = {
        fontSize: r === 0 ? 15 : 14,
        bold: r === 0,
        color: r === 0 ? "#263641" : "#415466",
        fontFamily: "Aptos",
      };
    }
  }
}

styleTable(leftTable, leftValues.length);
styleTable(rightTable, rightValues.length);

leftTable.getCell(1, 0).fill = "#FFF2E8";
leftTable.getCell(1, 0).text.style = {
  fontSize: 14, bold: true, color: "#A75A24", fontFamily: "Aptos",
};
leftTable.getCell(6, 0).fill = "#EAF3FA";
leftTable.getCell(6, 0).text.style = {
  fontSize: 14, bold: true, color: "#276F9A", fontFamily: "Aptos",
};
rightTable.getCell(1, 0).fill = "#F2EAF8";
rightTable.getCell(1, 0).text.style = {
  fontSize: 14, bold: true, color: "#80579E", fontFamily: "Aptos",
};

for (const row of [2, 4]) {
  leftTable.getCell(row, 1).text.style = {
    fontSize: 15, bold: true, color: "#A75A24", fontFamily: "Aptos",
  };
}
rightTable.getCell(7, 1).text.style = {
  fontSize: 15, bold: true, color: "#80579E", fontFamily: "Aptos",
};

textBox(
  "slide3-source",
  "Source: original IEEE-14 MAPPO baseline hyperparameters",
  50, 658, 700, 21,
  { fontSize: 11, color: "#77838C" }
);

// Slide 4: observation features and graph attribution.
slide = pres.slides.add();
slide.background.fill = "#FFFFFF";

textBox(
  "slide4-title",
  "The graph encoder maps injections to nodes and line conditions to edges",
  48, 27, 1184, 51,
  { fontSize: 30, bold: true, color: "#111820" }
);
textBox(
  "slide4-subtitle",
  "WCCI-36 adds maintenance information on edges; redispatch and curtailment stay outside the core graph",
  50, 80, 1160, 26,
  { fontSize: 15, color: "#5A6875" }
);

const featureTableFrame = slide.shapes.add({
  geometry: "roundRect",
  name: "feature-table-frame",
  position: { left: 32, top: 122, width: 1216, height: 526 },
  fill: "#FBFCFD",
  line: { style: "solid", fill: "#D7E3EC", width: 1 },
  borderRadius: 14,
});
featureTableFrame.sendToBack();

const featureValues = [
  ["Feature category", "Observation variables", "Graph role"],
  ["Generation", "gen_p, gen_theta", "Node"],
  ["Demand", "load_p, load_theta", "Node"],
  ["Line loading", "rho", "Edge"],
  ["Line status", "line_status", "Edge + active mask"],
  ["Overload duration", "timestep_overflow", "Edge"],
  ["Line cooldown", "time_before_cooldown_line", "Edge"],
  ["Substation cooldown", "time_before_cooldown_sub", "Node"],
  ["Topology", "topo_vect", "Busbar nodes + edge mask"],
  ["Maintenance forecast", "time_next_maintenance, duration_next_maintenance", "Edge · WCCI-36 only"],
  ["Redispatch state", "target_dispatch, actual_dispatch, gen_margin_up/down", "Outside core graph"],
  ["Curtailment state", "gen_p_before_curtail, curtailment, curtailment_limit", "Outside core graph"],
  ["Controlled region", "Agent's assigned substations", "Node · domain_mask"],
];

const featureTable = slide.tables.add({
  rows: featureValues.length,
  columns: 3,
  left: 48,
  top: 138,
  width: 1184,
  height: 494,
  columnWidths: [260, 620, 304],
  values: featureValues,
});
featureTable.borders.assign({ style: "solid", fill: "#DDE4E8", width: 0.4 });

for (let r = 0; r < featureValues.length; r += 1) {
  for (let c = 0; c < 3; c += 1) {
    const cell = featureTable.getCell(r, c);
    cell.fill = r === 0 ? "#F1F4F6" : "#FFFFFF";
    cell.text.style = {
      fontSize: r === 0 ? 14 : 12.5,
      bold: r === 0 || (c === 0 && r > 0),
      color: r === 0 ? "#263641" : (c === 0 ? "#263641" : "#415466"),
      fontFamily: "Aptos",
      verticalAlignment: "middle",
    };
  }
}

const nodeRows = [1, 2, 7, 12];
const edgeRows = [3, 5, 6, 9];
const topologyRows = [4, 8];
const outsideRows = [10, 11];

for (const row of nodeRows) {
  featureTable.getCell(row, 2).fill = "#EAF3FA";
  featureTable.getCell(row, 2).text.style = {
    fontSize: 12.5, bold: true, color: "#276F9A", fontFamily: "Aptos",
  };
}
for (const row of edgeRows) {
  featureTable.getCell(row, 2).fill = "#EAF8F7";
  featureTable.getCell(row, 2).text.style = {
    fontSize: 12.5, bold: true, color: "#146E6B", fontFamily: "Aptos",
  };
}
for (const row of topologyRows) {
  featureTable.getCell(row, 2).fill = "#F2EAF8";
  featureTable.getCell(row, 2).text.style = {
    fontSize: 12.5, bold: true, color: "#80579E", fontFamily: "Aptos",
  };
}
for (const row of outsideRows) {
  featureTable.getCell(row, 2).fill = "#FFF2E8";
  featureTable.getCell(row, 2).text.style = {
    fontSize: 12.5, bold: true, color: "#A75A24", fontFamily: "Aptos",
  };
}

textBox(
  "slide4-source",
  "Source: observation schema implemented in GridGraphBuilder",
  50, 658, 700, 21,
  { fontSize: 11, color: "#77838C" }
);

// Slide 5: concise literature review.
slide = pres.slides.add();
slide.background.fill = "#FFFFFF";

textBox(
  "slide5-title",
  "Topology-control research is moving toward coordinated regional agents",
  48, 27, 1184, 51,
  { fontSize: 30, bold: true, color: "#111820" }
);
textBox(
  "slide5-subtitle",
  "Representative Grid2Op / L2RPN studies · references [1]–[6]",
  50, 80, 900, 26,
  { fontSize: 15, color: "#5A6875" }
);

// Column backgrounds establish the comparison without turning the slide into a dense table.
const singleBand = slide.shapes.add({
  geometry: "roundRect",
  name: "single-agent-band",
  position: { left: 38, top: 126, width: 577, height: 456 },
  fill: "#F6FAFD",
  line: { style: "solid", fill: "#D7E3EC", width: 1 },
  borderRadius: 14,
});
singleBand.sendToBack();
const multiBand = slide.shapes.add({
  geometry: "roundRect",
  name: "multi-agent-band",
  position: { left: 665, top: 126, width: 577, height: 456 },
  fill: "#F5FBFA",
  line: { style: "solid", fill: "#D1E7E4", width: 1 },
  borderRadius: 14,
});
multiBand.sendToBack();

textBox("single-agent-header", "SINGLE AGENT", 62, 143, 220, 31, {
  fontSize: 20, bold: true, color: "#276F9A",
});
textBox("single-agent-summary", "One policy sees the grid and chooses from a global action set", 62, 174, 515, 28, {
  fontSize: 15, color: "#607386",
});
textBox("multi-agent-header", "MULTI-AGENT", 689, 143, 220, 31, {
  fontSize: 20, bold: true, color: "#146E6B",
});
textBox("multi-agent-summary", "Control is divided by substation or region, then coordinated", 689, 174, 515, 28, {
  fontSize: 15, color: "#607386",
});

function paperEntry(name, refNo, year, title, contribution, x, y, color, tint) {
  const marker = slide.shapes.add({
    geometry: "roundRect",
    name: `${name}-marker`,
    position: { left: x, top: y + 5, width: 52, height: 29 },
    fill: tint,
    line: { style: "solid", fill: color, width: 1.3 },
    borderRadius: 8,
  });
  textBox(`${name}-year`, year, x, y + 6, 52, 25, {
    fontSize: 12.5, bold: true, color, alignment: "center",
  });
  textBox(`${name}-title`, `${title}  [${refNo}]`, x + 68, y, 450, 30, {
    fontSize: 16, bold: true, color: "#233442",
  });
  textBox(`${name}-contribution`, contribution, x + 68, y + 31, 448, 50, {
    fontSize: 14, color: "#526574", verticalAlignment: "top",
  });
}

paperEntry(
  "subramanian", 1, "2021", "Simple deep RL baseline",
  "IEEE-14 topology switching; strong transfer from one training scenario.",
  62, 221, "#276F9A", "#EAF3FA"
);
paperEntry(
  "yoon", 2, "2021", "Hierarchical SMAAC",
  "Graph attention + afterstate evaluation reduce the global decision burden.",
  62, 329, "#276F9A", "#EAF3FA"
);
paperEntry(
  "zhou", 3, "2021", "Policy-guided safe search",
  "The policy proposes candidates; Grid2Op simulation filters unsafe actions.",
  62, 437, "#276F9A", "#EAF3FA"
);

paperEntry(
  "vandersar", 4, "2023", "Substation-level MARL",
  "Independent / dependent PPO and SACD agents are tested in a hierarchy.",
  689, 221, "#146E6B", "#EAF8F7"
);
paperEntry(
  "demol", 5, "2025", "Centrally coordinated MARL",
  "Regional agents propose actions; a learned or greedy coordinator selects one.",
  689, 329, "#146E6B", "#EAF8F7"
);
paperEntry(
  "marl2grid", 6, "2026", "MARL2Grid-TR benchmark",
  "Configurable local observability and agents for topology and redispatch control.",
  689, 437, "#146E6B", "#EAF8F7"
);

const takeawayBand = slide.shapes.add({
  geometry: "roundRect",
  name: "literature-takeaway-band",
  position: { left: 38, top: 603, width: 1204, height: 70 },
  fill: "#FFF7EE",
  line: { style: "solid", fill: "#F0D5B7", width: 1 },
  borderRadius: 12,
});
takeawayBand.sendToBack();
textBox("literature-takeaway-label", "OPEN CHALLENGE", 62, 616, 174, 23, {
  fontSize: 13, bold: true, color: "#A75A24",
});
textBox(
  "literature-takeaway",
  "Stable coordination on large grids under partial observability and safety constraints remains unresolved.",
  238, 611, 965, 38,
  { fontSize: 18, bold: true, color: "#49392C" }
);

// Slide 6: references.
slide = pres.slides.add();
slide.background.fill = "#FFFFFF";

textBox("slide6-title", "References", 48, 27, 1184, 51, {
  fontSize: 30, bold: true, color: "#111820",
});
textBox(
  "slide6-subtitle",
  "Representative Grid2Op / L2RPN topology-control literature cited in the review",
  50, 80, 1080, 26,
  { fontSize: 15, color: "#5A6875" }
);

const refDivider = slide.shapes.add({
  geometry: "rect",
  name: "reference-divider",
  position: { left: 636, top: 138, width: 1, height: 494 },
  fill: "#DCE4EA",
  line: { style: "solid", fill: "#DCE4EA", width: 0 },
});

function referenceEntry(name, number, citation, link, x, y, accent) {
  const num = slide.shapes.add({
    geometry: "ellipse",
    name: `${name}-number`,
    position: { left: x, top: y, width: 35, height: 35 },
    fill: accent,
    line: { style: "solid", fill: accent, width: 0 },
  });
  textBox(`${name}-number-text`, String(number), x, y + 1, 35, 31, {
    fontSize: 14, bold: true, color: "#FFFFFF", alignment: "center",
  });
  textBox(`${name}-citation`, citation, x + 50, y - 3, 500, 91, {
    fontSize: 14.5, color: "#2D3F4D", verticalAlignment: "top",
  });
  textBox(`${name}-link`, link, x + 50, y + 88, 500, 25, {
    fontSize: 11.5, color: accent, verticalAlignment: "top",
  });
}

referenceEntry(
  "ref1", 1,
  "Subramanian, M., Viebahn, J., Tindemans, S. H., Donnot, B., & Marot, A. (2021). Exploring grid topology reconfiguration using a simple deep reinforcement learning approach. IEEE Madrid PowerTech.",
  "doi.org/10.1109/PowerTech46648.2021.9494879",
  58, 149, "#2F7CA8"
);
referenceEntry(
  "ref2", 2,
  "Yoon, D., Hong, S., Lee, B.-J., & Kim, K.-E. (2021). Winning the L2RPN Challenge: Power grid management via semi-Markov afterstate actor-critic. ICLR 2021.",
  "openreview.net/forum?id=LmUJqB1Cz8",
  58, 311, "#2F7CA8"
);
referenceEntry(
  "ref3", 3,
  "Zhou, B., Zeng, H., Liu, Y., Li, K., Wang, F., & Tian, H. (2021). Action set based policy optimization for safe power grid management. ECML PKDD 2021.",
  "arxiv.org/abs/2106.15200",
  58, 473, "#2F7CA8"
);

referenceEntry(
  "ref4", 4,
  "van der Sar, E., Zocca, A., & Bhulai, S. (2023). Multi-agent reinforcement learning for power grid topology optimization. arXiv:2310.02605.",
  "arxiv.org/abs/2310.02605",
  671, 149, "#16817C"
);
referenceEntry(
  "ref5", 5,
  "de Mol, B., Barbieri, D., Viebahn, J., & Grossi, D. (2025). Centrally coordinated multi-agent reinforcement learning for power grid topology control. arXiv:2502.08681.",
  "arxiv.org/abs/2502.08681",
  671, 311, "#16817C"
);
referenceEntry(
  "ref6", 6,
  "Marchesini, E., Boguslawski, E., Leite, A., Amato, C., Dussartre, M., Schoenauer, M., Donnot, B., & Donti, P. L. (2026). MARL2Grid-TR: A multi-agent RL benchmark in power grid operations. ICLR 2026.",
  "openreview.net/forum?id=mpAMH1OyMO",
  671, 473, "#16817C"
);

await fs.mkdir(OUT, { recursive: true });
for (const [index, currentSlide] of pres.slides.items.entries()) {
  const slideNo = index + 1;
  const png = await pres.export({ slide: currentSlide, format: "png", scale: 2 });
  await writeBlob(`${OUT}/wcci_gine_presentation-slide-${slideNo}.png`, png);
  const layout = await currentSlide.export({ format: "layout" });
  await fs.writeFile(`${OUT}/wcci_gine_presentation-slide-${slideNo}.layout.json`, await layout.text());
}
const montage = await pres.export({ format: "webp", montage: true, scale: 1 });
await writeBlob(`${OUT}/wcci_gine_presentation-montage.webp`, montage);
const pptx = await PresentationFile.exportPptx(pres);
await pptx.save(`${OUT}/wcci_gine_presentation_with_literature_review.pptx`);
