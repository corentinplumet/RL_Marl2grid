# Final Week To-Do List

This checklist turns the full-report review into a one-week finishing plan. The objective is not to add another broad experimental campaign. The objective is to make the thesis internally coherent, methodologically defensible, and explicit about what the experiments do and do not establish.

The current scientific story is already sound:

> Stabilize decentralized MAPPO on `bus14`, study sparse intervention, replace fixed-width actors with graph and candidate-action models, diagnose cross-grid input mismatch, and evaluate zero-shot transfer and target adaptation on WCCI.

The final week should preserve that story while fixing the protocol language, completing the missing conclusion, integrating results that already exist, and running only the smallest high-value confirmation experiments if time permits.

---

## Definition of done

The thesis is ready to submit when all of the following are true:

- [X] The document no longer presents itself as a “Thesis Plan.”
- [X] An abstract summarizes the problem, methods, main numerical findings, limitations, and conclusion.
- [X] The research questions match the actual contents of Chapters 3–9.
- [X] “Survival” has one unambiguous definition throughout the report.
- [X] Checkpoint selection and the role of validation/test data are described consistently.
- [ ] Single-seed results are described as observations from a screen, not universal optima.
- [X] The two different pooling stages are clearly distinguished.
- [ ] The existing `k=32` fine-tuning results are included in the transfer story.
- [ ] A final Discussion and Conclusions chapter directly answers every research question.
- [ ] The compiled PDF has no clipped titles, broken references, placeholders, or obvious layout problems.
- [ ] Every principal number in the conclusion can be traced to a table or figure in the main text or appendix.

---

# Priority system

## P0 — Must finish before submission

These tasks affect the validity or basic completeness of the thesis. Do these before running any new experiment.

1. Finalize the thesis framing, title page, abstract, and research questions.
2. Correct the survival metric definition.
3. Clarify the checkpoint-selection and test-set limitation.
4. Moderate claims based on one seed.
5. Integrate the already available `k=32` fine-tuning evaluations.
6. Add the final Discussion and Conclusions chapter.
7. Recompile and visually inspect the complete PDF.

## P1 — Strongly recommended

These tasks substantially improve the argument without requiring a large training campaign.

1. Evaluate only the finalist policies on a larger untouched WCCI cohort, ideally all 576 maintenance-disabled test chronics.
2. Add paired chronic-level uncertainty for the main WCCI comparisons.
3. Shorten or split the oversized Methods chapter.
4. Make the chapter-summary boxes and terminology consistent.

## P2 — Only if time remains

1. Repeat only the final models with two additional training seeds.
2. Run a matched-budget scratch-versus-fine-tuning comparison.
3. Add latency and parameter-cost measurements.
4. Add maintenance-enabled robustness as explicitly out-of-domain evaluation.

Do not start another large architecture factorial during the final week.

---

# Day 1 — Freeze the thesis claim and repair the front matter

## 1. Replace the “Thesis Plan” framing

Files to inspect:

- `latex/main.tex`
- `latex/chapters/01_plan.tex`

Tasks:

- [X] Replace `Thesis Plan` in the visible title and PDF metadata.
- [X] Replace “Working Title” with the final title.
- [X] Rename Chapter 1 to `Research Questions and Contributions`, or merge it into the Introduction.
- [X] Rename “Expected Contributions” to “Contributions.”
- [X] Correct `environemnt` and `chainging`.
- [X] Update the document date according to the submission rules rather than relying on `\today` if a fixed date is required.

Suggested central thesis question:

> To what extent can graph-based decentralized policies trained on a small grid retain useful control performance on a larger grid, and which representation and action-selection choices determine transfer?

This formulation does not assume in advance that zero-shot transfer succeeds.

Completion criterion:

- [X] The title page and first chapter read like a finished thesis, not a proposal.

## 2. Align the research questions with the actual thesis

Recommended research questions:

1. **Baseline learning:** Which training changes are needed for stable decentralized MAPPO topology control on `bus14`?
2. **Sparse intervention:** Can the agents reduce unnecessary topology interventions without sacrificing source-grid survival?
3. **Graph representation:** Which graph representation, message-passing, readout, and candidate-action design choices produce a useful local actor?
4. **Cross-grid transfer:** Can the same graph actor weights execute and retain useful performance when transferred from `bus14` to WCCI?
5. **Target adaptation:** Does source initialization improve WCCI learning compared with zero-shot deployment and training from scratch?

Tasks:

- [X] Restore sparse intervention as an explicit research question.
- [X] Add target adaptation as either its own question or a clearly stated sub-question of transfer.
- [X] Map every question to the chapters that answer it.
- [X] Make sure every claimed contribution corresponds to actual evidence.

Completion criterion:

- [X] No major results chapter is disconnected from the declared research questions.

## 3. Add an abstract

Target length: approximately 200–300 words unless the institution specifies otherwise.

The abstract should contain:

- [X] Problem: decentralized topology control and the scaling limitation of fixed-width actors.
- [X] Method: MAPPO, shared graph encoders, candidate-action scoring, reduced WCCI action spaces, and target adaptation.
- [X] Source result: stable/high-survival `bus14` controller.
- [X] Transfer result: structural weight compatibility does not automatically preserve control quality.
- [X] Main WCCI findings: high source-grid performance, weak WCCI retraining from scratch, and meaningful zero-shot transfer with a transferable graph actor.
- [X] Limitation: one training seed for the main architecture screens and a 50-chronic transfer cohort.
- [X] Conclusion: GNNs remove the dimensional transfer barrier, but reliable behavioral transfer still requires action-space control and target adaptation.

Completion criterion:

- [X] A reader can understand the full thesis contribution from the abstract alone.

---

# Day 2 — Fix definitions and methodological consistency

## 4. Give survival one definition everywhere

Current problem:

- The Introduction describes episodic survival as the fraction of episodes completed without blackout.
- The experiments primarily report normalized episode duration averaged over chronics.
- Completed episodes are also reported separately.

Use this definition consistently:

> For one chronic, survival is the number of survived timesteps divided by the maximum episode length. Mean survival is this normalized duration averaged over evaluated chronics. A completed episode is a separate binary outcome with 100% survival.

Tasks:

- [X] Correct the definition in `latex/chapters/02_introduction.tex`.
- [X] Check Chapters 3, 4, 6, 8, and 9 for the same terminology.
- [X] Reserve “completion rate” for the fraction of episodes reaching 100%.
- [X] Reserve “rescues” for difficult chronics completed by the model.
- [X] Prefer “mean survival” over ambiguous uses of “survival rate.”

Completion criterion:

- [X] A value such as 93% cannot be misread as “93% of episodes were completed.”

## 5. Clarify checkpoint selection and data splits

Verified protocol:

- Policy updates use the 80% training split.
- Checkpoint selection uses mean survival over a 20-chronic rollout from the
  training split.
- A new rollout is used at every evaluation, so the 20 training chronics change
  between evaluations rather than forming a fixed subset.
- The selected checkpoint is then evaluated over the held-out test split.

Remaining limitation:

- The held-out `bus14` results compare architecture cells and guide later
  screens. They are therefore comparative screening results, not an untouched
  final confirmation of the architecture selected across screens.

Tasks:

- [X] State the changing 20-training-chronic rollout selection rule in Chapter 3.
- [X] State the same rule in the shared protocol of Chapter 6.
- [X] Remove claims that the highest test score selects the checkpoint.
- [X] Distinguish train-only checkpoint selection from architecture screening
  on the held-out split.
- [X] Use final checkpoints for the primary WCCI-trained comparisons, as
  Chapter 9 already does.

Completion criterion:

- [X] The report states that checkpoint selection uses only training-split
  chronics and explains the separate role of the held-out architecture screens.

## 6. Separate the different meanings of pooling

Add a short nomenclature paragraph or diagram near the end of Chapter 5 or the beginning of Chapter 6.

The architecture contains three distinct aggregation operations:

1. **Message aggregation:** neighbors are aggregated during each GNN layer.
2. **Graph-level readout:** all/controlled/energized nodes are reduced to a graph embedding using mean or maximum. Screen B studies this operation.
3. **Candidate-action pooling:** nodes touched by one candidate action are reduced using joint mean, typed mean, or another candidate context. Screen E and Chapter 9 study this operation.

Tasks:

- [X] Explicitly say that `energized_max` and `typed_mean` do not contradict each other because they operate at different stages.
- [X] Use `graph readout` for Screen B.
- [X] Use `candidate pooling` for Screen E and the transfer chapter.
- [X] Check figure captions and table headers for ambiguous uses of “pooling.”

Completion criterion:

- [X] A reader can explain why maximum graph readout and typed-mean candidate pooling can both be selected.

---

# Day 3 — Calibrate claims and improve chapter transitions

## 7. Replace overstrong single-seed language

Files to prioritize:

- `latex/chapters/03_baseline.tex`
- `latex/chapters/06_experiments.tex`
- `latex/chapters/09_screen_e_transfer.tex`

Replace expressions such as:

- `strictly dominates` → `ranked first in this seed-0 screen`;
- `optimal` → `best observed configuration` or `selected configuration`;
- `ideal balance` → `best observed trade-off under the reported diagnostics`;
- `significantly more stable` → `had lower late-training variability in this run`.

Tasks:

- [ ] Make clear that Screens A–E are descriptive single-seed factorials.
- [ ] Keep matched effects as the main evidence.
- [ ] Label physical explanations of message direction as hypotheses or interpretations.
- [ ] Do not use statistical language unless an actual statistical test supports it.
- [ ] In Chapter 9, retain the existing warning that architecture-cell bootstrap intervals are not seed uncertainty.

Completion criterion:

- [ ] Every conclusion is proportional to the number of seeds and independent observations supporting it.

## 8. Repair the Chapter 3 baseline argument

Current issue:

- The text says the optimized MLP consistently beats the ablations.
- The smaller MLP has a slightly higher final survival.
- The 70% initial do-nothing bias has the highest final endpoint.

Tasks:

- [ ] State the actual baseline-selection criterion: convergence speed, curve mean, stability, or repeatability—not simply peak/final survival.
- [ ] Explain why the 70% bias is not selected despite its final endpoint.
- [ ] If the criterion cannot be quantified, soften the claim and describe the optimized MLP as a stable reference rather than the unique best model.
- [ ] Add a late-window or across-seed statistic if it already exists in the notebook.

Completion criterion:

- [ ] The chosen reference follows from a stated decision rule that matches the table and curves.

## 9. Repair cross-screen comparability in Chapter 6

Current issue:

- Chapter 6 calls the screens a “unified, numerically comparable sequence.”
- An identical baseline obtains 99.39% in Screen B and 95.45% in Screen C, attributed to hardware and nondeterminism.

Tasks:

- [ ] State that comparisons are controlled **within** each factorial.
- [ ] Avoid direct absolute ranking across campaigns executed under different hardware/nondeterministic trajectories.
- [ ] Retain shared anchor cells where they exist, but treat their variation as an empirical estimate of run sensitivity.
- [ ] Rewrite the Chapter 6 summary to report selections rather than universal optima.
- [ ] Verify or remove the statement that most relevant substations are within two hops. If retained, support it with an actual hop-distance measurement.

Completion criterion:

- [ ] The reader is never asked to interpret cross-campaign differences as controlled effects.

## 10. Update the Introduction outline and transitions

Tasks:

- [ ] Add Chapter 7 to the Introduction’s chapter outline.
- [ ] Explain that Chapter 7 diagnoses the input mismatch before WCCI transfer.
- [ ] Describe Chapter 8 as the target-environment and evaluation-protocol chapter.
- [ ] Describe Chapter 9 as zero-shot transfer plus scratch/fine-tuned target adaptation.
- [ ] Keep Chapter 4’s early WCCI experiment explicitly labelled as a preliminary, maintenance-enabled stress test that changes several factors at once.

Completion criterion:

- [ ] The outline accurately predicts the actual contents and order of every chapter.

---

# Day 4 — Integrate the existing `k=32` adaptation evidence

## 11. Add the `k=32` fine-tuning results before running anything new

Relevant local results include the `ft32c` evaluations under:

- `Topology_Task/outputs/full_test_eval/shared/wcci/ft32c/`
- corresponding action summaries under `Topology_Task/outputs/full_test_eval_actions/shared/wcci/`

Current narrative gap:

- Chapter 9 identifies `k=32` as the strongest zero-shot action space.
- The adaptation section compares zero-shot, scratch, and fine-tuned models only at `k=64`.
- Existing `k=32` fine-tuning evaluations are therefore directly relevant and should be incorporated.

Tasks:

- [ ] Update the analysis notebook from the permanent JSON and action-summary files.
- [ ] Compare final checkpoints under no heuristic and local-`rho=0.95` separately.
- [ ] Use the same 50 chronics and the same easy/difficult cohort definitions.
- [ ] Report overall mean survival.
- [ ] Report difficult-cohort mean survival.
- [ ] Report rescues and easy chronics kept.
- [ ] Report the paired difference from the corresponding zero-shot actor.
- [ ] Report the paired difference from the corresponding scratch actor if a matched `k=32` scratch result exists.
- [ ] Keep best-test checkpoint numbers secondary; use final checkpoints in the main comparison.
- [ ] Update the Chapter 9 adaptation figure or add a focused `k=32`/`k=64` comparison.

Questions the new text must answer:

1. Does fine-tuning still improve the best zero-shot action-space setting?
2. Is `k=32` still preferable after target training, or was its advantage specific to zero-shot ranking?
3. Does local gating preserve easy chronics or suppress useful fine-tuned interventions?
4. Is the advantage driven by a few rescued chronics or broadly distributed across the difficult cohort?

Completion criterion:

- [ ] The report no longer selects `k=32` as the zero-shot winner and then silently switches to `k=64` for adaptation.

## 12. Decide how to handle dangerous-state BC/ranking experiments

Recommendation for this version of the thesis:

- [ ] Do not place the recent dangerous-state BC/ranking experiments in the central transfer narrative unless deployed survival becomes competitive.
- [ ] If retained, present them as an exploratory negative result or future direction.
- [ ] Distinguish high supervised ranking accuracy from closed-loop control performance.
- [ ] State that offline imitation on queried dangerous states creates distribution-shift and compounding-error risks during deployment.

Completion criterion:

- [ ] The main story remains focused on evidence that affects closed-loop WCCI survival.

---

# Day 5 — Add the final discussion and conclusion

## 13. Create a final `Discussion and Conclusions` chapter

Recommended structure:

### 13.1 Research questions revisited

Answer each research question in one concise subsection with explicit evidence.

### 13.2 Main contributions

Recommended contribution wording:

1. A stabilized decentralized MAPPO baseline on `bus14`.
2. A comparison of learned and evaluation-time sparse-intervention mechanisms.
3. A systematic graph-design screen covering representation, message passing, readout, and candidate scoring.
4. A graph and candidate-action architecture whose parameter dimensions do not depend on target grid/action-space size.
5. A transfer diagnosis separating feature mismatch, action-space quality, policy ranking, and target adaptation.

### 13.3 What transferred and what did not

Make this distinction explicit:

- **Structural transfer succeeded:** the same shared encoder/scorer can execute with a different grid, number of agents, and reduced action list.
- **Behavioral transfer was limited:** zero-shot performance was sensitive to action-space size, pooling, and gating and remained far below the greedy reference.
- **Target adaptation helped:** fine-tuning generally improved difficult-chronic performance over zero-shot and scratch in the matched experiments.

### 13.4 Practical interpretation

State that:

- Small action lists can be easier for a learned ranker despite containing weaker greedy actions.
- Overall survival alone is insufficient because easy do-nothing chronics dominate it.
- Local heuristics should be validated per policy rather than treated as universal safety layers.
- Graph compatibility is necessary but not sufficient for reliable transfer.

### 13.5 Limitations

Include at least:

- [ ] one training seed in the main graph and transfer screens;
- [ ] checkpoint selection on the held-out `bus14` evaluation split;
- [ ] only 50 chronics in the primary WCCI policy comparison;
- [ ] maintenance disabled for the controlled transfer study;
- [ ] only one source-target grid pair;
- [ ] reduced rather than complete target action spaces;
- [ ] different target-training budgets for scratch and fine-tuning;
- [ ] greedy policy is a reference, not an upper bound;
- [ ] hardware/nondeterministic trajectory differences across campaigns.

### 13.6 Future work

Prioritize:

1. independent validation/test selection;
2. multi-seed finalist replication;
3. matched-budget scratch/fine-tune learning curves;
4. improved action ranking and action-space construction;
5. maintenance-enabled and second-grid transfer;
6. offline dangerous-state learning followed by safe on-policy correction.

Recommended final thesis conclusion:

> Graph and candidate-action architectures remove the dimensional barrier to reusing a decentralized policy across grids, but reliable behavioral transfer is not obtained from graph structure alone. In the reported `bus14`-to-WCCI case study, transfer depends jointly on comparable physical inputs, a manageable candidate set, candidate-aware pooling, policy-specific intervention control, and target-grid adaptation. Fine-tuning the transferable actor is therefore more promising than either direct zero-shot deployment or relearning the same architecture from random initialization.

Completion criterion:

- [ ] The final chapter answers the central question without requiring the reader to reconstruct the answer from several results chapters.

---

# Day 6 — High-value confirmation evaluation, only if compute is available

## 14. Evaluate finalists on a larger untouched WCCI cohort

This is the highest-value additional experiment because it requires evaluation rather than another training sweep.

Preferred cohort:

- all 576 maintenance-disabled WCCI test chronics; or
- a new fixed confirmation subset that was not used to choose models.

Recommended finalist set:

1. do nothing;
2. best final gated zero-shot actor at `k=32`;
3. best final fine-tuned actor at `k=32`;
4. matched final scratch actor at `k=32`;
5. optionally best final fine-tuned actor at `k=64`.

Rules:

- [ ] Freeze the model list before looking at the confirmation results.
- [ ] Use final checkpoints, not WCCI-selected best-test checkpoints.
- [ ] Use identical deterministic evaluation settings.
- [ ] Keep maintenance disabled to match the primary study.
- [ ] Save episode-level summaries and action distributions.
- [ ] Report results by month as a distribution check, not as twelve independent model-selection opportunities.

Metrics:

- [ ] overall mean survival;
- [ ] median survival;
- [ ] completion rate;
- [ ] difficult-cohort mean survival using a do-nothing-defined cohort;
- [ ] rescues;
- [ ] easy chronics retained;
- [ ] mean intervention/action-0 fraction;
- [ ] paired per-chronic difference from do nothing;
- [ ] paired difference from the corresponding zero-shot model.

Completion criterion:

- [ ] The main transfer conclusion is reproduced on data that did not choose the finalist.

## 15. Add chronic-level uncertainty

For each finalist comparison:

- [ ] Compute the paired survival difference on every common chronic.
- [ ] Bootstrap chronics, not architecture cells, to form a confidence interval for the mean paired difference.
- [ ] Report the fraction of chronics improved, unchanged, and worsened.
- [ ] Show a per-chronic paired plot or survival-difference distribution.
- [ ] Keep easy and difficult cohorts separate.

Important interpretation rule:

> A model that has a higher mean because of three complete rescues but is worse on most other difficult chronics should be described differently from a model with a smaller but broadly distributed improvement.

Completion criterion:

- [ ] The thesis shows whether improvement is broad or driven by a few outliers.

---

# Day 7 — Compression, consistency, and final PDF quality assurance

## 16. Shorten or split the Methods chapter

Chapter 5 is approximately 9,100 words and contains three summary boxes. It is effectively several chapters inside one chapter.

Low-risk option for the final week:

- Keep the graph design rationale, mathematical definitions, and key architecture diagrams in the main chapter.
- Move exhaustive feature schemas, row-index construction, padding/masking mechanics, and implementation details into the appendices.

Alternative if renumbering is manageable:

- `Graph Representation and Shared Encoder`
- `Transferable Candidate-Action Scoring`

Tasks:

- [ ] Remove repetition between Chapter 5 and architecture appendices.
- [ ] Keep only details needed to understand the experimental hypotheses.
- [ ] Ensure every moved table is still referenced from the main text.

Completion criterion:

- [ ] Results chapters remain the visual and argumentative centre of the thesis.

## 17. Standardize all chapter summaries

Use this template consistently:

1. **Purpose**
2. **Content**
3. **Findings**
4. **Takeaway**

Tasks:

- [ ] Convert Chapters 3 and 4 from `Objective/Findings/Conclusion` to the shared template.
- [ ] Keep summary claims shorter and more cautious than the detailed chapter discussion.
- [ ] Ensure each takeaway creates a clear transition to the next chapter.

Completion criterion:

- [ ] Every chapter closes in a visually and rhetorically consistent way.

## 18. Terminology pass

Check and standardize:

- [ ] `bus14`, IEEE 14-bus, and source grid;
- [ ] WCCI, WCCI36, and `bus36_wcci_nomaint`;
- [ ] do nothing / do-nothing;
- [ ] local-`rho` gate / heuristic;
- [ ] mean survival / completion rate / rescues;
- [ ] graph readout / candidate pooling / message aggregation;
- [ ] typed-mean spelling in prose and `typed_mean` in configuration labels;
- [ ] physical scaling: NL versus NLS;
- [ ] final checkpoint versus selected/best-test checkpoint;
- [ ] `greedy reference`, never `greedy ceiling` or `upper bound`.

Completion criterion:

- [ ] A single term refers to a single concept throughout the report.

## 19. Numerical consistency pass

Create a small private verification table or script covering all headline values.

Verify at least:

- [ ] full WCCI do-nothing maintenance-on and maintenance-off values;
- [ ] 50-chronic do-nothing overall and difficult-cohort values;
- [ ] number of easy and difficult chronics;
- [ ] greedy survival for `k=32`, `64`, `128`, and `256`;
- [ ] zero-shot medians in Chapter 9;
- [ ] architecture-effect values;
- [ ] heuristic-effect values;
- [ ] scratch/fine-tuning matched means and medians;
- [ ] all rescue and easy-kept counts;
- [ ] training budgets and learning rates.

For every number in a chapter summary or conclusion:

- [ ] identify its source JSON/notebook cell;
- [ ] identify its table or figure in the report;
- [ ] verify that it uses the intended checkpoint policy and heuristic condition.

Completion criterion:

- [ ] No headline number is manually copied without a reproducible source.

## 20. Full visual QA of the compiled PDF

Known issue:

- The Chapter 9 opening title is clipped/misaligned in the current compiled PDF.

Tasks:

- [ ] Recompile from a clean auxiliary-file state if necessary.
- [ ] Check the title page and PDF metadata.
- [ ] Check the abstract.
- [ ] Check all chapter-opening pages.
- [ ] Fix the Chapter 9 title layout.
- [ ] Check every red summary box for overflow or awkward page breaks.
- [ ] Check tables for unreadably small text.
- [ ] Check figures for cut-off legends and inconsistent fonts.
- [ ] Check that captions explain checkpoint type, seed count, heuristic condition, action-space size, and cohort where necessary.
- [ ] Check all internal references and bibliography citations.
- [ ] Confirm there are no `??`, unresolved citations, pending placeholders, TODOs, or stale review colours.
- [ ] Check the table of contents after adding the conclusion.
- [ ] Inspect the final PDF at normal reading zoom and as printed A4 pages.

Completion criterion:

- [ ] The PDF can be read from title page to appendices without a visible unfinished element.

---

# Optional experiments after the thesis-critical work

## 21. Multi-seed finalist replication

Only do this if training capacity and analysis time remain.

Repeat with seeds 1 and 2:

- [ ] selected source architecture;
- [ ] best scratch WCCI model;
- [ ] best fine-tuned WCCI model.

Do not repeat all 16 architecture cells.

Report:

- [ ] seed-level mean and spread;
- [ ] final checkpoints;
- [ ] the same fixed evaluation chronics;
- [ ] easy/difficult results;
- [ ] whether the ordering survives across seeds.

## 22. Matched-budget adaptation

To support a sample-efficiency claim:

- [ ] train scratch and fine-tuned actors for the same number of WCCI steps;
- [ ] use the same architecture, `k`, optimizer schedule, and evaluation frequency;
- [ ] plot difficult-cohort survival against target environment steps;
- [ ] compare area under the learning curve and final performance;
- [ ] report whether pretraining helps early learning, final learning, or both.

## 23. Runtime and deployment cost

Optional but useful for the scalability claim:

- [ ] actor parameter count;
- [ ] forward-pass time per agent;
- [ ] candidate-scoring time as a function of `k`;
- [ ] evaluation wall time per chronic;
- [ ] comparison with fixed-list GNN and flat MLP.

## 24. Maintenance-enabled robustness

Treat this as out-of-domain robustness, not as directly comparable to the maintenance-disabled main study.

- [ ] Evaluate only finalists.
- [ ] Keep the same action spaces and checkpoint policy.
- [ ] Report the maintenance-disabled and maintenance-enabled values separately.
- [ ] Do not merge them into one average.

---

# Things not to do during the final week

- [ ] Do not launch another exhaustive 16-cell architecture sweep.
- [ ] Do not select a new “best” model on the final confirmation cohort.
- [ ] Do not report best-test checkpoints as if they were final independent test estimates.
- [ ] Do not call one-step greedy an upper bound.
- [ ] Do not convert a one-seed ranking into a universal architectural rule.
- [ ] Do not add the dangerous-state BC work to the main narrative solely because supervised accuracy is high.
- [ ] Do not let new experiments delay the abstract, conclusion, protocol corrections, or PDF QA.

---

# Recommended final structure

1. **Introduction and research questions**
2. **Problem setting, decentralized MAPPO, and evaluation metrics**
3. **Establishing a stable `bus14` baseline**
4. **Sparse intervention control**
5. **Graph representation and transferable candidate-action scoring**
6. **Graph-design screening on `bus14`**
7. **Transfer-aware encoder inputs**
8. **WCCI target environment and controls**
9. **Zero-shot transfer and target adaptation**
10. **Discussion and conclusions**
11. **Appendices**

The existing chapter order already follows this logic. The essential structural change is to merge or reframe the current “Thesis Plan,” add an abstract, and add Chapter 10.

---

# Final submission checklist

## Scientific

- [ ] Every research question is answered.
- [ ] Every strong claim is supported by matched evidence.
- [ ] Single-seed limitations are visible wherever relevant.
- [ ] Test/validation/checkpoint selection is accurately described.
- [ ] Easy and difficult WCCI cohorts are both reported.
- [ ] Greedy-relative performance is interpreted as a reference, not an upper bound.
- [ ] Structural transfer and behavioral transfer are distinguished.
- [ ] Final conclusions do not exceed the scope of one source-target grid pair.

## Reproducibility

- [ ] Configurations, seeds, budgets, checkpoints, and action spaces are identifiable.
- [ ] Final versus best-test checkpoints are labelled.
- [ ] Evaluation heuristic and threshold are labelled.
- [ ] Main tables can be regenerated from notebooks or scripts.
- [ ] The final PDF and the analysis inputs correspond to the same result version.

## Writing

- [ ] Abstract added.
- [ ] Introduction outline updated.
- [ ] Chapter summaries standardized.
- [ ] Final conclusion added.
- [ ] Terminology standardized.
- [ ] Spelling and grammar checked.
- [ ] No proposal language remains.

## PDF

- [ ] Correct title and metadata.
- [ ] Chapter 9 title fixed.
- [ ] Table of contents updated.
- [ ] No unresolved references.
- [ ] No clipped figures, tables, or boxes.
- [ ] No placeholder or review markup remains.
- [ ] Final PDF visually inspected page by page.

---

# Minimum viable plan if time becomes very short

If only two or three days remain, do these tasks in this order:

1. Fix the title/front matter and add the abstract.
2. Correct the survival definition.
3. Disclose and clarify checkpoint-selection/test-set reuse.
4. Moderate the single-seed language in Chapter 6.
5. Integrate the existing `k=32` fine-tuning results.
6. Add the final Discussion and Conclusions chapter.
7. Fix the Chapter 9 title and perform a full PDF QA pass.

Skip new training before skipping any item in this minimum list.
