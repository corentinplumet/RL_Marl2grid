# bus14 shared-candidate actor + adaptive intervention budget

Chapter 4 tested the adaptive intervention budget (AIB) on the **flat MLP**
actor. Chapter 8 transfers the **shared candidate GNN** actor. This folder is
the missing cell: AIB applied to the architecture that is actually transferred.

## Why this experiment

Measured from the gated WCCI evaluations (`action_summary.json`, 46 runs,
4 agents):

| quantity | value |
|---|---|
| agent-steps where that agent is concerned (local rho >= threshold) | 0.82 % |
| agent-steps where the policy proposes a non-idle action | 33.6 % |
| P(>=2 agents concerned given >=1), independence estimate | ~1 % |

Training is ungated, so all four agents act freely and every one of them
receives the **identical** advantage: the reward is the same global scalar for
all agents and the critic is a single value on the joint observation
(`alg/mappo/core.py`, GAE loop). With `share_actor_gnn` and
`share_candidate_scorer`, ~99 % of the policy-gradient terms reaching the shared
scorer come from agent-steps where that agent's own domain was not stressed.

The AIB local-safe cost `w_i = sigmoid(k * (rho_safe - rho_i_local))` is the only
per-agent term available in this objective, so it is a credit-assignment change
as much as a sparsity mechanism. `rho_local` is dimensionless, so the timing rule
it encodes is grid-independent by construction.

## Configurations

The **full 2 x 2 x 2 x 2 grid**, matched one-to-one against `Z_nl_cas_hl_shared`:
2 families (`NL_shared`, `NLS_izar_shared`) x `mean`/`tmean` pooling x action
features off/on x do-nothing head off/on = **16 runs**, all at seed 0, exactly
like the `Z_` screen they pair with. (`NLS_shared` is identical to
`NLS_izar_shared` apart from naming, so it is not duplicated here.)

Every config differs from its `Z_` parent **only** in the run name and the
intervention block; all architecture, environment and optimisation lines are
byte-identical. That makes the `Y_` vs `Z_` comparison a clean paired contrast
in every cell.

The grid is run in full rather than pre-filtered to the best-transferring
variants, because the zero-shot transfer ranking is not a reliable predictor of
how a variant responds to a different training objective. Two cells are
scientifically load-bearing on their own:

- **`mean`+`f0`** collapsed in 20/20 zero-shot transfers. If AIB rescues it,
  the collapse is a *timing* failure; if it does not, it is a *representation*
  failure. Only running it answers this.
- **`tmean`+`f0`** is the most robust cell (1/20 collapses) and is the natural
  candidate for the best AIB transfer.

Budget `d = 0.10` follows chapter 4, which reached ~98.4 % survival at a 0.98
action-0 fraction. Evaluation stays **ungated** on purpose: the question is
whether the timing rule ends up in the weights instead of in an evaluation-time
override.

## Launch

```bash
DRY_RUN=true bash configs/no_leakage_config/Y_nl_cas_hl_aib/launch_izar.sh
bash configs/no_leakage_config/Y_nl_cas_hl_aib/launch_izar.sh
```

Filters match on the config name, e.g. `... launch_izar.sh tmean` or
`... launch_izar.sh NLS_izar`.

## Reading the result

Two diagnostics, in order:

1. **On bus14** — does survival hold near the `Z_` baseline while the non-idle
   rate collapses towards the ~1 % physical danger rate? That is AIB working.
2. **On WCCI** — transfer each checkpoint and compare against the matched `Z_`
   parent at the same action cap, *ungated*. The claim under test is that the
   AIB model no longer needs the local-rho override, and that the gap between
   its ungated and gated scores is much smaller than the parent's.

Do **not** select the checkpoint on bus14 survival: across the 16 shared-scorer
variants, bus14 survival correlates with WCCI difficult-cohort survival at
r = -0.01. Runs save both `<name>.tar` (best) and `final_<name>.tar`; transfer
and evaluate both.
