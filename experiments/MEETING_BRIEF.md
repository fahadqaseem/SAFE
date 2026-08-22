# Does the Hide-and-Seek loss fix SAFE's label smearing — or does dropping `cumsum`?

Prepared for the meeting with Prof. Zhu. Working code, results and figures are in
`experiments/`. Everything below was run locally; nothing is quoted from either paper's tables.

- **SAFE** — Gu et al., *Multitask Failure Detection for VLA Models*, NeurIPS 2025 ([arXiv:2506.09937](https://arxiv.org/abs/2506.09937)). This repo.
- **Hide-and-Seek (H&S)** — *Hide-and-Seek in Trajectories: Discovering Failure Signals for VLA Runtime Monitoring*, May 2026 ([arXiv:2605.30834](https://arxiv.org/abs/2605.30834)).

---

## 1. Four things to correct before we build on the original plan

**(a) The repo ships no data.** 416 KB of Python, zero rollouts. "Load the extracted
rollout features from the SAFE GitHub repository" is not possible. Features come either
from re-running OpenVLA/π0 on LIBERO (GPU + simulator) or from two Google Drive
real-robot dumps linked in the README. This is the binding constraint on everything else.

**(b) SAFE's time penalty is exponential, not linear — and it is switched off.**
[`failure_prob/model/utils.py:52`](../failure_prob/model/utils.py#L52):

```python
time_weights = 5 * torch.exp(- 3 * time_weights) + 1     # weight 6.0 at t=0 → 1.25 at t=T
```

And [`conf/__init__.py:146`](../failure_prob/conf/__init__.py#L146) sets
`use_time_weighting: bool = False`, with no released script in `scripts/` ever
turning it on. So the published SAFE MLP/LSTM numbers use **uniform** propagation of the
trajectory label to every timestep. This *strengthens* the smearing critique — the paper's
own mitigation is dead code — but we must not attack a mechanism they never used.

**(c) The headline comparison already exists.** H&S benchmarks against SAFE-MLP and
SAFE-LSTM on LIBERO-10/OpenVLA and reports 85.2 % vs 82.3 % bACC. Re-running it on SAFE's
data reproduces a published result; it is not a contribution. We need a question neither
paper answers.

**(d) The proposed intra-only loss cannot work, and the proposed plot is confounded.**
Two separate problems, both fatal to the original plan:

- H&S's total loss is `L = L_inter + λ·L_intra`. `L_intra` only constrains the *difference*
  between post- and pre-onset means *inside* failure trajectories. It is invariant to a
  global shift and says nothing about success trajectories. Alone, it is degenerate.
  Measured below: 0.531 balanced accuracy, i.e. chance.
- SAFE's MLP applies `sigmoid` then **`cumsum`** ([`indep.py:61`](../failure_prob/model/indep.py#L61),
  `cumsum: bool = True`). A cumulative sum of strictly positive increments is monotone
  non-decreasing, so that head **cannot** stay flat and then spike under *any* loss.
  A "SAFE smears / ours spikes" figure that also flips `cumsum` off is not evidence
  about the loss. The confound has to be controlled, or the first reviewer kills it.

Also worth knowing: `t_onset = argmax_t(s_t − s_{t−1})` depends on the model's own scores.
Let gradients through it and the model minimises the loss by making one arbitrary step-jump
large instead of by localising failure. We compute it under `no_grad`
([`hns_loss.py:66`](../failure_prob/model/hns_loss.py#L66)).

---

## 2. The question we propose instead

> H&S changes the loss *and* the temporal aggregation *and* the backbone at once.
> **How much of the gain is the contrastive loss, and how much is simply not accumulating
> the score?**

Cheap to answer, nobody has, and it decides whether "intra-trajectory contrastive loss"
is the right thing to build on. We cross the two factors on **one shared architecture**
— SAFE's own 2×256 MLP probe — so the only differences are the two switches.

Two metrics that neither paper reports, both threshold-free and scale-free so a `cumsum`
score (grows to ~T) and a per-step sigmoid (in [0,1]) stay comparable:

- **smear AUC** — separability of *pre-onset* failure timesteps from success timesteps,
  computed **within relative-time bins** and averaged. 0.50 = no smearing. Binning is what
  stops a monotonically growing score from being scored as smearing per se.
- **post AUC** — same for *post-onset* timesteps. Higher = better.

Testbed: multitask synthetic rollouts where the pre-onset segment of a failure trajectory
is drawn from **exactly** the same distribution as a success trajectory of the same task —
label smearing in its pure form — with failure directions 75 % shared across tasks so that
zero-shot transfer to held-out tasks is possible at all. Onset is known exactly, which
SAFE's real data can only approximate through human annotation. Evaluation follows SAFE's
protocol: whole tasks held out, threshold calibrated on seen tasks.

---

## 3. What we found

Zero-shot on held-out tasks, mean ± sd over 3 seeds. Step-supervised ceiling = 0.835 post AUC.

| variant | smear AUC | post AUC | bal. acc | onset err |
|---|---|---|---|---|
| SAFE (cumsum, as published) | 0.484 ± 0.063 | 0.593 ± 0.020 | 0.572 ± 0.051 | +0.159 |
| SAFE + time weighting (dead code, enabled) | 0.507 ± 0.033 | 0.587 ± 0.064 | 0.539 ± 0.070 | −0.029 |
| **SAFE loss, no cumsum** | 0.504 ± 0.027 | **0.801 ± 0.040** | 0.614 ± 0.039 | −0.057 |
| H&S inter only | 0.494 ± 0.041 | 0.633 ± **0.287** | 0.596 ± 0.090 | −0.180 |
| H&S intra only | 0.517 ± 0.033 | 0.547 ± 0.117 | 0.531 ± 0.030 | +0.187 |
| **H&S inter+intra** | 0.518 ± 0.041 | 0.792 ± 0.024 | **0.675 ± 0.042** | +0.050 |
| H&S inter+intra, cumsum | 0.521 ± 0.038 | 0.678 ± 0.025 | 0.625 ± 0.082 | +0.062 |

**Three findings, in order of how much they should change what we do next.**

**1. Post-onset discriminability is the aggregation, not the loss.** Keeping SAFE's loss
and only removing `cumsum` moves post AUC 0.593 → 0.801, consistent across all 3 seeds and
close to the 0.835 step-supervised ceiling. H&S's full loss at matched aggregation gets
0.792 ± 0.024 — indistinguishable. On this metric the entire apparent advantage of H&S is
the one-line aggregation change. `figures: fig2_factorial.png`

**2. The loss earns its keep elsewhere — trajectory accuracy, and stability.** At matched
aggregation, H&S beats SAFE's loss on balanced accuracy, 0.675 ± 0.042 vs 0.614 ± 0.039.
And note `inter only`: post AUC 0.633 ± **0.287**, one seed collapsing to 0.302. The intra
term's real job is **stabilising** the inter term (0.792 ± 0.024, tight), not detecting
anything by itself. That is a cleaner account of the mechanism than "intra localises the
onset", and it is testable on real data.

**3. The "smeared curve" in published figures is partly a time-of-rollout artifact.**
smear AUC sits at 0.48–0.52 for *every* variant — no method assigns class-specific failure
evidence to nominal timesteps. But the left panel of `fig1_onset_aligned.png` shows SAFE's
loss without `cumsum` scoring *early* timesteps high (~0.62) — for successes and failures
alike. It looks like smearing in a raw plot and vanishes under a time-matched contrast.
This matters: it means the standard way of showing smearing is not measuring smearing.

The flat-then-rise shape the original plan predicted **does** appear — right panel of
`fig1_onset_aligned.png`, H&S flat near 0 until onset then rising sharply — but only in the
loss comparison at matched aggregation, and for a different reason than assumed.

`fig3_single_rollout.png` is the single-trajectory plot from the original Step 4, included
to show why it should not be the evidence: on one rollout the curves are dominated by noise
and the per-step SAFE variant is saturated at 1.0 throughout. Onset-aligned population
averages are the defensible version.

### What this does not show

Synthetic data, chosen so the onset is known exactly — it tests **mechanism**, not either
paper's numbers, and cannot substitute for LIBERO-10. Three seeds: finding 1 is large and
consistent, finding 2 is +0.06 with ±0.04 and is suggestive rather than settled. Results
depend on generator choices (75 % shared failure direction, ramp shape), all in
`experiments/smearing_experiment.py`. No hyperparameter search was run for either loss;
H&S margins are at the paper's defaults.

---

## 4. What we would like to decide in the meeting

1. **Is the factorial framing the right contribution?** "Loss vs aggregation, and intra-as-
   stabiliser" is a real gap. If it holds on LIBERO-10 it is a short paper or a strong
   workshop submission; if it does not, we have saved ourselves from building on a
   misattributed gain.
2. **Compute for LIBERO-10.** The blocker. Re-running OpenVLA on LIBERO-10 to extract
   features needs GPU hours and the simulator; alternatively the two real-robot dumps in
   the README run today on CPU but match neither paper's headline benchmark. A third
   option: email Qiao Gu for the extracted features — cheapest by far if he says yes.
3. **Ask the H&S authors whether they ran the aggregation-matched control.** If they did
   and it is unpublished, finding 1 is already known and we redirect to finding 2.

**Ready to run the moment features exist.** The loss is implemented as a drop-in for SAFE's
own hydra pipeline — `python -m failure_prob.train model=hns model.loss_type=hns
model.agg=none` — and `model.loss_type=safe` reproduces SAFE's hinge through SAFE's own
code path, so the harness can be verified against the baseline before we trust any number.

## Files

| path | what |
|---|---|
| [`failure_prob/model/hns_loss.py`](../failure_prob/model/hns_loss.py) | H&S inter/intra losses + `no_grad` onset estimation |
| [`failure_prob/model/hns.py`](../failure_prob/model/hns.py) | probe with loss × aggregation as switches; `loss_type=safe` reproduces SAFE |
| [`experiments/smearing_experiment.py`](smearing_experiment.py) | testbed, metrics, 7-variant × 3-seed run |
| [`experiments/make_figures.py`](make_figures.py) | the three figures |
| `experiments/out/` | `summary.csv`, `results.json`, three PNGs |

Only one existing file was modified: an additive config registration in
`failure_prob/conf/__init__.py`. SAFE's `IndepModel` is untouched and remains the reference baseline.

---

# Part 2 — Real WidowX rollouts: SAFE's conformal guarantee does not survive task shift

Run on the downloaded `openvla_widowx/` data: **532 real OpenVLA-on-WidowX episodes,
8 distinct tasks, 46 % success, 50 steps each.** All numbers below are mean ± sd over
**12 runs** (6 three-task holdouts × 2 seeds). Reproduce with:

```bash
python3 experiments/cache_widowx.py          # 532 pickles -> one tensor, 3 s
python3 experiments/run_real_widowx.py       # 12 runs x 3 detectors x 4 alphas, ~5 min
python3 experiments/make_figures_real.py
```

## The hole

[`routines.py:104`](../failure_prob/utils/routines.py#L104) calls
`eval_functional_conformal(..., calib_split_names=["val_seen"], test_split_names=["val_unseen"])`.
SAFE calibrates its conformal band on held-out episodes of **seen** tasks and applies it to
**unseen** tasks. Functional CP's 1−α guarantee requires calibration and test data to be
exchangeable; different tasks are not. **So the false-alarm guarantee does not hold in the
exact zero-shot setting the paper is about.** We measured the size of that gap.

## The harness is verified before anything is claimed

| gate | result |
|---|---|
| Our reconstructed band == SAFE's own `get_one_sided_prediction_band` | rel. diff **1.1e-06** |
| ACI at γ=0 == split CP | diff **exactly 0** |
| Split CP on an **exchangeable** stream (held-out episodes of the *same* tasks) | **on nominal** — see below |
| SAFE-MLP unseen-task trajectory AUROC | **0.796 ± 0.052** — real signal, not plumbing |

The conformal band is SAFE's own code, imported unmodified; the loss is SAFE's
`get_time_weight` / `aggregate_monitor_loss`. The exchangeability control is the load-bearing
one: split CP hits nominal there, so the failures below are caused by task shift and not by us.

Coverage at nominal 0.80, split CP: **exchangeable 0.792 ± 0.129** → **unseen tasks 0.710 ± 0.148**.

## Result 1 — split CP under-covers on unseen tasks, for every detector

α = 0.20, nominal coverage 0.80, task-blocked deployment stream:

| detector | exchangeable (control) | unseen tasks | gap | AUROC |
|---|---|---|---|---|
| SAFE-MLP | 0.792 | **0.710 ± 0.148** | −0.090 | 0.796 ± 0.052 |
| SAFE-Embed (training-free) | 0.775 | **0.001 ± 0.003** | **−0.799** | 0.731 ± 0.082 |
| H&S inter+intra | 0.769 | **0.567 ± 0.144** | −0.233 | 0.727 ± 0.063 |

**SAFE-Embed is the headline.** Perfectly calibrated in-distribution (0.775 at nominal 0.80)
and **0.001** under task shift — it raises a false alarm on essentially *every* successful
unseen-task episode, at every α (see the middle panel of `fig5_calibration.png`, flat on
zero). Its balanced accuracy is 0.498: a coin flip. A guarantee that holds on seen tasks and
collapses this completely on unseen ones is worth a paragraph in anyone's paper.

## Result 2 — ACI restores nominal coverage, and collapses the variance

α = 0.20, unseen-task stream:

| detector | split CP (SAFE) | ACI γ=0.05 | **ACI γ=0.05 + growing cal.** | DtACI + growing cal. |
|---|---|---|---|---|
| SAFE-MLP | 0.710 ± 0.148 | 0.775 ± 0.076 | **0.817 ± 0.029** | 0.772 ± 0.042 |
| SAFE-Embed | 0.001 ± 0.003 | 0.289 ± 0.224 | **0.774 ± 0.018** | 0.639 ± 0.051 |
| H&S inter+intra | 0.567 ± 0.144 | 0.760 ± 0.055 | **0.797 ± 0.019** | 0.760 ± 0.033 |

The **standard deviation** matters as much as the mean: 0.148 → 0.029 for SAFE-MLP,
0.144 → 0.019 for H&S. Coverage stops being a lottery across task holdouts. For a robot
you would actually deploy, predictable false-alarm behaviour is the point.

## Result 3 — adapting the level alone is not enough; the calibration pool has to grow

SAFE-Embed with a **fixed** calibration pool: ACI gets from 0.001 only to 0.289 ± 0.224. It
saturates, because the attainable quantile is capped at `max(cal_scores)` and every
unseen-task score sits above it — no level can widen a band past the data it was built from.
Letting the pool absorb each revealed episode removes the ceiling: **0.774 ± 0.018**.

That is a concrete methodological finding: under severe shift, ACI needs the fully-online
variant, not just an adaptive α.

## Result 4 — validity is not bought by going silent

Right panel of `fig6_adaptation_power.png`. Balanced accuracy on the unseen stream, α = 0.20:

| detector | split CP | ACI + growing cal. |
|---|---|---|
| SAFE-MLP | 0.667 | **0.697** |
| SAFE-Embed | 0.498 | **0.663** |
| H&S inter+intra | 0.593 | 0.586 |

SAFE-MLP and SAFE-Embed get *better* on balanced accuracy while becoming valid. H&S is the
honest exception: its TPR falls 0.619 → 0.375 and balanced accuracy is flat, so there the
band widening does cost real detection power.

## Result 5 — SAFE's commented-out line is the correct one

[`functional_predictor.py:154`](../failure_prob/utils/conformal/functional_predictor.py#L154)
uses `np.quantile`; the finite-sample `ceil((n+1)(1-α))` order statistic is implemented on the
next line and commented out. Restoring it improves coverage everywhere — SAFE-MLP at α=0.20:
0.710 → 0.772 on unseen tasks, 0.792 → 0.840 on the control. A one-line change worth ~4–6
coverage points.

## What this does not show

- **SAFE-MLP is the best detector here (AUROC 0.796 vs H&S 0.727).** On real WidowX data
  SAFE's own probe beats our H&S implementation — the opposite of the synthetic result in
  Part 1. Likely causes: H&S margins left at the paper's defaults with no tuning, 50-step
  episodes, and H&S being designed around an LSTM backbone with window pooling rather than
  SAFE's per-step MLP. We are not claiming H&S is better on this data. It is not.
- **ACI cannot react instantly to a task switch.** In `fig4_rolling_coverage.png` coverage
  still dips after each task boundary and the final block stays below nominal for every
  method. ACI improves the average and the variance, not the transient.
- 8 tasks, ~200-episode streams. ACI's guarantee is asymptotic; this is short.
- One VLA, one embodiment. No failure-onset annotations in this dataset, so no
  onset-localisation metrics — that is what Part 1's synthetic study is for.
- ACI needs the episode outcome after each episode. SAFE already assumes that supervision for
  training, but offline CP does not need it at deployment.
- `SAFE-Embed` reduced to PCA-128 before the top-k distance, to keep a 26.6k × 10k pairwise
  distance tractable on CPU. Deviation from `EmbedModel`'s full 4096-dim distance.
- Standalone harness, not `python -m failure_prob.train` — SAFE's loss and conformal code run
  verbatim, but their hydra/wandb entrypoint is not exercised.

## Figures

| file | what |
|---|---|
| `out/fig4_rolling_coverage.png` | rolling coverage along the deployment stream, task boundaries marked |
| `out/fig5_calibration.png` | nominal vs realised coverage — on-diagonal = valid |
| `out/fig6_adaptation_power.png` | α_t adaptation, coverage gap, and detection power |
| `out/real_cp_table_{exch,blocked}.csv` | full detector × method tables |

## Suggested next step

The gap is real, measured, and fixed by a method with a published guarantee. The natural paper
is *"conformal failure detection for VLAs is not valid zero-shot, and online CP fixes it"* —
with the honest caveat that the transient after a task switch is still open. Reproducing this
on LIBERO-10 (where both SAFE and H&S report numbers) is the obvious next ask, and it needs
GPU hours to regenerate features.
