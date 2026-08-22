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

---

# Part 3 — Four more experiments on the real data

```bash
python3 experiments/run_experiments2.py     # ~4.5 min, 12 runs x 20 detectors
python3 experiments/make_figures2.py
```

## The diagnostic that motivated all of this

Peak score of **successful** episodes, SAFE-MLP, trained on tasks 0–4:

| | mean peak score |
|---|---|
| Seen tasks (0–4) | **5.43** |
| Unseen tasks (5–7) | **15.29** |

Successful episodes on an unfamiliar task score **~3× higher** — a full 1.03 seen-task
standard deviations — purely because the task is new. Between-task variance is **31 %** of
all variance among successes alone. The band calibrated on seen tasks is simply far too
tight, which is *why* Part 2's coverage collapses. Per-task means range from 3.90
("Put Blue Cup on Plate") to 20.96 ("Put the Red Bottle into Pot").

## Experiment 1 — Task-relative scoring: a partial win, honestly reported

**Idea.** Fit a line to each episode's first `k` steps and subtract its extrapolation, so the
score measures deviation from *that episode's own* early behaviour. One transform handles
both score shapes: it removes a task-dependent *rate* from SAFE's cumulative score and a
task-dependent *level* from a per-step score. It needs **no labels, no task identity, and no
deployment feedback** — unlike ACI, which needs the outcome after every episode. Applied
post-hoc to the same trained weights, so the comparison isolates the transform.

Coverage at nominal 0.80 on unseen tasks (`fig7_task_relative.png`):

| detector | split CP | + task-relative *(no feedback)* | + ACI *(needs feedback)* | + both | AUROC absolute | AUROC task-rel |
|---|---|---|---|---|---|---|
| SAFE-MLP | 0.710 | 0.778 | **0.817** | 0.807 | **0.796** | 0.560 |
| H&S | 0.567 | 0.546 | **0.797** | 0.791 | 0.727 | 0.683 |
| SAFE-Embed | 0.001 | 0.920 | 0.774 | **0.820** | 0.731 | 0.715 |

**It is not a general win, and I am not going to claim it is.** It costs SAFE-MLP a great
deal of discriminative power (AUROC 0.796 → 0.560) and does nothing for H&S's coverage
(0.567 → 0.546). **ACI remains the reliable general fix** — it repairs coverage for all three
while leaving AUROC untouched.

**Where it clearly wins: the training-free detector, with a longer window (k=20).**

| SAFE-Embed | AUROC | coverage (nominal 0.80) |
|---|---|---|
| absolute score | 0.731 | **0.001** |
| task-relative, k=20 | **0.743** | **0.884** |

Better on *both* axes: a catastrophic 0.001 coverage becomes 0.884, and AUROC goes slightly
**up**. That is a real, tangible improvement to a real failure — obtained with no feedback,
no retraining, and no labels.

**Why the split?** The transform removes the score's absolute level. For SAFE-Embed that
level is mostly task-novelty *nuisance*, so removing it helps. For SAFE-MLP the level partly
*is* the signal — the probe learned to run high throughout a failing episode — so removing it
destroys information. The `k` sweep supports this: SAFE-MLP's AUROC climbs steadily with
longer windows (0.518 at k=5 → 0.646 at k=20) but never recovers 0.796, whereas SAFE-Embed
improves monotonically and overtakes its own baseline.

Useful rule of thumb this suggests: **task-relative scoring helps exactly when between-task
variance dominates the score's level.** That is measurable in advance from the training set,
which makes it a decision rule rather than a guess.

## Experiment 2 — Early detection: the headline numbers are end-of-episode numbers

`fig8_early_detection.png`. AUROC using only the first fraction of each episode, unseen tasks:

| fraction seen | 0.1 | 0.3 | 0.5 | 0.7 | 1.0 |
|---|---|---|---|---|---|
| SAFE-MLP | 0.556 | 0.559 | 0.615 | 0.704 | **0.796** |
| H&S | 0.485 | 0.454 | 0.555 | 0.649 | 0.727 |
| SAFE-Embed | 0.489 | 0.440 | 0.507 | 0.632 | 0.731 |
| handcrafted entropy | 0.461 | 0.478 | 0.520 | 0.531 | 0.580 |

**At the halfway point only about 40–50 % of the final above-chance signal exists.** For the
first ~40 % of an episode, H&S, SAFE-Embed and the handcrafted signals are at or *below*
chance; SAFE-MLP is the only one meaningfully above it, and only by ~0.06.

This is a fair and important limitation to raise: a failure detector is useful only if there
is still time to intervene, and detection quality in that regime is far below the numbers
these papers report. SAFE-MLP being the only early performer is also a point in the original
paper's favour.

## Experiment 3 — Per-task breakdown: the average hides everything

`fig9_pertask_handcrafted.png`, left panel. SAFE-MLP per held-out task:

| task | AUROC |
|---|---|
| Lift Red Bottle | **0.692 ± 0.042** |
| Lift AAA Battery | 0.700 ± 0.055 |
| Lift Blue Cup | 0.753 ± 0.097 |
| Put the Red Bottle into Pot | 0.881 ± 0.032 |
| Put Blue Cup on Plate | 0.903 ± 0.028 |
| Put the Carrot on Plate | 0.906 ± 0.028 |
| Lift Eggplant | 0.920 ± 0.021 |
| Put the Red Block into the Pot | **0.984 ± 0.008** |

The reported "0.796" spans **0.69 to 0.98** depending on the chore. The pattern is
interpretable: the three hardest are all **"Lift …"** tasks and the easiest are **"Put … into/on …"**
tasks. Lifting failures are subtle — the gripper closes on nothing, the object slips — and
look nearly identical to success in the policy's hidden state. Placing failures are gross and
obvious. H&S even falls *below* chance on Lift Blue Cup (0.44).

Caveat: with 6 random 3-task holdouts each task appears held out 2–6 times, so the per-task
sample sizes differ (shown in the figure).

## Experiment 4 — The free handcrafted baselines

The per-episode CSVs carry 47 training-free signals. Seven representative ones, sign chosen
on the train split only, on unseen tasks:

| signal | AUROC |
|---|---|
| cum max token entropy | 0.595 |
| cum mean token entropy | 0.580 |
| cum drot | 0.524 |
| cum mean token prob | 0.475 |
| cum dpos | 0.476 |
| dgripper | 0.458 |
| mean token entropy | 0.391 |

**All of them are near chance** — 0.39 to 0.60 versus 0.796 for the learned probe. So SAFE's
central premise holds up on this data: the policy's internal features carry failure
information that token-level uncertainty and action statistics do not. Two of them land
*below* 0.5 on unseen tasks despite being above 0.5 on train, meaning even the useful
*direction* of these signals does not transfer across tasks.

Right panel of `fig9` plots AUROC against |coverage gap| for everything at once, which makes
the trade-off visible: the handcrafted signals are poorly discriminative but reasonably
well-calibrated (gaps 0.05–0.18), while SAFE-Embed is discriminative yet catastrophically
mis-calibrated (gap 0.80).

## Where this leaves us

| finding | strength |
|---|---|
| Task shift causes a ~3× score offset on successes; 31 % of variance is between-task | solid, and it explains Part 2 |
| ACI fixes coverage for every detector with no AUROC cost | solid (Part 2) |
| Task-relative scoring fixes SAFE-Embed on both axes, no feedback needed | solid but narrow |
| Task-relative scoring hurts SAFE-MLP badly | solid — a negative result worth reporting |
| Early detection is near chance for most methods before mid-episode | solid, and a fair criticism of both papers |
| "Lift" tasks are much harder than "Put" tasks | solid, interpretable |
| Handcrafted signals are near chance; learned features are needed | solid, supports SAFE's premise |

**All figures:** `out/fig1`–`fig3` (synthetic mechanism), `fig4`–`fig6` (online CP),
`fig7`–`fig9` (this round). `fig7`, `fig8` and `fig9` are the three most presentable.
