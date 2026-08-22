"""
Online conformal prediction for VLA failure detection.

Why this exists
---------------
SAFE calibrates its functional conformal band on held-out episodes of SEEN tasks
and applies it to UNSEEN tasks (`failure_prob/utils/routines.py:104`,
`metrics.py:494`). Functional CP's 1-alpha guarantee needs calibration and test
data to be exchangeable, and different tasks are not. So the false-alarm
guarantee does not hold in the zero-shot setting the paper is about.

Adaptive Conformal Inference (ACI, Gibbs & Candes 2021) replaces the fixed level
with one that adapts to realised miscoverage:

    alpha_{t+1} = alpha_t + gamma * (alpha - err_t)

giving long-run coverage ~ alpha under arbitrary distribution shift, with no
exchangeability assumption. DtACI (Gibbs & Candes 2022) aggregates several gamma
"experts" so gamma need not be hand-tuned.

Reduction to a scalar quantile problem
--------------------------------------
SAFE's band at level alpha is

    band(t) = pred(t) + Q_{1-alpha}(S_cal) * mod(t)

where `pred` is the mean training-success trajectory, `mod` the modulation
trajectory, and for a trajectory x the nonconformity score is

    S(x) = max_t (x(t) - pred(t)) / mod(t)

A trajectory crosses the band somewhere iff S(x) > Q_{1-alpha}(S_cal). So the
whole thing is scalar quantile tracking, and `pred`/`mod`/`S` all come from
SAFE's own code (`regress`, `_get_modulation_trajectory`) rather than a
reimplementation.

One design decision worth flagging: SAFE's `Tfunc` modulation itself depends on
alpha (it trims training trajectories at an alpha-dependent quantile,
`functional_predictor.py:191-218`). We freeze `pred` and `mod` at the NOMINAL
alpha and let ACI adapt only the quantile level. ACI's theory assumes a fixed
score function and an adaptive level; letting the score function drift with
alpha_t would void it.
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# SAFE's own conformal code, imported unmodified (numpy-only import chain).
from failure_prob.utils.conformal.functional_predictor import (  # noqa: E402
    FunctionalPredictor, ModulationType, RegressionType, regress,
)


# ---------------------------------------------------------------------------
# Score function, built from SAFE's regress + modulation
# ---------------------------------------------------------------------------

class FunctionalScorer:
    """SAFE's functional-CP nonconformity score, frozen at the nominal alpha."""

    def __init__(self, train_success: np.ndarray, alpha_nominal: float,
                 modulation_type=ModulationType.Tfunc):
        assert train_success.ndim == 2, train_success.shape
        self.predictor = FunctionalPredictor(modulation_type, RegressionType.Mean)
        self.pred = regress(train_success, RegressionType.Mean)              # (1, T)
        self.mod = self.predictor._get_modulation_trajectory(
            train_success, self.pred, alpha_nominal
        )                                                                     # (1, T)
        self.alpha_nominal = alpha_nominal
        self._train = train_success

    def __call__(self, curves: np.ndarray) -> np.ndarray:
        """curves (N, T) -> scores (N,). One-sided, upper: large = anomalous."""
        if curves.ndim == 1:
            curves = curves[None, :]
        return np.max((curves - self.pred) / self.mod, axis=1)

    def band(self, q: float) -> np.ndarray:
        """The band trajectory for a given nonconformity quantile."""
        return (self.pred + q * self.mod)[0]

    def verify_against_safe(self, cal_success: np.ndarray, alpha: float) -> float:
        """Max abs difference between the band we reconstruct from our scores and
        the band SAFE's own `get_one_sided_prediction_band` returns, on the same
        train/calibration data. Should be 0 up to float error.

        Only meaningful at `alpha == self.alpha_nominal`, since SAFE recomputes
        the alpha-dependent modulation internally while we hold it frozen.
        """
        theirs = self.predictor.get_one_sided_prediction_band(
            self._train, cal_success, alpha, lower_bound=False
        )[0]
        ours = self.band(quantile(self(cal_success), alpha))
        scale = max(float(np.max(np.abs(theirs))), 1e-12)
        return float(np.max(np.abs(theirs - ours))) / scale


def quantile(cal_scores: np.ndarray, alpha: float, finite_sample: bool = False) -> float:
    """Quantile of calibration scores at level 1 - alpha.

    `finite_sample=False` reproduces SAFE exactly: plain `np.quantile`, as in
    `functional_predictor.py:154`. `finite_sample=True` uses the
    ceil((n+1)(1-alpha))-th order statistic that SAFE has implemented directly
    below that line but commented out -- the version that actually carries the
    finite-sample guarantee.

    Returns +inf when the requested level exceeds what n calibration points can
    support (alpha < 1/(n+1)); the band is then vacuous and never fires.
    """
    # float64 throughout: the scores arrive as float32 from torch, and mixing
    # dtypes makes np.quantile interpolate slightly differently, which showed up
    # as a spurious ACI(gamma=0) vs split-CP mismatch.
    cal_scores = np.asarray(cal_scores, dtype=np.float64)
    n = len(cal_scores)
    if n == 0:
        return np.inf
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if finite_sample:
        k = int(np.ceil((n + 1) * (1 - alpha)))
        if k > n:
            return np.inf
        return float(np.sort(cal_scores)[k - 1])
    if alpha <= 0.0:
        return np.inf
    return float(np.quantile(cal_scores, 1 - alpha))


# ---------------------------------------------------------------------------
# The three CP schemes over a stream of episodes
# ---------------------------------------------------------------------------
#
# Stream convention. Each step is one episode, in deployment order. We are given
#   stream_scores   (M,)  nonconformity score per episode
#   is_success      (M,)  bool, revealed at episode end
#
# The band is calibrated on SUCCESS trajectories, so the coverage claim is about
# successes: a success stays inside the band with prob >= 1 - alpha. Coverage
# events -- and therefore the alpha updates -- are defined on the success
# subsequence only. Failure episodes still record whether they fired (that is
# detection power) but do not move alpha.


def _finish(fired, err, alpha_t, q_t, is_success):
    err = np.asarray(err, dtype=float)
    succ = np.asarray(is_success, dtype=bool)
    n_succ = int(succ.sum())
    return {
        "fired": np.asarray(fired, dtype=bool),
        "err": err,                                  # nan on failure episodes
        "alpha_t": np.asarray(alpha_t, dtype=float),
        "q_t": np.asarray(q_t, dtype=float),
        # realised false-alarm rate on successes; coverage = 1 - that
        "false_alarm_rate": float(np.nanmean(err)) if n_succ else np.nan,
        "coverage": float(1 - np.nanmean(err)) if n_succ else np.nan,
        "n_success": n_succ,
    }


def split_cp_stream(cal_scores, stream_scores, is_success, alpha,
                    finite_sample=False):
    """SAFE's scheme: one fixed band, computed once from the calibration pool."""
    q = quantile(cal_scores, alpha, finite_sample)
    fired, err, alphas, qs = [], [], [], []
    for s, ok in zip(stream_scores, is_success):
        f = bool(s > q)
        fired.append(f)
        err.append(float(f) if ok else np.nan)
        alphas.append(alpha)
        qs.append(q)
    return _finish(fired, err, alphas, qs, is_success)


def aci_stream(cal_scores, stream_scores, is_success, alpha, gamma,
               grow_calibration=False, finite_sample=False,
               clip=(1e-3, 1 - 1e-3)):
    """Adaptive Conformal Inference (Gibbs & Candes 2021).

        alpha_{t+1} = clip(alpha_t + gamma * (alpha - err_t))

    gamma = 0 reduces exactly to `split_cp_stream` (verified in the runner).

    grow_calibration: if True, each revealed SUCCESS episode's score is appended
    to the calibration pool ("fully online" CP). With a fixed pool the attainable
    quantile saturates at max(cal_scores), so if the shifted distribution sits
    entirely above that, adapting the level alone cannot restore coverage.
    Growing the pool removes that ceiling. Reported both ways.
    """
    cal = list(np.asarray(cal_scores, dtype=float))
    a_t = float(alpha)
    fired, err, alphas, qs = [], [], [], []

    for s, ok in zip(stream_scores, is_success):
        q = quantile(np.asarray(cal), a_t, finite_sample)
        f = bool(s > q)

        fired.append(f)
        alphas.append(a_t)
        qs.append(q)

        if ok:
            e = float(f)
            err.append(e)
            a_t = float(np.clip(a_t + gamma * (alpha - e), *clip))
            if grow_calibration:
                cal.append(float(s))
        else:
            err.append(np.nan)

    return _finish(fired, err, alphas, qs, is_success)


def dtaci_stream(cal_scores, stream_scores, is_success, alpha,
                 gammas=(0.005, 0.01, 0.05, 0.1), eta=2.72, sigma=0.02,
                 grow_calibration=False, finite_sample=False,
                 clip=(1e-3, 1 - 1e-3)):
    """DtACI (Gibbs & Candes 2022): exponentially-weighted aggregation over a
    grid of gamma experts, so gamma does not have to be chosen in advance.

    Each expert i keeps its own alpha^i and is scored by the pinball loss at the
    target level alpha. Weights are multiplicatively updated and mixed with the
    uniform distribution (parameter sigma) to keep them from collapsing.

    eta and sigma are NOT tuned here; they are fixed at the paper's suggested
    order of magnitude. The point of including DtACI is to show the result does
    not hinge on picking a good gamma, not to squeeze out the best number.
    """
    gammas = np.asarray(gammas, dtype=float)
    k = len(gammas)
    cal = list(np.asarray(cal_scores, dtype=float))
    a_exp = np.full(k, float(alpha))
    w = np.full(k, 1.0 / k)

    fired, err, alphas, qs = [], [], [], []

    for s, ok in zip(stream_scores, is_success):
        cal_arr = np.asarray(cal)
        a_bar = float(np.clip(np.dot(w, a_exp) / w.sum(), *clip))
        q = quantile(cal_arr, a_bar, finite_sample)
        f = bool(s > q)

        fired.append(f)
        alphas.append(a_bar)
        qs.append(q)

        if ok:
            e = float(f)
            err.append(e)

            # Pinball loss of each expert's own quantile against this score.
            q_exp = np.array([quantile(cal_arr, ai, finite_sample) for ai in a_exp])
            u = s - q_exp
            finite = np.isfinite(q_exp)
            loss = np.where(finite, alpha * np.maximum(u, 0)
                            + (1 - alpha) * np.maximum(-u, 0), 0.0)

            w_bar = w * np.exp(-eta * loss)
            tot = w_bar.sum()
            w = ((1 - sigma) * (w_bar / tot) + sigma / k) if tot > 0 \
                else np.full(k, 1.0 / k)

            # Each expert updates on its OWN miscoverage indicator.
            e_exp = (s > q_exp).astype(float)
            a_exp = np.clip(a_exp + gammas * (alpha - e_exp), *clip)

            if grow_calibration:
                cal.append(float(s))
        else:
            err.append(np.nan)

    out = _finish(fired, err, alphas, qs, is_success)
    out["final_weights"] = w
    return out


# ---------------------------------------------------------------------------
# Detection power, so validity is never reported on its own
# ---------------------------------------------------------------------------

def detection_metrics(res, curves, scorer, is_success):
    """TPR on failures, plus mean relative detection time among detected failures.

    Detection time uses the per-step band implied by that step's q_t, so an
    adaptive scheme is judged by the band it actually had at that episode.
    """
    is_success = np.asarray(is_success, dtype=bool)
    fail = ~is_success
    tpr = float(res["fired"][fail].mean()) if fail.any() else np.nan
    tnr = float(1 - res["fired"][is_success].mean()) if is_success.any() else np.nan

    det_times = []
    for i in np.where(fail)[0]:
        q = res["q_t"][i]
        if not np.isfinite(q):
            continue
        over = curves[i] > scorer.band(q)
        if over.any():
            det_times.append(int(np.argmax(over)) / len(over))

    return {
        "tpr": tpr,
        "tnr": tnr,
        "bal_acc": float(0.5 * (tpr + tnr)) if np.isfinite(tpr) and np.isfinite(tnr) else np.nan,
        "det_time": float(np.mean(det_times)) if det_times else np.nan,
        "n_detected": len(det_times),
    }


def rolling_coverage(res, window=30):
    """Coverage in a sliding window over the success subsequence, placed back on
    the stream index so it can be plotted against task boundaries."""
    err = res["err"]
    idx = np.where(~np.isnan(err))[0]
    e = err[idx]
    out_x, out_y = [], []
    for j in range(len(e)):
        lo = max(0, j - window + 1)
        out_x.append(idx[j])
        out_y.append(1 - e[lo:j + 1].mean())
    return np.asarray(out_x), np.asarray(out_y)
