"""Shared post-hoc calibration utilities (temperature scaling + isotonic)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax as scipy_softmax
from sklearn.isotonic import IsotonicRegression


def nll_loss_hard(T, logits, labels):
    """NLL for fitting T against hard one-hot labels."""
    scaled = logits / T[0]
    probs = scipy_softmax(scaled, axis=1)
    probs = np.clip(probs, 1e-9, 1.0)
    return -np.mean(np.log(probs[np.arange(len(labels)), labels]))


def brier_loss_soft(T, logits, soft_targets):
    """Brier Score for fitting T against soft label distributions."""
    scaled = logits / T[0]
    probs = scipy_softmax(scaled, axis=1)
    return np.mean(np.sum((probs - soft_targets) ** 2, axis=1))


def _validate_logits(logits, name="logits"):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {logits.shape}")
    if not np.isfinite(logits).all():
        n_nan = int(np.isnan(logits).sum())
        n_inf = int(np.isinf(logits).sum())
        raise ValueError(
            f"{name} contains non-finite values (nan={n_nan}, inf={n_inf})"
        )
    return logits


def _assert_temperature_fit(result, x0=1.0, name="temperature"):
    """Reject silent optimizer failures that leave T at the starting value."""
    T = float(result.x[0])
    if not bool(result.success):
        raise RuntimeError(
            f"{name} fit failed: success={result.success}, "
            f"message={getattr(result, 'message', None)!r}, T={T}"
        )
    if T == float(x0):
        raise RuntimeError(
            f"{name} fit returned the untouched starting value T={T} "
            f"(x0={x0}); logits are likely degenerate or optimization did not move."
        )
    if not np.isfinite(T):
        raise RuntimeError(f"{name} fit returned non-finite T={T}")
    return T


def fit_temperature_hard(logits, labels):
    """Fit T* by minimising NLL against hard labels."""
    logits = _validate_logits(logits, name="hard-fit logits")
    labels = np.asarray(labels)
    x0 = [1.0]
    result = minimize(
        nll_loss_hard,
        x0=x0,
        args=(logits, labels),
        method="L-BFGS-B",
        bounds=[(0.01, 20.0)],
        options={"maxiter": 100},
    )
    return _assert_temperature_fit(result, x0=x0[0], name="fit_temperature_hard")


def fit_temperature_soft(logits, soft_targets):
    """Fit T* by minimising Brier Score against soft label distributions."""
    logits = _validate_logits(logits, name="soft-fit logits")
    soft_targets = np.asarray(soft_targets, dtype=np.float64)
    x0 = [1.0]
    result = minimize(
        brier_loss_soft,
        x0=x0,
        args=(logits, soft_targets),
        method="L-BFGS-B",
        bounds=[(0.01, 20.0)],
        options={"maxiter": 100},
    )
    return _assert_temperature_fit(result, x0=x0[0], name="fit_temperature_soft")


def compute_ece(probs, labels, n_bins=15):
    """Expected Calibration Error with equal-width bins."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / len(labels)) * abs(acc - conf)
    return float(ece)


def compute_brier_soft(probs, soft_targets):
    """Brier Score against full soft label distribution. Lower is better."""
    return float(np.mean(np.sum((probs - soft_targets) ** 2, axis=1)))


def compute_accuracy(probs, labels):
    return float((probs.argmax(axis=1) == labels).mean())


class MulticlassIsotonicCalibrator:
    """Per-class one-vs-rest isotonic regression on softmax probabilities.

    Chosen over CalibratedClassifierCV because we calibrate an already-trained
    neural net from cached logits/softmax scores, not a sklearn classifier.
    OvR isotonic on class scores is the standard multiclass post-hoc approach
    and mirrors temperature scaling as a post-hoc map from model outputs.
    """

    def __init__(self):
        self.calibrators = None
        self.n_classes = None

    def fit_hard(self, logits, labels):
        """Fit against hard integer labels (mirrors T_hard / NLL role)."""
        probs = scipy_softmax(logits, axis=1)
        self.n_classes = probs.shape[1]
        self.calibrators = []
        for c in range(self.n_classes):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            y = (labels == c).astype(np.float64)
            ir.fit(probs[:, c], y)
            self.calibrators.append(ir)
        return self

    def fit_soft(self, logits, soft_targets):
        """Fit against soft label distributions (mirrors T_oracle / soft role).

        For each class c, fit isotonic regression of P_model(c) -> soft_target(c).
        Soft targets are valid continuous responses in [0, 1], so this directly
        uses the soft distribution rather than sampling hard labels from it.
        """
        probs = scipy_softmax(logits, axis=1)
        soft_targets = np.asarray(soft_targets, dtype=np.float64)
        assert soft_targets.shape == probs.shape
        self.n_classes = probs.shape[1]
        self.calibrators = []
        for c in range(self.n_classes):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            ir.fit(probs[:, c], soft_targets[:, c])
            self.calibrators.append(ir)
        return self

    def predict_proba(self, logits):
        if self.calibrators is None:
            raise RuntimeError("Calibrator must be fit before predict_proba")
        probs = scipy_softmax(logits, axis=1)
        calibrated = np.column_stack(
            [ir.predict(probs[:, c]) for c, ir in enumerate(self.calibrators)]
        )
        calibrated = np.clip(calibrated, 1e-9, None)
        calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
        return calibrated
