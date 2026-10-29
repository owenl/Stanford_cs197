"""
encoders.py
===========

Numerical preprocessing, also called encoding. 
Only relevant for Project 3!

Every encoder has the same four parts (the "interface"):

    encoder.fit(x)                  learn from a 1-D array of training values
    encoder.transform(x)            array of shape (N,)   ->  array of shape (N, m)
    encoder.inverse_transform(z)    array of shape (N, m) ->  array of shape (N,)
"""

import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import QuantileTransformer, PowerTransformer


class StandardEncoder:
    """x  ->  (x - mean) / std.   Output has m = 1 dimension."""
    """
    This is equivalent to sklearn StandardScaler
    """
    def fit(self, x):
        x = np.asarray(x, dtype=float)
        self.mean_ = x.mean()
        self.std_ = x.std() if x.std() > 1e-8 else 1.0   # avoid dividing by zero
        self.output_dim_ = 1
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        return (x - self.mean_) / self.std_

    def inverse_transform(self, z):
        z = np.asarray(z, dtype=float)
        return z[:, 0] * self.std_ + self.mean_


class QuantileEncoder:
    """Rank -> uniform -> Gaussian quantiles (sklearn QuantileTransformer).
    Output m = 1. Robust to outliers and skew; destroys original spacing."""
    def fit(self, x):
        x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
        self.qt_ = QuantileTransformer(output_distribution="normal",
                                       n_quantiles=min(100, len(x)),
                                       random_state=42)
        self.qt_.fit(x.reshape(-1, 1))
        self.output_dim_ = 1
        return self
    def transform(self, x):
        return self.qt_.transform(np.asarray(x, dtype=float).reshape(-1, 1))
    def inverse_transform(self, z):
        return self.qt_.inverse_transform(np.asarray(z, dtype=float).reshape(-1, 1))[:, 0]
class ModeSpecificEncoder:
    """
    Mode-specific normalization, inspired by CTGAN.

    A feature like "hours worked per week" may have several peaks ("modes"),
    e.g. one near 20 (part-time) and one near 40 (full-time). A single mean and
    std describe that badly. So we:

        1. fit a Gaussian mixture (one bell curve per peak),
        2. find the peak each value most likely belongs to,
        3. describe the value with two things:
               alpha        = (x - peak_mean) / (4 * peak_std)   position relative to its peak
               one_hot(k)   which peak it belongs to

    Output has m = 1 + K dimensions, where K is the number of peaks found.
    Example with K = 3:   x  ->  [0.17, 0, 1, 0]
    This was generated with help from Claude.
    """

    def __init__(self, max_components=10, weight_threshold=0.005, random_state=42):
        # FIX (was a bug in the provided skeleton): __init__ never stored
        # max_components / weight_threshold, but fit() uses both -> AttributeError.
        self.max_components = max_components
        self.weight_threshold = weight_threshold
        self.random_state = random_state

    def fit(self, x):
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        x = x[np.isfinite(x[:, 0])]  # FIX: drop NaNs, a GMM cannot fit them

        self.model_ = BayesianGaussianMixture(
            n_components=min(self.max_components, len(np.unique(x))),
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=0.001,
            max_iter=200,
            random_state=self.random_state,
        ).fit(x)

        # Keep only the peaks that a meaningful share of the data uses.
        keep = self.model_.weights_ > self.weight_threshold
        if not keep.any():
            keep[np.argmax(self.model_.weights_)] = True
        self.keep_ = np.flatnonzero(keep)

        self.means_ = self.model_.means_.reshape(-1)[self.keep_]
        variances = self.model_.covariances_.reshape(-1)[self.keep_]
        self.stds_ = np.maximum(np.sqrt(variances), 1e-6)

        self.output_dim_ = 1 + len(self.keep_)
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=float).reshape(-1, 1)

        # Which peak does each value most likely belong to?
        probs = self.model_.predict_proba(x)[:, self.keep_]
        peak = probs.argmax(axis=1)

        alpha = (x[:, 0] - self.means_[peak]) / (4.0 * self.stds_[peak])
        alpha = np.clip(alpha, -0.99, 0.99)
        one_hot = np.eye(len(self.keep_))[peak]
        return np.column_stack([alpha, one_hot])

    def inverse_transform(self, z):
        z = np.asarray(z, dtype=float)
        alpha = np.clip(z[:, 0], -1.0, 1.0)
        peak = z[:, 1:].argmax(axis=1)
        return alpha * 4.0 * self.stds_[peak] + self.means_[peak]
class QuantileBinningEncoder:
    """Cut x into n_bins equal-frequency bins -> one-hot. Output m = n_bins.
    Loses within-bin information; inverse returns the bin's training median."""
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
    def fit(self, x):
        x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
        edges = np.unique(np.quantile(x, np.linspace(0, 1, self.n_bins + 1)))
        self.edges_ = edges
        self.output_dim_ = len(edges) - 1
        idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, self.output_dim_ - 1)
        self.bin_medians_ = np.array([np.median(x[idx == k]) if (idx == k).any()
                                      else 0.5 * (edges[k] + edges[k + 1])
                                      for k in range(self.output_dim_)])
        return self
    def transform(self, x):
        x = np.asarray(x, dtype=float)
        idx = np.clip(np.searchsorted(self.edges_, x, side="right") - 1, 0, self.output_dim_ - 1)
        out = np.zeros((len(x), self.output_dim_)); out[np.arange(len(x)), idx] = 1.0
        return out
    def inverse_transform(self, z):
        return self.bin_medians_[np.asarray(z).argmax(axis=1)]
class QuantilePLEEncoder:
    """
    Piecewise-linear encoding (PLE) with quantile bins.  Output has m = (number of bins) dimensions.
 
    Idea: cut the feature into bins that each hold about the same number of training rows,
    then describe x by "how full" each bin is.
 
    Example: bin edges [0, 10, 20, 40]  ->  3 bins (0-10, 10-20, 20-40)
 
        x =  5   ->  [0.5, 0.0, 0.0]    half way through bin 1, later bins empty
        x = 15   ->  [1.0, 0.5, 0.0]    bin 1 full, half way through bin 2
        x = 30   ->  [1.0, 1.0, 0.5]    bins 1 and 2 full, half way through bin 3
 
    Reverse direction (inverse_transform): add up how much of each bin is filled.
 
        x = lowest edge + sum over bins of (fill_fraction * bin_width)
        e.g. [1.0, 0.5, 0.0]  ->  0 + 1.0*10 + 0.5*10 + 0.0*20 = 15
    
    This was generated with help from Claude.
    """
 
    def __init__(self, n_bins=8, clip_extremes=False):
        self.n_bins = n_bins
        # False: values below the first edge / above the last edge give fill fractions
        #        outside [0, 1] (extrapolation).  True: they are squashed into [0, 1].
        self.clip_extremes = clip_extremes
 
    def fit(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        x = x[np.isfinite(x)]
 
        # Bin edges = quantiles of the training data (0%, 12.5%, 25%, ... 100% for 8 bins).
        edges = np.quantile(x, np.linspace(0, 1, self.n_bins + 1))
        # Many repeated values can give identical edges; keep each edge once.
        self.edges_ = np.unique(edges)
 
        if len(self.edges_) < 2:
            raise ValueError("Feature has fewer than two unique quantile boundaries.")
        # FIX (was a bug in the provided skeleton): output_dim_ was never set,
        # so inverse_transform() - and TabularPreprocessor, which reads
        # encoder.output_dim_ - crashed with AttributeError.
        self.output_dim_ = len(self.edges_) - 1
        return self
 
    def _constrain(self, z):
        """
        Enforce the valid range of each column. Used by BOTH directions.
 
        Middle bins:  must lie in [0, 1]   (a bin cannot be less than empty or more than full)
        First bin:    at most 1            (it can go below 0 for x under the lowest edge)
        Last bin:     at least 0           (it can go above 1 for x over the highest edge)
        """
        z = z.copy()
 
        if self.clip_extremes:
            return np.clip(z, 0.0, 1.0)
 
        if z.shape[1] > 1:
            z[:, 0] = np.minimum(z[:, 0], 1.0)
            z[:, -1] = np.maximum(z[:, -1], 0.0)
            if z.shape[1] > 2:
                z[:, 1:-1] = np.clip(z[:, 1:-1], 0.0, 1.0)
        return z
 
    def transform(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        if not np.isfinite(x).all():
            raise ValueError("transform() expects finite values.")
 
        left = self.edges_[:-1]
        widths = np.diff(self.edges_)
 
        # For every bin: how far along the bin is x?  0 = at its left edge, 1 = at its right edge.
        z = (x[:, None] - left[None, :]) / widths[None, :]
        return self._constrain(z)
 
    def inverse_transform(self, z):
        z = np.asarray(z, dtype=float)
        if z.ndim != 2 or z.shape[1] != self.output_dim_:
            raise ValueError(f"Expected an array of shape (N, {self.output_dim_}).")
 
        widths = np.diff(self.edges_)
 
        # Generated data (e.g. from a VAE decoder) can be slightly out of range or noisy,
        # so first push it back into the valid range.
        z = self._constrain(z)
 
        # Start at the lowest edge and add the filled part of every bin.
        return self.edges_[0] + (z * widths[None, :]).sum(axis=1)
 
    def fit_transform(self, x):
        return self.fit(x).transform(x)
class YeoJohnsonEncoder:
    """Encoding 6 (our choice): Yeo-Johnson power transform + standardize.
    Motivation: many labs (creatinine, BUN, glucose, WBC) are right-skewed but
    can be <= 0 after centering, so log() does not apply. Yeo-Johnson learns a
    power per feature that makes it more Gaussian, keeps m = 1, is smooth and
    exactly invertible, and (unlike quantile) preserves monotone spacing."""
    def fit(self, x):
        x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
        self.pt_ = PowerTransformer(method="yeo-johnson", standardize=True)
        self.pt_.fit(x.reshape(-1, 1))
        self.output_dim_ = 1
        return self
    def transform(self, x):
        return self.pt_.transform(np.asarray(x, dtype=float).reshape(-1, 1))
    def inverse_transform(self, z):
        return self.pt_.inverse_transform(np.asarray(z, dtype=float).reshape(-1, 1))[:, 0]



class PeriodicEncoder:
    """Periodic (sin/cos) embedding, Gorishniy et al. 2022 style — the team's
    chosen sixth encoding from Week 2. The standardized value z is mapped to
    [sin(2*pi*c_j*z), cos(2*pi*c_j*z)] for k frequencies c_j ~ N(0, sigma),
    drawn once at fit time with a fixed seed (frozen features, not learned).
    Output dim = 2k. Not exactly invertible: inverse_transform is a
    least-squares linear readout fitted on the training values."""

    def __init__(self, n_frequencies=4, sigma=1.0, seed=0):
        self.n_frequencies = n_frequencies
        self.sigma = sigma
        self.seed = seed

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        self.mean_ = x.mean()
        self.std_ = x.std() if x.std() > 0 else 1.0
        rng = np.random.default_rng(self.seed)
        self.coef_ = rng.normal(0.0, self.sigma, size=self.n_frequencies)
        self.output_dim_ = 2 * self.n_frequencies
        z = self.transform(x)
        A = np.hstack([z, np.ones((len(x), 1))])
        self.readout_, *_ = np.linalg.lstsq(A, x, rcond=None)
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        z = (x - self.mean_) / self.std_
        ang = 2 * np.pi * np.outer(z, self.coef_)
        return np.hstack([np.sin(ang), np.cos(ang)])

    def inverse_transform(self, z):
        z = np.atleast_2d(np.asarray(z, dtype=float))
        A = np.hstack([z, np.ones((len(z), 1))])
        return (A @ self.readout_).ravel()


class LogEncoder:
    """Log transformation + standardization — one of the team's six encodings
    (per progress_report5.tex 'Next week's plan': 'log transformation').
    x -> log(x - train_min + 1), then standardized with train statistics."""

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        self.min_ = float(x.min())
        z = np.log(x - self.min_ + 1.0)
        self.mean_ = float(z.mean())
        self.std_ = float(z.std()) if z.std() > 0 else 1.0
        self.output_dim_ = 1
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        z = np.log(np.clip(x - self.min_ + 1.0, 1e-9, None))
        return ((z - self.mean_) / self.std_).reshape(-1, 1)

    def inverse_transform(self, z):
        z = np.asarray(z, dtype=float).ravel()
        return np.exp(z * self.std_ + self.mean_) - 1.0 + self.min_


class MinMaxBinningEncoder:
    """Min-max (equal-width) binning + one-hot — one of the team's six encodings
    (per progress_report5.tex: 'min--max binning'). Bins span the training
    [min, max]; values outside are clipped to the edge bins. Inverse returns
    bin centres (lossy, like quantile binning)."""

    def __init__(self, n_bins=10):
        self.n_bins = n_bins

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        self.lo_, self.hi_ = float(x.min()), float(x.max())
        if self.hi_ <= self.lo_:
            self.hi_ = self.lo_ + 1.0
        self.edges_ = np.linspace(self.lo_, self.hi_, self.n_bins + 1)
        self.centers_ = 0.5 * (self.edges_[:-1] + self.edges_[1:])
        self.output_dim_ = self.n_bins
        return self

    def _bin(self, x):
        idx = np.digitize(np.asarray(x, dtype=float), self.edges_[1:-1])
        return np.clip(idx, 0, self.n_bins - 1)

    def transform(self, x):
        idx = self._bin(x)
        out = np.zeros((len(idx), self.n_bins))
        out[np.arange(len(idx)), idx] = 1.0
        return out

    def inverse_transform(self, z):
        z = np.atleast_2d(np.asarray(z, dtype=float))
        return self.centers_[np.argmax(z, axis=1)]
