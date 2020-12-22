import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from scipy.stats import norm


def calibration_plot(y_pred, y_true, bins=100, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111)

    y_true = y_true.reshape(-1, 1)
    prob = np.sum(
        y_pred[:, :, 0] * (1 - norm.cdf((y_true - y_pred[:, :, 1]) / y_pred[:, :, 2])),
        axis=-1, keepdims=True)
    sns.distplot(prob, norm_hist=True, bins=bins, hist_kws={'range': (0, 1)}, kde=False, ax=ax)
    ax.axhline(1., linestyle='--', color='r')
    ax.set_xlim(0, 1)
    ax.set_ylim(0)
    return ax


def true_predicted(y_true, y_pred, agg='mean', quantile=True, ms=None, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    if quantile:
        c_quantile = np.sum(y_pred[:, :, 0] * (1 - norm.cdf((y_true.reshape(-1, 1) - y_pred[:, :, 1]) / y_pred[:, :, 2])),
                            axis=-1, keepdims=False)
    else:
        c_quantile = None

    if agg == 'mean':
        y_pred_point = np.sum(y_pred[:, :, 0] * y_pred[:, :, 1], axis=1)
    elif agg == 'point':
        y_pred_point = y_pred
    else:
        raise ValueError(f'Aggregation type "{agg}" unknown')

    limits = (np.min(y_true) - 0.5, np.max(y_true) + 0.5)
    ax.plot(limits, limits, 'k-', zorder=1)
    if ms is None:
        cbar = ax.scatter(y_true, y_pred_point, c=c_quantile, cmap='coolwarm', zorder=2)
    else:
        cbar = ax.scatter(y_true, y_pred_point, s=ms, c=c_quantile, cmap='coolwarm', zorder=2)
    ax.set_xlabel('$y_{true}$')
    ax.set_ylabel('$y_{pred}$')

    r2 = metrics.r2_score(y_true, y_pred_point)
    ax.text(min(np.min(y_true), limits[0]), max(np.max(y_pred_point), limits[1]), f"$R^2={r2:.2f}$", va='top')

    return ax, cbar