"""Every figure the project produces, each saved into `reports/figures/`.

Imported by both `train.py` and the analysis notebook so the two never draw the
same chart twice from different code. Uses the non-interactive Agg backend so it
works headless; the notebook re-enables inline display itself.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src import config as cfg
from src.evaluate import format_inr

# --------------------------------------------------------------------------- #
# Shared style
# --------------------------------------------------------------------------- #
PALETTE = "colorblind"
CITY_ORDER = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Kolkata"]
DPI = 130


def set_style() -> None:
    sns.set_theme(style="whitegrid", palette=PALETTE)
    plt.rcParams.update({
        # DejaVu Sans is matplotlib's bundled default and is one of the few
        # available fonts that contains U+20B9, the Indian rupee sign. Seaborn's
        # theme otherwise resolves to Arial on macOS, which lacks the glyph and
        # renders every rupee label as a hollow box.
        "font.family": "DejaVu Sans",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 110,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.frameon": True,
        "figure.autolayout": False,
    })


set_style()


def _inr_axis(ax, axis: str = "y") -> None:
    """Label a rupee axis with Indian digit grouping."""
    fmt = mticker.FuncFormatter(lambda v, _: format_inr(v))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def save(fig, name: str) -> str:
    """Save a figure into reports/figures/ and return its path as a string."""
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return str(path.relative_to(cfg.PROJECT_ROOT))


def _city_order(df: pd.DataFrame) -> list[str]:
    present = set(df[cfg.COL_CITY].unique())
    return [c for c in CITY_ORDER if c in present] + sorted(present - set(CITY_ORDER))


# --------------------------------------------------------------------------- #
# Section 17 -- EDA
# --------------------------------------------------------------------------- #
def plot_rent_distribution(df: pd.DataFrame) -> str:
    """Rent histogram before and after the log1p transform."""
    rent = df[cfg.TARGET]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    axes[0].hist(rent, bins=80, color=sns.color_palette(PALETTE)[0],
                 edgecolor="white")
    axes[0].set_title(f"Raw Rent  (skew = {rent.skew():.2f})")
    axes[0].set_xlabel("Monthly rent")
    axes[0].set_ylabel("Number of listings")
    axes[0].set_yscale("log")
    _inr_axis(axes[0], "x")
    axes[0].annotate(
        f"max = {format_inr(rent.max())}\nmedian = {format_inr(rent.median())}",
        xy=(0.62, 0.78), xycoords="axes fraction", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.85))

    logged = np.log1p(rent)
    axes[1].hist(logged, bins=60, color=sns.color_palette(PALETTE)[2],
                 edgecolor="white")
    axes[1].set_title(f"log1p(Rent)  (skew = {logged.skew():.2f})")
    axes[1].set_xlabel("log1p(monthly rent)")
    axes[1].set_ylabel("Number of listings")

    fig.suptitle("Why the target is log-transformed: skew collapses from "
                 f"{rent.skew():.1f} to {logged.skew():.2f}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return save(fig, "eda_rent_distribution.png")


def plot_rent_by_city(df: pd.DataFrame) -> str:
    """Boxplot of rent per city on a log scale, with the per-city IQR story."""
    order = _city_order(df)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x=cfg.COL_CITY, y=cfg.TARGET, order=order,
                hue=cfg.COL_CITY, legend=False, ax=ax, fliersize=2)
    ax.set_yscale("log")
    _inr_axis(ax, "y")
    ax.set_title("Rent distribution by city (log scale)\n"
                 "Rent levels differ so much between cities that outlier "
                 "detection must be done per city")
    ax.set_xlabel("")
    ax.set_ylabel("Monthly rent (log scale)")

    global_upper = (df[cfg.TARGET].quantile(0.75) + 1.5 *
                    (df[cfg.TARGET].quantile(0.75) - df[cfg.TARGET].quantile(0.25)))
    ax.axhline(global_upper, color="crimson", ls="--", lw=1.4,
               label=f"Global IQR upper bound = {format_inr(global_upper)}")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    return save(fig, "eda_rent_by_city.png")


def plot_rent_vs_size(df: pd.DataFrame) -> str:
    """Rent vs Size scattered and coloured by city, with per-city fitted slopes."""
    order = _city_order(df)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    sns.scatterplot(data=df, x=cfg.COL_SIZE, y=cfg.TARGET, hue=cfg.COL_CITY,
                    hue_order=order, alpha=0.55, s=18, ax=axes[0],
                    edgecolor="none")
    axes[0].set_yscale("log")
    _inr_axis(axes[0], "y")
    axes[0].set_title("Rent vs Size, coloured by city")
    axes[0].set_xlabel("Size (sq ft, as reported)")
    axes[0].set_ylabel("Monthly rent (log scale)")
    axes[0].legend(title="City", fontsize=8, title_fontsize=9)

    # The key observation: rent scales with size, but the SLOPE differs sharply
    # between cities. Plotting median rent per sqft makes the gap explicit.
    d = df.copy()
    d["rps"] = d[cfg.TARGET] / d[cfg.COL_SIZE]
    med = d.groupby(cfg.COL_CITY)["rps"].median().reindex(order)
    bars = axes[1].bar(med.index, med.to_numpy(),
                       color=sns.color_palette(PALETTE)[:len(med)])
    axes[1].bar_label(bars, fmt="%.1f", padding=2, fontsize=9)
    axes[1].set_title(f"Median rent per sq ft by city\n"
                      f"Mumbai is {med.iloc[0] / med.iloc[-1]:.1f}x "
                      f"{med.index[-1]}")
    axes[1].set_ylabel("Rupees per sq ft per month")
    axes[1].set_xlabel("")
    fig.tight_layout()
    return save(fig, "eda_rent_vs_size.png")


def plot_avg_rent_by_category(df: pd.DataFrame) -> str:
    """Average rent by BHK, Furnishing Status and Tenant Preferred."""
    cols = [cfg.COL_BHK, cfg.COL_FURNISHING, cfg.COL_TENANT]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, col in zip(axes, cols):
        grp = df.groupby(col)[cfg.TARGET].agg(["mean", "median", "size"])
        grp = grp.sort_index() if col == cfg.COL_BHK else grp.sort_values("mean")
        x = np.arange(len(grp))
        ax.bar(x - 0.2, grp["mean"], width=0.4, label="mean")
        ax.bar(x + 0.2, grp["median"], width=0.4, label="median")
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in grp.index], rotation=15, ha="right")
        _inr_axis(ax, "y")
        ax.set_title(f"Rent by {col}")
        ax.set_ylabel("Monthly rent")
        ax.legend(fontsize=8)
        for xi, n in zip(x, grp["size"]):
            ax.annotate(f"n={n}", (xi, 0), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=7,
                        color="dimgrey")
    fig.suptitle("Mean vs median rent by category (the mean/median gap shows "
                 "how far outliers pull the average)", fontweight="bold")
    fig.tight_layout()
    return save(fig, "eda_avg_rent_by_category.png")


def plot_correlation_heatmap(df_eda: pd.DataFrame) -> str:
    """Correlation heatmap of the numeric features and the target."""
    cols = [c for c in [cfg.COL_BHK, cfg.COL_SIZE, cfg.COL_BATHROOM,
                        cfg.FEAT_FLOOR_NUM, cfg.FEAT_TOTAL_FLOORS,
                        cfg.FEAT_CARPET_EQUIV, cfg.TARGET]
            if c in df_eda.columns]
    d = df_eda[cols].copy()
    d["log1p(Rent)"] = np.log1p(d[cfg.TARGET])
    corr = d.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation of numeric features with rent\n"
                 "(log1p(Rent) row included -- correlations are stronger on the "
                 "log scale)")
    fig.tight_layout()
    return save(fig, "eda_correlation_heatmap.png")


def plot_listings_per_city(df: pd.DataFrame) -> str:
    """Listing counts and locality diversity per city."""
    order = _city_order(df)
    counts = df[cfg.COL_CITY].value_counts().reindex(order)
    localities = df.groupby(cfg.COL_CITY)[cfg.COL_LOCALITY].nunique().reindex(order)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    b1 = axes[0].bar(counts.index, counts.to_numpy(),
                     color=sns.color_palette(PALETTE)[:len(counts)])
    axes[0].bar_label(b1, padding=2, fontsize=9)
    axes[0].set_title("Listings per city")
    axes[0].set_ylabel("Number of listings")

    b2 = axes[1].bar(localities.index, localities.to_numpy(),
                     color=sns.color_palette(PALETTE)[:len(localities)])
    axes[1].bar_label(b2, padding=2, fontsize=9)
    axes[1].set_title("Distinct localities per city\n"
                      "(more localities per listing = thinner coverage = harder "
                      "to predict)")
    axes[1].set_ylabel("Unique `Area Locality` values")
    fig.tight_layout()
    return save(fig, "eda_listings_per_city.png")


def plot_outlier_overview(df: pd.DataFrame, flags_per_city: pd.Series,
                          flags_global: pd.Series) -> str:
    """Compare what a per-city IQR rule flags against a global one."""
    order = _city_order(df)
    t = pd.DataFrame({"city": df[cfg.COL_CITY].to_numpy(),
                      "per_city": np.asarray(flags_per_city),
                      "global": np.asarray(flags_global)})
    agg = t.groupby("city").agg(n=("per_city", "size"),
                                per_city=("per_city", "sum"),
                                glob=("global", "sum")).reindex(order)
    pc = agg["per_city"] / agg["n"] * 100
    gl = agg["glob"] / agg["n"] * 100

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(agg))
    b1 = ax.bar(x - 0.2, pc.to_numpy(), width=0.4, label="Per-city IQR (used)")
    b2 = ax.bar(x + 0.2, gl.to_numpy(), width=0.4, label="Global IQR (rejected)",
                color="crimson", alpha=0.8)
    ax.bar_label(b1, fmt="%.1f%%", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.1f%%", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(agg.index)
    ax.set_ylabel("% of the city's listings flagged as outliers")
    ax.set_title("Why outlier detection is done PER CITY\n"
                 "A single global IQR rule would discard a large share of "
                 "legitimate Mumbai listings")
    ax.legend()
    fig.tight_layout()
    return save(fig, "eda_outlier_per_city_vs_global.png")


# --------------------------------------------------------------------------- #
# Section 7 -- model comparison
# --------------------------------------------------------------------------- #
def plot_rmse_by_model(comparison: pd.DataFrame) -> str:
    """Test RMSE by model, with CV RMSE and its fold-to-fold spread alongside.

    Plotted on a log scale: an unregularised polynomial expansion can post an
    RMSE twenty times every other model's, which on a linear axis would flatten
    every informative bar.
    """
    d = comparison.sort_values("test_rmse")
    x = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    b1 = ax.bar(x - 0.2, d["test_rmse"], width=0.4, label="Held-out test RMSE")
    b2 = ax.bar(x + 0.2, d["cv_rmse_mean"], width=0.4,
                yerr=d["cv_rmse_std"], capsize=4,
                label="5-fold CV RMSE (mean ± std)",
                color=sns.color_palette(PALETTE)[2])
    ax.set_yscale("log")
    ax.bar_label(b1, fmt=lambda v: format_inr(v), fontsize=7.5, padding=2,
                 rotation=90)
    ax.bar_label(b2, fmt=lambda v: format_inr(v), fontsize=7.5, padding=2,
                 rotation=90)
    ax.set_ylim(top=float(d[["test_rmse", "cv_rmse_mean"]].to_numpy().max()) * 6)
    ax.set_xticks(x)
    ax.set_xticklabels(d["model"], rotation=20, ha="right")
    _inr_axis(ax, "y")
    ax.set_ylabel("RMSE (monthly rent, log scale)")
    ax.set_title("Model comparison — RMSE (lower is better, log scale)\n"
                 "CV RMSE carries a very large fold-to-fold spread: on this "
                 "target it is dominated by a few extreme listings per fold")
    ax.legend()
    fig.tight_layout()
    return save(fig, "model_rmse_comparison.png")


def _bar_panel(ax, labels, values, title: str, rupees: bool) -> None:
    """Bar panel that stays readable when one model's metric is pathological.

    An unregularised polynomial expansion can score R² = -200 or an RMSE twenty
    times every other model's. Plotted on a shared linear axis that flattens all
    the informative bars to invisible slivers, so the axis is clipped to the
    range of the sane models and any bar beyond it is drawn to the edge and
    labelled with its true value.
    """
    v = np.asarray(values, dtype="float64")
    bars = ax.bar(np.arange(len(v)), np.clip(v, None, None),
                  color=sns.color_palette(PALETTE)[:len(v)])

    finite = v[np.isfinite(v)]
    lo, hi = float(finite.min()), float(finite.max())
    srt = np.sort(finite)

    # Clip the top if the worst value dwarfs the next one.
    if len(srt) > 1 and srt[-1] > 0 and srt[-1] > 4 * max(abs(srt[-2]), 1e-9):
        hi = float(srt[-2]) * 1.35
    # Clip the bottom likewise (matters for R², which can go far negative).
    if len(srt) > 1 and srt[0] < 0 and abs(srt[0]) > 4 * max(abs(srt[1]), 1e-9):
        lo = float(srt[1]) - abs(float(srt[1])) * 0.35 - 1e-9
    lo = min(lo, 0.0) if hi > 0 else lo
    pad = (hi - lo) * 0.18 or 1.0
    ax.set_ylim(lo - pad * 0.25, hi + pad)

    fmt = (lambda x: format_inr(x)) if rupees else (lambda x: f"{x:.3f}")
    for i, (bar, val) in enumerate(zip(bars, v)):
        off_scale = val > hi or val < lo
        if off_scale:
            bar.set_hatch("///")
            bar.set_edgecolor("crimson")
            ax.annotate(f"{fmt(val)}\n(off scale)",
                        xy=(i, hi if val > hi else lo),
                        xytext=(0, -30 if val > hi else 12),
                        textcoords="offset points", ha="center", fontsize=7.5,
                        color="crimson", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec="crimson", lw=0.8, alpha=0.95))
        else:
            ax.annotate(fmt(val), xy=(i, val),
                        xytext=(0, 3 if val >= 0 else -11),
                        textcoords="offset points", ha="center", fontsize=8)

    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(np.arange(len(v)))
    ax.set_title(title)
    if rupees:
        _inr_axis(ax, "y")


def plot_metric_panel(comparison: pd.DataFrame) -> str:
    """R² / RMSE / MAE / generalisation gap for every model, side by side."""
    d = comparison.sort_values("test_rmse")
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9))
    panels = [
        ("test_r2", "Test R² (higher is better)", False),
        ("test_rmse", "Test RMSE (lower is better)", True),
        ("test_mae", "Test MAE (lower is better)", True),
        ("cv_overfit_gap_rmse",
         "CV generalisation gap: validation − train RMSE\n"
         "(the sound overfitting check; see §12)", True),
    ]
    for ax, (col, title, rupees) in zip(axes.ravel(), panels):
        if col not in d.columns:
            ax.axis("off")
            continue
        _bar_panel(ax, d["model"].tolist(), d[col].to_numpy(), title, rupees)
        ax.set_xticklabels(d["model"], rotation=22, ha="right", fontsize=8)

    fig.suptitle("Model comparison across all reported metrics\n"
                 "Hatched bars fall outside the plotted range and are labelled "
                 "with their true value",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return save(fig, "model_metric_panel.png")


# --------------------------------------------------------------------------- #
# Section 8 -- residual plots for ALL models
# --------------------------------------------------------------------------- #
def plot_residual_grid(residuals: dict[str, pd.DataFrame]) -> str:
    """Residuals-vs-predicted for every model, as a comparative grid.

    The case study lists Residual Plot as a *comparison* metric, not merely a
    best-model diagnostic, so all models appear side by side at a shared scale.

    Axis limits are derived from the ACTUAL rent distribution rather than from
    the predictions. An unregularised polynomial model can predict tens of lakhs
    for a flat worth twenty thousand; letting that set the shared scale would
    compress every other panel into an unreadable sliver. Points that fall
    outside the window are counted in each panel instead.
    """
    names = list(residuals)
    ncols = 3
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.4 * nrows),
                             squeeze=False)

    actual = next(iter(residuals.values()))["actual"]
    x_hi = float(actual.quantile(0.99))
    y_lim = float(actual.quantile(0.99))

    for idx, name in enumerate(names):
        ax = axes[idx // ncols][idx % ncols]
        r = residuals[name]
        inside = ((r["predicted"] <= x_hi) & (r["residual"].abs() <= y_lim))
        n_off = int((~inside).sum())

        ax.scatter(r.loc[inside, "predicted"], r.loc[inside, "residual"],
                   s=10, alpha=0.45, edgecolor="none",
                   color=sns.color_palette(PALETTE)[idx % 10])
        ax.axhline(0, color="black", lw=1.1)
        ax.set_xlim(0, x_hi)
        ax.set_ylim(-y_lim, y_lim)

        rmse = float(np.sqrt(np.mean(r["residual"] ** 2)))
        mae = float(r["residual"].abs().mean())
        ax.set_title(f"{name}\nRMSE {format_inr(rmse)}  ·  MAE {format_inr(mae)}",
                     fontsize=10)
        ax.set_xlabel("Predicted rent")
        ax.set_ylabel("Residual (actual − predicted)")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(mticker.MaxNLocator(nbins=5))
        _inr_axis(ax, "x"); _inr_axis(ax, "y")
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        if n_off:
            ax.annotate(f"{n_off} of {len(r)} points off-scale",
                        xy=(0.98, 0.04), xycoords="axes fraction",
                        ha="right", fontsize=8, color="crimson",
                        bbox=dict(boxstyle="round", fc="white", ec="crimson",
                                  alpha=0.85))

    for idx in range(len(names), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Residuals vs predicted rent — all models (held-out test set)\n"
                 "Shared axes set from the actual rent distribution; "
                 "off-scale points are counted per panel",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return save(fig, "residuals_all_models.png")


# --------------------------------------------------------------------------- #
# Section 9 -- the outlier experiment
# --------------------------------------------------------------------------- #
def plot_outlier_experiment(exp: pd.DataFrame) -> str:
    """Grouped bars of test RMSE and MAE for each model under each setting."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    for ax, metric, label in [(axes[0], "test_rmse", "Test RMSE"),
                              (axes[1], "test_mae", "Test MAE")]:
        pivot = exp.pivot(index="model", columns="setting", values=metric)
        pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
        pivot.plot(kind="bar", ax=ax, width=0.78,
                   color=sns.color_palette(PALETTE)[:pivot.shape[1]])
        # Log scale for the same reason as the RMSE chart: one unregularised
        # model can be an order of magnitude worse and would flatten the rest.
        ax.set_yscale("log")
        ax.set_title(f"{label} by model and target/outlier setting (log scale)")
        ax.set_ylabel(f"{label} (log scale)")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        _inr_axis(ax, "y")
        ax.legend(title="Setting", fontsize=8, title_fontsize=9)
    fig.suptitle("Section 9 -- effect of the log transform and outlier treatment\n"
                 "(every bar is scored on the SAME untreated held-out test set)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return save(fig, "outlier_experiment.png")


def plot_premium_tradeoff(tradeoff: pd.DataFrame) -> str:
    """MAE within rent bands, showing what outlier treatment costs at the top."""
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot = tradeoff.pivot(index="rent_band", columns="setting", values="mae")
    pivot.plot(kind="bar", ax=ax, width=0.78,
               color=sns.color_palette(PALETTE)[:pivot.shape[1]])
    ax.set_title("The outlier-treatment trade-off\n"
                 "Treatment lowers error on ordinary listings but makes "
                 "premium-property predictions worse")
    ax.set_ylabel("MAE within the rent band")
    ax.set_xlabel("Actual rent band (held-out test set)")
    ax.tick_params(axis="x", rotation=15)
    _inr_axis(ax, "y")
    ax.legend(title="Setting", fontsize=8, title_fontsize=9)
    fig.tight_layout()
    return save(fig, "outlier_premium_tradeoff.png")


# --------------------------------------------------------------------------- #
# Section 10 -- deep residual analysis of the best model
# --------------------------------------------------------------------------- #
def plot_best_model_residuals(resid: pd.DataFrame, model_name: str) -> str:
    """Residuals vs predicted, histogram, Q-Q plot and residual vs actual."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

    ax = axes[0][0]
    ax.scatter(resid["predicted"], resid["residual"], s=12, alpha=0.45,
               edgecolor="none")
    ax.axhline(0, color="black", lw=1.1)
    ax.set_title("Residuals vs predicted\n"
                 "(a widening funnel means error grows with rent)")
    ax.set_xlabel("Predicted rent"); ax.set_ylabel("Residual")
    _inr_axis(ax, "x"); _inr_axis(ax, "y")

    ax = axes[0][1]
    ax.hist(resid["residual"], bins=70, edgecolor="white",
            color=sns.color_palette(PALETTE)[2])
    ax.axvline(0, color="black", lw=1.1)
    ax.axvline(resid["residual"].mean(), color="crimson", ls="--", lw=1.3,
               label=f"mean = {format_inr(resid['residual'].mean())}")
    ax.set_title(f"Histogram of residuals\n"
                 f"(skew = {resid['residual'].skew():.2f}, should be ~bell-shaped "
                 f"around 0)")
    ax.set_xlabel("Residual"); ax.set_ylabel("Count")
    _inr_axis(ax, "x"); ax.legend(fontsize=8)

    ax = axes[1][0]
    stats.probplot(resid["residual"], dist="norm", plot=ax)
    ax.set_title("Q-Q plot of residuals\n"
                 "(departures at the tails = non-normal extreme errors)")
    ax.get_lines()[0].set(markersize=3.5, alpha=0.6)
    ax.get_lines()[1].set(color="crimson", lw=1.4)

    ax = axes[1][1]
    ax.scatter(resid["actual"], resid["pct_residual"], s=12, alpha=0.45,
               edgecolor="none", color=sns.color_palette(PALETTE)[3])
    ax.axhline(0, color="black", lw=1.1)
    ax.set_xscale("log")
    ax.set_title("Percentage error vs actual rent\n"
                 "(shows whether cheap or expensive flats are predicted worse)")
    ax.set_xlabel("Actual rent (log scale)")
    ax.set_ylabel("Error as % of actual rent")
    _inr_axis(ax, "x")

    fig.suptitle(f"Section 10 -- residual diagnostics for the selected model: "
                 f"{model_name}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return save(fig, "best_model_residual_diagnostics.png")


def plot_residuals_by_city(resid: pd.DataFrame, model_name: str) -> str:
    """Residual distribution per city -- which cities predict worst."""
    order = [c for c in CITY_ORDER if c in set(resid["city"])]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(data=resid, x="city", y="residual", order=order,
                hue="city", legend=False, ax=axes[0], fliersize=2)
    axes[0].axhline(0, color="black", lw=1.1)
    axes[0].set_title("Residuals by city (rupees)")
    axes[0].set_xlabel(""); axes[0].set_ylabel("Residual")
    _inr_axis(axes[0], "y")

    sns.boxplot(data=resid, x="city", y="pct_residual", order=order,
                hue="city", legend=False, ax=axes[1], fliersize=2)
    axes[1].axhline(0, color="black", lw=1.1)
    axes[1].set_title("Residuals by city (% of actual rent)\n"
                      "the fair comparison, since rent scales differ ~5x")
    axes[1].set_xlabel(""); axes[1].set_ylabel("Error as % of actual rent")

    fig.suptitle(f"Residuals grouped by city -- {model_name}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return save(fig, "best_model_residuals_by_city.png")


# --------------------------------------------------------------------------- #
# Sections 11 and 13
# --------------------------------------------------------------------------- #
def plot_feature_importance(tree_imp: pd.DataFrame, perm_imp: pd.DataFrame,
                            top_n: int = 18) -> str:
    """Tree-based importance next to permutation importance."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    t = tree_imp.head(top_n).iloc[::-1]
    axes[0].barh(t["feature"], t["importance"],
                 color=sns.color_palette(PALETTE)[0])
    axes[0].set_title("Tree-based (impurity) importance\n"
                      "fast, but biased towards high-cardinality features")
    axes[0].set_xlabel("Normalised importance")

    p = perm_imp.head(top_n).iloc[::-1]
    axes[1].barh(p["feature"], p["importance_mean"],
                 xerr=p["importance_std"], capsize=3,
                 color=sns.color_palette(PALETTE)[2])
    axes[1].set_title("Permutation importance (more reliable)\n"
                      "rise in test MAE when the column is shuffled")
    axes[1].set_xlabel("Mean increase in test MAE (rupees)")
    _inr_axis(axes[1], "x")

    fig.suptitle("Section 11 -- feature importance", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    return save(fig, "feature_importance.png")


def plot_ablation(ablation: pd.DataFrame) -> str:
    """RMSE across the three locality-ablation models."""
    fig, ax = plt.subplots(figsize=(9.5, 5))
    d = ablation.copy()
    bars = ax.bar(d["label"], d["test_rmse"],
                  color=sns.color_palette(PALETTE)[:len(d)])
    ax.bar_label(bars, fmt=lambda v: format_inr(v), padding=3, fontsize=9)
    for xi, (r2, mae) in enumerate(zip(d["test_r2"], d["test_mae"])):
        ax.annotate(f"R² = {r2:.3f}\nMAE = {format_inr(mae)}",
                    (xi, 0), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=8, color="white", fontweight="bold")
    _inr_axis(ax, "y")
    ax.set_ylabel("Test RMSE")
    ax.set_title("Section 11 -- does locality dominate?\n"
                 "Same model and hyperparameters, three feature sets")
    ax.tick_params(axis="x", rotation=10)
    fig.tight_layout()
    return save(fig, "locality_ablation.png")


def plot_per_city_performance(per_city: pd.DataFrame) -> str:
    """Per-city error, both absolute and relative to each city's mean rent."""
    d = per_city.sort_values("rmse_pct_of_mean")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    b = axes[0].bar(d["City"], d["rmse"], color=sns.color_palette(PALETTE)[0])
    axes[0].bar_label(b, fmt=lambda v: format_inr(v), fontsize=8, padding=2,
                      rotation=90)
    axes[0].set_title("RMSE by city (absolute)")
    _inr_axis(axes[0], "y")

    b = axes[1].bar(d["City"], d["rmse_pct_of_mean"],
                    color=sns.color_palette(PALETTE)[2])
    axes[1].bar_label(b, fmt="%.0f%%", fontsize=8, padding=2)
    axes[1].set_title("RMSE as % of the city's mean rent\n(the fair comparison)")
    axes[1].set_ylabel("%")

    b = axes[2].bar(d["City"], d["r2"], color=sns.color_palette(PALETTE)[3])
    axes[2].bar_label(b, fmt="%.3f", fontsize=8, padding=2)
    axes[2].set_title("R² by city")

    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        ax.set_xlabel("")

    fig.suptitle("Section 13 -- per-city performance of the selected model",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return save(fig, "per_city_performance.png")


def plot_prediction_interval_check(actual, low, high, model_name: str) -> str:
    """Empirical coverage of the 80% prediction interval on the test set."""
    actual = np.asarray(actual, dtype="float64")
    low = np.asarray(low, dtype="float64")
    high = np.asarray(high, dtype="float64")
    inside = (actual >= low) & (actual <= high)
    order = np.argsort(actual)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    idx = np.arange(len(actual))
    ax.fill_between(idx, low[order], high[order], alpha=0.3,
                    label="80% prediction interval")
    ax.plot(idx, actual[order], lw=0.9, color="black", label="Actual rent")
    ax.set_yscale("log")
    _inr_axis(ax, "y")
    ax.set_title(f"Prediction interval vs actual rent\n"
                 f"empirical coverage = {inside.mean():.1%} (target 80%)")
    ax.set_xlabel("Test listings, sorted by actual rent")
    ax.set_ylabel("Monthly rent (log scale)")
    ax.legend(fontsize=8)

    width = (high - low) / np.clip(actual, 1, None) * 100
    axes[1].hist(width[np.isfinite(width)], bins=60, edgecolor="white",
                 color=sns.color_palette(PALETTE)[2])
    axes[1].set_title("Interval width as % of actual rent\n"
                      f"median = {np.nanmedian(width):.0f}%")
    axes[1].set_xlabel("(high − low) / actual  [%]")
    axes[1].set_ylabel("Count")

    fig.suptitle(f"Section 14 -- 80% prediction interval from the quantile "
                 f"models ({model_name})", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return save(fig, "prediction_interval_coverage.png")
