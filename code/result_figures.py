"""Display functions for selected saved research estimates; no model fitting."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
TEAL = "#247A89"
ORANGE = "#C5703B"
GREY = "#AAB4BC"

def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "figure.facecolor": "white"})


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    return fig


def reported_associations(estimates, weights, path):
    style()
    labels = {"learning": "Everyday learning / interest\n(unweighted responses)",
              "low_sexuality_iat": "Implicit sexual-orientation attitudes\n(reversed signed IAT D)",
              "climate": "Climate-opinion composite", "patents": "Patent activity"}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained", gridspec_kw={"width_ratios": [1.2, 1]})
    for j, row in estimates.iterrows():
        ax.errorbar(row.estimate, j, xerr=[[row.estimate-row.ci_lower], [row.ci_upper-row.estimate]],
                    fmt="o", capsize=4, color=TEAL)
    ax.set(yticks=range(len(estimates)), yticklabels=[labels[k] for k in estimates.outcome],
           xlabel="Standardized coefficient (95% CI)", title="Reported adjusted associations")
    ax.invert_yaxis(); ax.axvline(0, color=GREY, ls="--")
    names = {"raw_unweighted_fixed": "Unweighted responses", "national_weighted": "National respondent weights"}
    for j, row in weights.iterrows():
        bx.errorbar(row.estimate, j, xerr=[[row.estimate-row.ci_lower], [row.ci_upper-row.estimate]],
                    fmt="o", capsize=4, color=[TEAL, ORANGE][j])
    bx.set(yticks=range(len(weights)), yticklabels=[names[k] for k in weights.specification],
           xlabel="Standardized coefficient (95% CI)", title="Learning depends on respondent aggregation", ylim=(1.8, -.8))
    bx.axvline(0, color=GREY, ls="--")
    fig.suptitle("Saved aggregate estimates; regressions are not refitted in this public walkthrough", fontsize=12)
    return save(fig, path)
