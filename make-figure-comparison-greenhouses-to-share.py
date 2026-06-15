
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from matplotlib.ticker import MultipleLocator, ScalarFormatter

# ============================================================
# Input file
# ============================================================

file_path = "C:/slugs_long_all.csv" #ADDRESS FOR THE LONG DATA FOR THE SLUGS

output_dir = "C:/slug_figures" #ADDRESS FOR THE OUTPUT FOR THE GENERATED FIGURE
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# Figure formatting
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# ============================================================
# Load data
# ============================================================

df = pd.read_csv(file_path)

site_col = "Unnamed: 0"
trap_col = "Trap"
treatment_col = "Treatment"
date_col = "Date"
count_col = "Number of slugs"

df[date_col] = pd.to_datetime(df[date_col])
df[site_col] = df[site_col].astype(str).str.strip()
df[treatment_col] = df[treatment_col].astype(str).str.strip()

df["Site"] = df[site_col].str.strip()

df.loc[df["Site"].str.contains("Alta", case=False, na=False), "Site"] = "Alta Nursery"
df.loc[df["Site"].str.contains("Altman", case=False, na=False), "Site"] = "Altman Nursery"

df["Treatment"] = df[treatment_col].str.title()

# ============================================================
# Assign categories using trap numbers
# ============================================================

def make_category(row):
    site = row["Site"]
    trap = int(row[trap_col])

    if site == "Altman Nursery":
        if 21 <= trap <= 25:
            return "A006\nSouth"
        elif 26 <= trap <= 30:
            return "A006\nNorth"
        elif 31 <= trap <= 35:
            return "A007\nSouth"
        elif 36 <= trap <= 40:
            return "A007\nNorth"
        else:
            return "Altman\nUnknown"

    if site == "Alta Nursery":
        if 1 <= trap <= 5:
            return "Greenhouse\nInside"
        elif 6 <= trap <= 10:
            return "Greenhouse\nEdge"
        elif 11 <= trap <= 15:
            return "Outdoors\nInside"
        elif 16 <= trap <= 20:
            return "Outdoors\nEdge"
        else:
            return "Alta\nUnknown"

    return "Other"

df["Category"] = df.apply(make_category, axis=1)

category_order = [
    "A006\nSouth",
    "A006\nNorth",
    "A007\nSouth",
    "A007\nNorth",
    "Greenhouse\nInside",
    "Greenhouse\nEdge",
    "Outdoors\nInside",
    "Outdoors\nEdge"
]

df = df[df["Category"].isin(category_order)].copy()

# ============================================================
# Response variable
# ============================================================

df["Slugs per day"] = df[count_col]

# ============================================================
# Summary statistics
# ============================================================

summary = (
    df.groupby(["Category", "Treatment"], as_index=False)["Slugs per day"]
    .agg(mean="mean", sd="std", n="count", total="sum")
)

treatments = ["Water", "Yeast"]

print("\nSummary by category and treatment:")
print(summary.to_string(index=False))

print("\nTrap number check:")
print(
    df.groupby(["Site", "Category"])[trap_col]
    .agg(min_trap="min", max_trap="max", n_rows="count")
    .reset_index()
    .to_string(index=False)
)

# ============================================================
# Paired Wilcoxon tests by category
# ============================================================

p_values = {}

for category in category_order:
    sub = df[df["Category"] == category].copy()

    paired = sub.pivot_table(
        index=[trap_col, date_col],
        columns="Treatment",
        values="Slugs per day",
        aggfunc="sum"
    ).dropna()

    if "Water" in paired.columns and "Yeast" in paired.columns:
        difference = paired["Yeast"] - paired["Water"]

        if len(paired) > 0 and difference.abs().sum() > 0:
            try:
                stat, p = wilcoxon(
                    paired["Yeast"],
                    paired["Water"],
                    alternative="greater",
                    zero_method="wilcox"
                )
                p_values[category] = p
            except ValueError:
                p_values[category] = np.nan
        else:
            p_values[category] = np.nan
    else:
        p_values[category] = np.nan

print("\nPaired Wilcoxon p values:")
for category, p in p_values.items():
    print(f"{category.replace(chr(10), ' ')}: {p}")

# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(11.5, 6.2))

x = np.arange(len(category_order))
bar_width = 0.34

viridis = plt.cm.viridis

colors = {
    "Water": viridis(0.25),
    "Yeast": viridis(0.75)
}

for i, treatment in enumerate(treatments):
    values = []
    errors = []

    for category in category_order:
        row = summary[
            (summary["Category"] == category) &
            (summary["Treatment"] == treatment)
        ]

        if row.empty:
            values.append(0)
            errors.append(0)
        else:
            values.append(row["mean"].iloc[0])
            errors.append(row["sd"].iloc[0])

    offset = (i - 0.5) * bar_width

    ax.bar(
        x + offset,
        values,
        width=bar_width,
        yerr=errors,
        capsize=4,
        color=colors[treatment],
        edgecolor="black",
        linewidth=1,
        label="Water" if treatment == "Water" else "Yeast"
    )

# ============================================================
# Axis formatting
# ============================================================

max_value = (summary["mean"] + summary["sd"].fillna(0)).max()
y_top = max(10, math.ceil(max_value + 3))

ax.set_yscale("linear")
ax.set_ylim(0, y_top)

ax.yaxis.set_major_locator(MultipleLocator(5))

formatter = ScalarFormatter(useOffset=False)
formatter.set_scientific(False)
ax.yaxis.set_major_formatter(formatter)

ax.ticklabel_format(axis="y", style="plain", useOffset=False)

ax.set_ylabel("Slugs per day (mean ± SD)")
ax.set_xticks(x)
ax.set_xticklabels(category_order)

ax.grid(axis="y", linestyle=":", alpha=0.5)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ============================================================
# Site divider and site labels
# ============================================================

divider_x = 3.5
ax.axvline(divider_x, color="0.65", linestyle="--", linewidth=1.2)

site_label_y = y_top * 0.96

ax.text(
    1.5,
    site_label_y,
    "Altman Nursery",
    ha="right",
    va="top",
    fontsize=14,
    fontstyle="italic",
    color="0.35"
)

ax.text(
    5.5,
    site_label_y,
    "Alta Nursery",
    ha="right",
    va="top",
    fontsize=14,
    fontstyle="italic",
    color="0.35"
)

# ============================================================
# Significance asterisks
# ============================================================

for category_index, category in enumerate(category_order):
    p = p_values.get(category, np.nan)

    if not np.isnan(p) and p < 0.05:
        cat_summary = summary[summary["Category"] == category]
        ymax = (cat_summary["mean"] + cat_summary["sd"].fillna(0)).max()

        if np.isfinite(ymax) and ymax > 0:
            ax.text(
                category_index + bar_width / 2,
                min(ymax + 0.9, y_top * 0.88),
                "*",
                ha="center",
                va="bottom",
                fontsize=14,
                color="black"
            )

# ============================================================
# Legend and Wilcoxon note
# ============================================================

ax.legend(
    frameon=False,
    loc="upper left",
    ncol=2,
    bbox_to_anchor=(0.01, 0.91)
)

ax.text(
    0.98,
    0.96,
    "* paired Wilcoxon p < 0.05",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=14,
    fontstyle="italic",
    color="0.35"
)

# ============================================================
# Save figure
# ============================================================

plt.tight_layout()

png_path = os.path.join(output_dir, "figure_yeast_vs_water_sites2.png")
pdf_path = os.path.join(output_dir, "figure_yeast_vs_water_sites2.pdf")

plt.savefig(png_path, dpi=300)
plt.savefig(pdf_path)
plt.close()

print("\nSaved:")
print(png_path)
print(pdf_path)

print("\nDetected categories:")
print(df.groupby(["Site", "Category"]).size())
