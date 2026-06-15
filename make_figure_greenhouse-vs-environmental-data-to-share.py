
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# Input files
# ============================================================

slug_file = "C:/slugs_long.csv" #original data 
env_file = "C:/environmental-variables-all.csv" #CIMIS data, environmental data

output_dir = "C:/slug_environment_combined_figure" #output file

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# Input files
# ============================================================

# Place this script in the same folder as the two CSV files, or replace these
# with full paths on your computer.
slug_file = "C:/slugs_long_all.csv"
env_file = "C:/slugs-environmental-variables-all.csv"
#"C:\Users\andre\Downloads\slugs-environmental-variables-all.csv"
#"C:\Users\andre\Downloads\slugs_long_Alta_all.csv"
# Output folder
base_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
slug_path = Path(slug_file)
env_path = Path(env_file)

if not slug_path.is_absolute():
    slug_path = base_dir / slug_path

if not env_path.is_absolute():
    env_path = base_dir / env_path

output_dir = "C:/slugs-june"
#"C:\Users\andre\Downloads\slugs-june"
output_dir = base_dir 
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# Global formatting
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 10
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# ============================================================
# Load data
# ============================================================

slugs = pd.read_csv(slug_path)
env = pd.read_csv(env_path)

# Remove empty rows from the CIMIS export
env = env.dropna(subset=["Date"]).copy()

slugs["Date"] = pd.to_datetime(slugs["Date"])
env["Date"] = pd.to_datetime(env["Date"])

slug_col = "Number of slugs"
date_col = "Date"
treatment_col = "Treatment"
incubation_col = "Incubation days"

# Basic check for the updated Alta dataset
n_dates = slugs[date_col].nunique()
expected_obs = n_dates * slugs["Trap"].nunique() * slugs[treatment_col].nunique()

print(f"Slug observations: {len(slugs)}")
print(f"Collection dates detected: {n_dates}")
print("Collection dates:")
for d in sorted(slugs[date_col].unique()):
    print(f"  {pd.Timestamp(d).date()}")

if n_dates < 8:
    print("Warning: fewer than eight Alta collection dates were detected.")
else:
    print("Updated Alta dataset detected with eight collection dates.")

# ============================================================
# Environmental variables included in Panel A
# ============================================================

env_vars = [
    "Precip (mm)",
    "Sol Rad (W/sq.m)",
    "Max Air Temp (C)",
    "Min Air Temp (C)",
    "Avg Air Temp (C)",
    "Max Rel Hum (%)",
    "Min Rel Hum (%)",
    "Avg Rel Hum (%)",
    "Avg Wind Speed (m/s)",
    "Wind Run (km)",
    "Avg Soil Temp (C)"
]

scatter_vars = [
    "Wind Run (km)",
    "Avg Wind Speed (m/s)"
]

missing_env_cols = [x for x in sorted(set(env_vars + scatter_vars)) if x not in env.columns]
if missing_env_cols:
    raise ValueError(f"Missing environmental columns: {missing_env_cols}")

def clean_label(label):
    label = label.replace("Avg ", "Average ")
    label = label.replace("Max ", "Maximum ")
    label = label.replace("Min ", "Minimum ")
    label = label.replace("Rel Hum", "Relative humidity")
    label = label.replace("Temp", "temperature")
    label = label.replace("Precip", "Precipitation")
    label = label.replace("Sol Rad", "Solar radiation")
    label = label.replace("sq.m", "m²")
    label = label.replace("(C)", "(°C)")
    return label

def short_label(label):
    replacements = {
        "Precip (mm)": "Precipitation",
        "Sol Rad (W/sq.m)": "Solar radiation",
        "Max Air Temp (C)": "Maximum air temp. (°C)",
        "Min Air Temp (C)": "Minimum air temp. (°C)",
        "Avg Air Temp (C)": "Average air temp. (°C)",
        "Max Rel Hum (%)": "Maximum RH (%)",
        "Min Rel Hum (%)": "Minimum RH (%)",
        "Avg Rel Hum (%)": "Average RH (%)",
        "Avg Wind Speed (m/s)": "Average wind speed (m/s)",
        "Wind Run (km)": "Wind run (km)",
        "Avg Soil Temp (C)": "Average soil temp. (°C)"
    }
    return replacements.get(label, clean_label(label))

# ============================================================
# Build environmental exposure windows
# ============================================================

def build_environment_windows(slug_df, env_df):
    collection_info = (
        slug_df.groupby(date_col, as_index=False)[incubation_col]
        .agg(lambda x: int(pd.Series(x).dropna().mode().iloc[0]))
    )

    env_window_rows = []

    for _, row in collection_info.iterrows():
        collection_date = row[date_col]
        incubation_days = int(row[incubation_col])

        # A 5 day incubation uses the 4 calendar days before collection.
        # A 4 day incubation uses the 3 calendar days before collection.
        # The collection day itself is excluded.
        prior_dates = [
            collection_date - pd.Timedelta(days=i)
            for i in range(1, incubation_days)
        ]

        env_window = env_df[env_df[date_col].isin(prior_dates)].copy()

        if env_window.empty:
            print(f"Warning: no environmental rows found for {collection_date.date()}")
            continue

        expected_dates = len(prior_dates)
        observed_dates = env_window[date_col].nunique()
        observed_stations = env_window["Stn Name"].nunique(dropna=True)

        if observed_dates < expected_dates:
            print(
                f"Warning: incomplete environmental window for {collection_date.date()}: "
                f"{observed_dates} of {expected_dates} days found."
            )

        vars_to_summarize = sorted(set(env_vars + scatter_vars))

        summarized = env_window[vars_to_summarize].mean(numeric_only=True).to_dict()
        summarized[date_col] = collection_date
        summarized[incubation_col] = incubation_days
        summarized["n_weather_rows"] = len(env_window)
        summarized["n_weather_dates"] = observed_dates
        summarized["n_weather_stations"] = observed_stations
        summarized["weather_window_start"] = min(prior_dates)
        summarized["weather_window_end"] = max(prior_dates)

        env_window_rows.append(summarized)

    return pd.DataFrame(env_window_rows)

env_summary = build_environment_windows(slugs, env)

# ============================================================
# Slug summaries by collection date
# ============================================================

def summarize_slugs_by_date(slug_df):
    total_by_date = (
        slug_df.groupby(date_col, as_index=False)[slug_col]
        .mean()
        .rename(columns={slug_col: "Total slugs per trap"})
    )

    yeast_by_date = (
        slug_df[slug_df[treatment_col].str.lower() == "yeast"]
        .groupby(date_col, as_index=False)[slug_col]
        .mean()
        .rename(columns={slug_col: "Yeast slugs per trap"})
    )

    water_by_date = (
        slug_df[slug_df[treatment_col].str.lower() == "water"]
        .groupby(date_col, as_index=False)[slug_col]
        .mean()
        .rename(columns={slug_col: "Water slugs per trap"})
    )

    analysis_df = env_summary.merge(total_by_date, on=date_col, how="left")
    analysis_df = analysis_df.merge(yeast_by_date, on=date_col, how="left")
    analysis_df = analysis_df.merge(water_by_date, on=date_col, how="left")

    return analysis_df

analysis = summarize_slugs_by_date(slugs)

# ============================================================
# Benjamini Hochberg correction
# ============================================================

def benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan, dtype=float)

    valid = ~np.isnan(p_values)
    p = p_values[valid]
    n = len(p)

    if n == 0:
        return adjusted

    order = np.argsort(p)
    ranked = p[order]

    bh = ranked * n / np.arange(1, n + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0, 1)

    corrected = np.empty(n)
    corrected[order] = bh
    adjusted[valid] = corrected

    return adjusted

# ============================================================
# Correlation table
# ============================================================

outcome_vars = [
    "Total slugs per trap",
    "Yeast slugs per trap",
    "Water slugs per trap"
]

all_env_vars_for_stats = sorted(set(env_vars + scatter_vars))

def compute_correlations(analysis_df, label):
    correlation_rows = []

    for outcome in outcome_vars:
        for var in all_env_vars_for_stats:
            temp = analysis_df[[outcome, var]].dropna()

            if len(temp) >= 3 and temp[outcome].nunique() > 1 and temp[var].nunique() > 1:
                r, p = stats.pearsonr(temp[var], temp[outcome])
            else:
                r, p = np.nan, np.nan

            correlation_rows.append({
                "Analysis": label,
                "Outcome": outcome,
                "Environmental variable": var,
                "Pearson r": r,
                "p value": p
            })

    corr_df = pd.DataFrame(correlation_rows)
    corr_df["adjusted p value"] = np.nan

    for outcome in outcome_vars:
        mask = corr_df["Outcome"] == outcome
        corr_df.loc[mask, "adjusted p value"] = benjamini_hochberg(
            corr_df.loc[mask, "p value"].values
        )

    return corr_df

correlations = compute_correlations(analysis, "All Alta collection dates")

# Sensitivity output for the first seven dates, matching the previous analysis logic
first_seven_dates = sorted(slugs[date_col].unique())[:7]
analysis_first_seven = analysis[analysis[date_col].isin(first_seven_dates)].copy()
correlations_first_seven = compute_correlations(
    analysis_first_seven,
    "First seven Alta collection dates"
)

correlations_combined = pd.concat(
    [correlations, correlations_first_seven],
    ignore_index=True
)

correlations.to_csv(
    output_dir / "correlation_statistics_all_alta_dates.csv",
    index=False
)

correlations_first_seven.to_csv(
    output_dir / "correlation_statistics_first_seven_dates.csv",
    index=False
)

correlations_combined.to_csv(
    output_dir / "correlation_statistics_all_and_first_seven_dates.csv",
    index=False
)

analysis.to_csv(
    output_dir / "date_level_environment_slug_dataset_all_alta_dates.csv",
    index=False
)

# Print key updated wind results
print("\nKey yeast trap wind correlations using all Alta dates:")
for var in ["Wind Run (km)", "Avg Wind Speed (m/s)"]:
    row = correlations[
        (correlations["Outcome"] == "Yeast slugs per trap") &
        (correlations["Environmental variable"] == var)
    ].iloc[0]
    print(
        f"{var}: r = {row['Pearson r']:.3f}, "
        f"p = {row['p value']:.4f}, "
        f"adjusted p = {row['adjusted p value']:.4f}"
    )

# ============================================================
# Heat map data
# ============================================================

heatmap_matrix = correlations.pivot(
    index="Outcome",
    columns="Environmental variable",
    values="Pearson r"
)

heatmap_matrix = heatmap_matrix.loc[outcome_vars, env_vars]

heatmap_padj = correlations.pivot(
    index="Outcome",
    columns="Environmental variable",
    values="adjusted p value"
)

heatmap_padj = heatmap_padj.loc[outcome_vars, env_vars]

# ============================================================
# Scatterplot helper
# ============================================================

def add_scatter(ax, data, x, y, panel_label):
    temp = data[[x, y]].dropna()

    viridis = plt.cm.viridis
    point_color = viridis(0.72)
    line_color = viridis(0.28)

    ax.scatter(
        temp[x],
        temp[y],
        s=50,
        color=point_color,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9
    )

    slope, intercept, r_value, p_value, std_err = stats.linregress(temp[x], temp[y])
    x_line = np.linspace(temp[x].min(), temp[x].max(), 100)
    y_line = intercept + slope * x_line
    ax.plot(x_line, y_line, linewidth=1.8, color=line_color)

    stat_row = correlations[
        (correlations["Outcome"] == y) &
        (correlations["Environmental variable"] == x)
    ]

    r = stat_row["Pearson r"].iloc[0]
    p = stat_row["p value"].iloc[0]
    p_adj = stat_row["adjusted p value"].iloc[0]

    stats_text = (
        f"r = {r:.3f}\n"
        f"p = {p:.4f}\n"
        f"adjusted p = {p_adj:.4f}"
    )

    ax.text(
        0.96,
        0.05,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.30",
            facecolor="white",
            edgecolor="0.6",
            alpha=0.95
        )
    )

    ax.set_title(f"{panel_label}.", loc="left", pad=8)
    ax.set_xlabel(clean_label(x))
    ax.set_ylabel("Slugs per trap (yeast)")

    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ============================================================
# Generate figure
# ============================================================

fig = plt.figure(figsize=(7.2, 6.2))

gs = fig.add_gridspec(
    2,
    2,
    height_ratios=[0.95, 1.35],
    width_ratios=[1, 1],
    hspace=0.8,
    wspace=0.45
)

# Panel A
ax_heatmap = fig.add_subplot(gs[0, :])

im = ax_heatmap.imshow(
    heatmap_matrix.values,
    aspect="auto",
    vmin=-1,
    vmax=1,
    cmap="viridis"
)

ax_heatmap.set_title(
    "A.",
    loc="left",
    pad=8
)

ax_heatmap.set_xticks(np.arange(len(heatmap_matrix.columns)))
ax_heatmap.set_yticks(np.arange(len(heatmap_matrix.index)))

ax_heatmap.set_xticklabels(
    [short_label(x) for x in heatmap_matrix.columns],
    rotation=35,
    ha="right",
    rotation_mode="anchor"
)

ax_heatmap.set_yticklabels(
    ["Total", "Yeast", "Water"]
)

for i in range(heatmap_matrix.shape[0]):
    for j in range(heatmap_matrix.shape[1]):
        value = heatmap_matrix.values[i, j]
        padj = heatmap_padj.values[i, j]

        if not np.isnan(value):
            label = f"{value:.2f}"
            if not np.isnan(padj) and padj < 0.05:
                label += "*"

            text_color = "white" if value < 0.35 else "black"

            ax_heatmap.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=10
            )

ax_heatmap.tick_params(axis="x", length=0, pad=6)
ax_heatmap.tick_params(axis="y", length=0)

for spine in ax_heatmap.spines.values():
    spine.set_visible(False)

cbar = fig.colorbar(
    im,
    ax=ax_heatmap,
    fraction=0.03,
    pad=0.015
)

cbar.set_label("Pearson r")
cbar.ax.tick_params(labelsize=10)

# Panel B
ax_b = fig.add_subplot(gs[1, 0])
add_scatter(
    ax_b,
    analysis,
    "Wind Run (km)",
    "Yeast slugs per trap",
    "B"
)

# Panel C
ax_c = fig.add_subplot(gs[1, 1])
add_scatter(
    ax_c,
    analysis,
    "Avg Wind Speed (m/s)",
    "Yeast slugs per trap",
    "C"
)

# Save outputs
figure_png = output_dir / "figure_environmental_effects_slug_capture_updated_all_alta_dates.png"
figure_pdf = output_dir / "figure_environmental_effects_slug_capture_updated_all_alta_dates.pdf"

plt.savefig(figure_png, dpi=300, bbox_inches="tight")
plt.savefig(figure_pdf, bbox_inches="tight")
plt.close()

print("\nSaved figure files:")
print(figure_png)
print(figure_pdf)

print("\nSaved analysis tables:")
print(output_dir / "correlation_statistics_all_alta_dates.csv")
print(output_dir / "correlation_statistics_first_seven_dates.csv")
print(output_dir / "correlation_statistics_all_and_first_seven_dates.csv")
print(output_dir / "date_level_environment_slug_dataset_all_alta_dates.csv")
