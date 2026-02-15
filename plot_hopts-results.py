import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RUN_DIR = "/home/provido/provido/luna_final/LUNA-main/run_histories"
OUTDIR  = "/home/provido/provido/luna_final/LUNA-main/out_data"
os.makedirs(OUTDIR, exist_ok=True)

# --------------------------
# Load all epoch-history CSVs
# --------------------------
paths = sorted(glob.glob(os.path.join(RUN_DIR, "*_epoch_history.csv")))
assert len(paths) > 0, f"No csvs found in {RUN_DIR}"

runs = []
for p in paths:
    df = pd.read_csv(p)
    df["run_file"] = os.path.basename(p)
    # run id from filename prefix: run12_...
    df["run_id"] = df["run_file"].str.split("_").str[0]
    runs.append(df)

all_epochs = pd.concat(runs, ignore_index=True)

# --------------------------
# Per-run summary table
# --------------------------
hp_cols = ["embed_dim","lr","weight_decay","dropout","lambda_val","supcon_temp","batch_size"]

def summarize_run(d):
    # best val_acc epoch
    i_best_acc = d["val_acc"].idxmax()
    # min val_loss epoch
    i_min_loss = d["val_loss"].idxmin()

    out = {c: d.iloc[0][c] for c in hp_cols}
    out["run_id"] = d.iloc[0]["run_id"]
    out["run_file"] = d.iloc[0]["run_file"]

    out["best_val_acc"] = float(d.loc[i_best_acc, "val_acc"])
    out["best_val_acc_epoch"] = int(d.loc[i_best_acc, "epoch"])
    out["val_loss_at_best_acc"] = float(d.loc[i_best_acc, "val_loss"])

    out["min_val_loss"] = float(d.loc[i_min_loss, "val_loss"])
    out["min_val_loss_epoch"] = int(d.loc[i_min_loss, "epoch"])
    out["val_acc_at_min_loss"] = float(d.loc[i_min_loss, "val_acc"])

    out["last_val_acc"] = float(d.iloc[-1]["val_acc"])
    out["last_val_loss"] = float(d.iloc[-1]["val_loss"])
    return pd.Series(out)

summary = all_epochs.groupby(["run_id","run_file"], as_index=False).apply(
    lambda g: summarize_run(g.reset_index(drop=True))
).reset_index(drop=True)

summary = summary.sort_values("best_val_acc", ascending=False)
summary.to_csv(os.path.join(OUTDIR, "run_summary.csv"), index=False)
print("Wrote:", os.path.join(OUTDIR, "run_summary.csv"))
print(summary.head(10)[["run_id","best_val_acc","best_val_acc_epoch","min_val_loss","min_val_loss_epoch"] + hp_cols])

# --------------------------
# Plot 3: Hyperparam sensitivity (best_val_acc)
# --------------------------
for col in ["embed_dim","batch_size","dropout","lambda_val","supcon_temp","weight_decay"]:
    plt.figure(figsize=(7,4))
    sns.boxplot(data=summary, x=col, y="best_val_acc")
    sns.stripplot(data=summary, x=col, y="best_val_acc", color="black", alpha=0.35, size=3)
    plt.title(f"Best val_acc by {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"best_val_acc_by_{col}.png"), dpi=200)
    plt.close()

# lr is continuous-ish: scatter with log x-axis
plt.figure(figsize=(7,4))
sns.scatterplot(data=summary, x="lr", y="best_val_acc", hue="embed_dim", style="batch_size", s=70, alpha=0.9)
plt.xscale("log")
plt.title("Best val_acc vs lr")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "best_val_acc_vs_lr.png"), dpi=200)
plt.close()

# --------------------------
# Plot 4: Heatmap (lambda_val x lr) showing mean best_val_acc
# --------------------------
# (bin lr into strings so heatmap axes are readable)
tmp = summary.copy()
tmp["lr_str"] = tmp["lr"].astype(str)
pivot = tmp.pivot_table(index="lambda_val", columns="lr_str", values="best_val_acc", aggfunc="mean")
plt.figure(figsize=(10,4))
sns.heatmap(pivot, cmap="viridis", annot=False)
plt.title("Mean best_val_acc heatmap: lambda_val (rows) x lr (cols)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "heat_lambda_lr_best_val_acc.png"), dpi=200)
plt.close()

print("Saved plots to:", OUTDIR)



###
#Plot 5
###
# --------------------------
# Plot 5: Accuracy vs epoch for many runs
#   - x: epoch
#   - y: accuracy
#   - color: chosen hyperparam
#   - linestyle: chosen hyperparam
#   - train lines alpha=0.5, val lines alpha=1.0
# --------------------------
COLOR_PARAM = "lambda_val"          # change to: "embed_dim", "batch_size", "lambda_val", ...
LINESTYLE_PARAM = "supcon_temp" # change to: "embed_dim", "batch_size", "lambda_val", ...

# Use top N runs to avoid an unreadable plot (increase if you want)
N_RUNS = min(30, len(summary))
chosen_runs = summary.head(N_RUNS)["run_file"].tolist()
dplot = all_epochs[all_epochs["run_file"].isin(chosen_runs)].copy()
dplot = dplot.sort_values(["run_file", "epoch"])

# Build a stable color map for COLOR_PARAM values
color_vals = sorted(dplot[COLOR_PARAM].unique().tolist())
palette = sns.color_palette("tab10", n_colors=max(10, len(color_vals)))
color_map = {v: palette[i % len(palette)] for i, v in enumerate(color_vals)}

# Build a stable linestyle map for LINESTYLE_PARAM values
ls_cycle = ["solid", "dashed", "dotted", "dashdot"]
ls_vals = sorted(dplot[LINESTYLE_PARAM].unique().tolist())
ls_map = {v: ls_cycle[i % len(ls_cycle)] for i, v in enumerate(ls_vals)}

plt.figure(figsize=(10,7))

for run_file, g in dplot.groupby("run_file"):
    g = g.sort_values("epoch")
    cval = g.iloc[0][COLOR_PARAM]
    lval = g.iloc[0][LINESTYLE_PARAM]
    color = color_map[cval]
    ls = ls_map[lval]

    # Train acc (faded)
    plt.plot(g["epoch"], g["train_acc"], color=color, linestyle=ls, alpha=0.25, linewidth=2)

    # Val acc (strong)
    plt.plot(g["epoch"], g["val_acc"],   color=color, linestyle=ls, alpha=1.0, linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"Train (alpha=0.25) vs Val (alpha=1.0) accuracy\nColor={COLOR_PARAM}, Linestyle={LINESTYLE_PARAM} (top {N_RUNS} runs)")

# Legends: one for colors (COLOR_PARAM), one for linestyles (LINESTYLE_PARAM)
from matplotlib.lines import Line2D
color_handles = [Line2D([0], [0], color=color_map[v], lw=3, label=f"{COLOR_PARAM}={v}") for v in color_vals]
ls_handles = [Line2D([0], [0], color="black", lw=3, linestyle=ls_map[v], label=f"{LINESTYLE_PARAM}={v}") for v in ls_vals]
train_val_handles = [
    Line2D([0],[0], color="gray", lw=3, alpha=0.25, label="train_acc"),
    Line2D([0],[0], color="gray", lw=3, alpha=1.0, label="val_acc"),
]

leg1 = plt.legend(handles=color_handles, title="Color", fontsize=8, title_fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0))
plt.gca().add_artist(leg1)
leg2 = plt.legend(handles=ls_handles, title="Linestyle", fontsize=8, title_fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 0.55))
plt.gca().add_artist(leg2)
plt.legend(handles=train_val_handles, title="Metric", fontsize=8, title_fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 0.25))

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, f"acc_curves_color-{COLOR_PARAM}_ls-{LINESTYLE_PARAM}.png"), dpi=200)
plt.close()
