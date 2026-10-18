import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COL = {"Standard": "#6b7280", "LBF": "#2563eb", "PLBF": "#dc2626"}


def style(ax, xl, yl, title):
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)


def main():
    # 1. FPR vs memory (log y-axis)
    d = pd.read_csv("results/exp1_fpr_vs_memory.csv")
    g = d.groupby(["filter", "bits_per_key"])[["fpr", "query_us"]].mean().reset_index()
    g["fpr"] = g["fpr"].clip(lower=1e-5)    # FPR of 0 cannot be drawn on a log axis
    fig, ax = plt.subplots(figsize=(6, 4))
    for f, sub in g.groupby("filter"):
        ax.plot(sub.bits_per_key, sub.fpr, "o-", label=f, color=COL[f])
    ax.set_yscale("log"); ax.legend()
    style(ax, "Total memory (bits per key)", "False-positive rate (log)",
          "FPR vs memory (same total memory for all filters)")
    fig.tight_layout(); fig.savefig("results/fig1_fpr_vs_memory.png", dpi=160)

    # 2. Classifier comparison
    c2 = pd.read_csv("results/exp2_classifiers.csv")
    c2 = c2[c2.bits_per_key == c2.bits_per_key.max()]
    c = c2.groupby("classifier").mean(numeric_only=True).reindex(["LR", "NB", "RF", "MLP"])
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    c["fpr"].plot.bar(ax=ax[0], color="#2563eb")
    style(ax[0], "", "FPR (no bar = model too big)", f"FPR at {int(c2.bits_per_key.max())} bits/key")
    (c["model_bits"] / 8 / 1024).plot.bar(ax=ax[1], color="#d97706")
    style(ax[1], "", "KiB", "Model size")
    c["query_us"].plot.bar(ax=ax[2], color="#059669")
    style(ax[2], "", "us / query", "Query latency")
    for a in ax:
        a.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig("results/fig2_classifiers.png", dpi=160)

    # 3. Partitions
    p = pd.read_csv("results/exp3_partitions.csv").groupby("partitions").mean(numeric_only=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].plot(p.index, p.fpr, "o-", color=COL["PLBF"])
    style(ax[0], "Partitions", "FPR", "FPR vs partitions")
    ax[1].plot(p.index, p.build_s, "o-", color="#7c3aed")
    style(ax[1], "Partitions", "seconds", "Build time vs partitions")
    fig.tight_layout(); fig.savefig("results/fig3_partitions.png", dpi=160)

    # 4. Distribution shift
    s = pd.read_csv("results/exp4_drift.csv").groupby(["filter", "setting"])["fpr"].mean().unstack()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    s.plot.bar(ax=ax, color=["#94a3b8", "#dc2626"])
    ax.tick_params(axis="x", rotation=0)
    style(ax, "", "FPR", "Effect of distribution shift on FPR")
    fig.tight_layout(); fig.savefig("results/fig4_drift.png", dpi=160)

    # 5. Your own measured version of Table I
    t = d.groupby(["filter", "bits_per_key"]).mean(numeric_only=True).reset_index()
    t = t[t.bits_per_key == t.bits_per_key.median()][
        ["filter", "memory_bits", "fpr", "false_neg", "query_us"]]
    t.to_csv("results/table_measured.csv", index=False)
    print("Measured table (middle memory budget):\n", t.round(5).to_string(index=False))
    print("Figures saved to results/")


if __name__ == "__main__":
    main()