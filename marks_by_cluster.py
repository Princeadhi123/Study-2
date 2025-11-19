import sys
from pathlib import Path
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def _detect_id_col(df: pd.DataFrame) -> str:
    cand = [c for c in df.columns if c.lower() in {"idcode", "id", "studentid", "student_id"}]
    if cand:
        return cand[0]
    for c in df.columns:
        if c.lower().startswith("id"):
            return c
    raise ValueError("Could not find an ID column (expected something like 'IDCode' or 'ID').")


def _detect_subject_cols(df: pd.DataFrame) -> list:
    cols = []
    for c in df.columns:
        cl = c.lower().strip()
        if re.fullmatch(r"s\d+", cl):
            cols.append(c)
    return sorted(cols, key=lambda x: int(re.findall(r"\d+", x)[0])) if cols else cols


def _detect_total_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if "total" in c.lower() or c.lower() in {"sum", "overall", "grandtotal", "grand_total"}:
            return c
    return None


def _zmean(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean()
    sd = df.std(ddof=0).replace(0, np.nan)
    return (df - mu) / sd


def _save_boxplots(df: pd.DataFrame, cluster_col: str, value_cols: list, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for col in value_cols:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x=cluster_col, y=col, showfliers=False)
        sns.stripplot(data=df, x=cluster_col, y=col, size=2, color="black", alpha=0.18)
        plt.title(f"{col} by {cluster_col}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{col}_by_{cluster_col}.png", dpi=150)
        plt.close()


def _save_zmean_heatmap(df: pd.DataFrame, cluster_col: str, value_cols: list, out_path: Path, title: str):
    z = _zmean(df[value_cols])
    mat = z.join(df[cluster_col]).groupby(cluster_col)[value_cols].mean().sort_index()
    # Order columns by discriminativeness (avg |z| across clusters)
    if not mat.empty:
        col_order = mat.abs().mean(axis=0).sort_values(ascending=False).index.tolist()
        mat = mat[col_order]
    vmax = float(np.nanmax(np.abs(mat.values))) if mat.size else 0.0
    vmax = max(1.0, min(3.0, vmax))
    vmin = -vmax
    fig_w = max(10.0, 1.3 * len(mat.columns))
    fig_h = max(8.0, 1.2 * max(6, len(mat)))
    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.heatmap(
        mat,
        cmap="RdBu_r",
        center=0,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        cbar=True,
        cbar_kws={"label": "z-mean"},
        annot_kws={"size": 11},
        linewidths=0.6,
        linecolor="#f0f0f0",
    )
    ax.set_xlabel("Subject / Total")
    ax.set_ylabel("Cluster")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    # Add cluster counts to y labels
    counts = df[cluster_col].value_counts().sort_index()
    ylabels = []
    for k in mat.index.tolist():
        ylabels.append(f"{k} (n={int(counts.get(k, 0))})")
    ax.set_yticklabels(ylabels, rotation=0)
    # Improve annotation contrast
    for text in ax.texts:
        try:
            val = float(text.get_text())
        except Exception:
            continue
        text.set_color("white" if abs(val) > (0.6 * vmax) else "black")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _save_violin(df: pd.DataFrame, cluster_col: str, value_col: str, out_path: Path, title: str):
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=df, x=cluster_col, y=value_col, inner=None, cut=0)
    sns.stripplot(data=df, x=cluster_col, y=value_col, size=2, color="black", alpha=0.18)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _save_subject_radar_all_clusters(df: pd.DataFrame, cluster_col: str, subj_cols: list, out_path: Path):
    mu = df[subj_cols].mean()
    sd = df[subj_cols].std(ddof=0).replace(0, np.nan)
    z = (df[subj_cols] - mu) / sd
    zmean = z.join(df[cluster_col]).groupby(cluster_col)[subj_cols].mean().sort_index()
    cats = subj_cols
    N = len(cats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    plt.figure(figsize=(9, 7))
    ax = plt.subplot(111, polar=True)
    palette = sns.color_palette("tab10", n_colors=len(zmean))
    for i, (k, row) in enumerate(zmean.iterrows()):
        vals = row.values.astype(float)
        vals = np.clip(vals, -3.0, 3.0)
        vals = vals.tolist() + [vals[0]]
        ax.plot(angles, vals, color=palette[i], linewidth=2, label=str(k))
        ax.fill(angles, vals, color=palette[i], alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats)
    ax.set_yticks([-3, -1.5, 0, 1.5, 3])
    ax.set_ylim(-3, 3)
    ax.set_title("Subject z-mean radar (all clusters)")
    ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def _save_correlation_tables(
    dfa: pd.DataFrame,
    id_col: str,
    cluster_col: str,
    subj_cols: list,
    total_col: str | None,
    derived_path: Path,
    out_profiles: Path,
    out_figs: Path,
):
    out_profiles.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)
    if subj_cols:
        clusters_sorted = sorted(dfa[cluster_col].unique(), key=lambda x: int(x))
        corr_rows = {}
        for k in clusters_sorted:
            ind = (dfa[cluster_col] == k).astype(int)
            ser = dfa[subj_cols].apply(lambda col: col.corr(ind))
            ser.name = str(k)
            corr_rows[str(k)] = ser
        corr_df = pd.DataFrame(corr_rows)
        corr_df.to_csv(out_profiles / "corr_subjects_vs_cluster.csv")
        plt.figure(figsize=(1.2 * max(6, len(corr_df.columns)), 0.6 * max(6, len(corr_df.index))))
        ax = sns.heatmap(
            corr_df,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f",
            cbar=True,
            cbar_kws={"label": "corr"},
        )
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Subject")
        plt.title("Subject vs cluster correlation (Pearson)")
        plt.tight_layout()
        plt.savefig(out_figs / "corr_subjects_vs_cluster_heatmap.png", dpi=150)
        plt.close()

    if total_col and derived_path.exists():
        dfd = pd.read_csv(derived_path)
        id_derived = "IDCode" if "IDCode" in dfd.columns else _detect_id_col(dfd)
        dfm = dfa[[id_col, total_col]].merge(dfd, left_on=id_col, right_on=id_derived, how="inner")
        exclude = {id_derived}
        feat_cols = [c for c in dfm.columns if c not in exclude | {id_col, total_col}]
        pearson = dfm[feat_cols].corrwith(dfm[total_col], method="pearson").to_frame("pearson")
        spearman = dfm[feat_cols].corrwith(dfm[total_col], method="spearman").to_frame("spearman")
        corr2 = pearson.join(spearman)
        corr2.index.name = "feature"
        corr2.sort_values("pearson", ascending=False).to_csv(out_profiles / "corr_total_vs_features.csv")
        plt.figure(figsize=(1.2 * max(6, len(corr2.index)), 5))
        ax2 = sns.heatmap(
            corr2[["pearson"]].T,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f",
            cbar=True,
            cbar_kws={"label": "corr"},
        )
        ax2.set_xlabel("Feature")
        ax2.set_ylabel("TOTAL")
        plt.title("Correlation: TOTAL vs derived features (Pearson)")
        plt.tight_layout()
        plt.savefig(out_figs / "corr_total_vs_features_heatmap.png", dpi=150)
        plt.close()

def run(marks_path: Path):
    base = Path(__file__).parent
    clusters_path = base / "student_clusters.csv"
    derived_path = base / "derived_features.csv"
    dfc = pd.read_csv(clusters_path)
    cluster_col = "gmm_bic_best_label"
    if cluster_col not in dfc.columns:
        raise ValueError(f"Missing '{cluster_col}' in {clusters_path}")
    id_clusters = "IDCode" if "IDCode" in dfc.columns else _detect_id_col(dfc)

    try:
        dfx = pd.read_excel(marks_path)
    except Exception as e:
        raise RuntimeError(f"Failed reading Excel: {marks_path}\n{e}")

    id_excel = "IDCode" if "IDCode" in dfx.columns else _detect_id_col(dfx)
    subj_cols = _detect_subject_cols(dfx)
    total_col = _detect_total_col(dfx)

    if not subj_cols and total_col is None:
        raise ValueError("No subject columns like s1..s6 or a total column detected in the Excel file.")

    keep_cols = [id_excel] + subj_cols + ([total_col] if total_col else [])
    dfx = dfx[keep_cols].copy()
    dfx[id_excel] = dfx[id_excel].astype(str).str.strip()

    dfc[id_clusters] = dfc[id_clusters].astype(str)
    dfc[id_clusters] = dfc[id_clusters].fillna("")

    dfa = dfx.merge(dfc[[id_clusters, cluster_col]].rename(columns={id_clusters: id_excel}), on=id_excel, how="inner")

    out_profiles = base / "profiles"
    out_figs = base / "figures" / "marks_by_cluster"

    dfa.to_csv(base / "marks_with_clusters.csv", index=False)

    value_cols = subj_cols + ([total_col] if total_col else [])

    stats = dfa.groupby(cluster_col)[value_cols].agg(["count", "mean", "std", "median"]).sort_index()
    stats.to_csv(out_profiles / "marks_by_cluster_stats.csv")

    _save_boxplots(dfa, cluster_col, value_cols, out_figs / "boxplots")
    _save_zmean_heatmap(dfa, cluster_col, value_cols, out_figs / "subject_zmean_by_cluster.png", "Subject-wise z-mean by cluster")

    if total_col:
        _save_violin(dfa, cluster_col, total_col, out_figs / "total_violin.png", f"{total_col} by cluster")

    if subj_cols:
        _save_subject_radar_all_clusters(dfa, cluster_col, subj_cols, out_figs / "subject_radar_all_clusters.png")

    _save_correlation_tables(
        dfa=dfa,
        id_col=id_excel,
        cluster_col=cluster_col,
        subj_cols=subj_cols,
        total_col=total_col,
        derived_path=derived_path,
        out_profiles=out_profiles,
        out_figs=out_figs,
    )

    # Dropout risk assessment (unsupervised risk score)
    if total_col and derived_path.exists():
        dfd = pd.read_csv(derived_path)
        _compute_dropout_risk(
            dfa=dfa,
            dfd=dfd,
            id_col=id_excel,
            cluster_col=cluster_col,
            total_col=total_col,
            out_profiles=out_profiles,
            out_figs=out_figs,
        )

def _compute_dropout_risk(
    dfa: pd.DataFrame,
    dfd: pd.DataFrame,
    id_col: str,
    cluster_col: str,
    total_col: str,
    out_profiles: Path,
    out_figs: Path,
):
    # Join marks+clusters with derived features
    id_derived = "IDCode" if "IDCode" in dfd.columns else _detect_id_col(dfd)
    candidate = [
        "accuracy",
        "avg_rt",
        "var_rt",
        "rt_cv",
        "longest_correct_streak",
        "longest_incorrect_streak",
        "consecutive_correct_rate",
        "response_variance",
        "total_correct",
        "total_incorrect",
    ]
    feat_cols = [c for c in candidate if c in dfd.columns]
    if not feat_cols:
        return
    dfm = dfa[[id_col, cluster_col, total_col]].merge(
        dfd[[id_derived] + feat_cols].rename(columns={id_derived: id_col}), on=id_col, how="inner"
    )

    # Standardize features and TOTAL
    zfeat = _zmean(dfm[feat_cols])
    ztot = _zmean(dfm[[total_col]])[total_col]

    # Orient weights toward low TOTAL = high risk
    corr = dfm[feat_cols].corrwith(dfm[total_col], method="pearson").fillna(0.0)
    weights = -corr  # positive weight raises risk when feature pattern predicts low TOTAL
    denom = float(weights.abs().sum()) or 1.0
    risk_feat = (zfeat * weights).sum(axis=1) / denom

    # Cluster offset from mean z(TOTAL) by cluster
    ztot_by_cluster = pd.Series(ztot).groupby(dfm[cluster_col]).mean()
    cluster_component = -dfm[cluster_col].map(ztot_by_cluster).fillna(0.0)

    # Combine: feature-driven + individual low TOTAL + cluster context
    risk_raw = 0.6 * risk_feat + 0.3 * (-ztot) + 0.1 * cluster_component
    risk = _zmean(pd.DataFrame({"risk": risk_raw}))["risk"]

    # Discrete levels by quantiles
    q_high = float(risk.quantile(0.80))
    q_med = float(risk.quantile(0.50))
    def _bucket(x: float) -> str:
        if x >= q_high:
            return "high"
        if x >= q_med:
            return "medium"
        return "low"
    levels = risk.apply(_bucket)

    # Save outputs
    out_profiles.mkdir(parents=True, exist_ok=True)
    out_figs_risk = out_figs / "risk"
    out_figs_risk.mkdir(parents=True, exist_ok=True)

    weights.to_frame("weight").sort_values("weight").to_csv(out_profiles / "risk_feature_weights.csv")
    res = dfm[[id_col, cluster_col, total_col]].copy()
    res["risk_score_z"] = risk.values
    res["risk_level"] = levels.values
    res.to_csv(out_profiles / "risk_assessment.csv", index=False)

    # Plots: histogram, risk by cluster, accuracy-RT colored by risk (if available)
    plt.figure(figsize=(8, 5))
    sns.histplot(risk, bins=30, kde=True, color="#d62728", alpha=0.6)
    plt.axvline(q_med, color="#ff7f0e", linestyle="--", label=f"median {q_med:.2f}")
    plt.axvline(q_high, color="#2ca02c", linestyle="--", label=f"80th {q_high:.2f}")
    plt.title("Dropout risk score distribution (z)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_figs_risk / "risk_hist.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(x=cluster_col, y="risk_score_z", data=res, estimator=np.mean, errorbar=None, palette="RdYlGn_r")
    plt.title("Mean risk score by cluster")
    plt.tight_layout()
    plt.savefig(out_figs_risk / "risk_by_cluster.png", dpi=150)
    plt.close()

    if {"accuracy", "avg_rt"}.issubset(set(dfm.columns)):
        cmap = sns.color_palette("RdYlGn_r", as_cmap=True)
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(dfm["accuracy"], dfm["avg_rt"], c=risk, cmap=cmap, s=18, alpha=0.7)
        cbar = plt.colorbar(sc)
        cbar.set_label("risk score (z)")
        plt.xlabel("Accuracy")
        plt.ylabel("Avg RT (s)")
        plt.title("Accuracy vs Avg RT colored by dropout risk")
        plt.tight_layout()
        plt.savefig(out_figs_risk / "acc_rt_colored_by_risk.png", dpi=150)
        plt.close()

    # end run()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(Path(sys.argv[1]))
    else:
        run(Path(Path(__file__).parent / "EQTd_DAi_25_cleaned 3_1 for Prince.xlsx"))
