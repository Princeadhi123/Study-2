import sys
from pathlib import Path
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


CLUSTER_COLORS = [
    "#E41A1C",  # strong red
    "#377EB8",  # strong blue
    "#4DAF4A",  # green
    "#984EA3",  # purple
    "#FF7F00",  # orange
    "#00CED1",  # cyan
    "#A65628",  # brown
    "#F781BF",  # pink
    "#1B9E77",  # teal
    "#D95F02",  # dark orange
    "#7570B3",  # indigo
    "#E7298A",  # magenta
    "#66A61E",  # olive green
    "#E6AB02",  # mustard
    "#A6761D",  # ochre
    "#666666",  # dark grey (used only if many clusters)
]


def get_cluster_palette(n: int) -> list:
    if n <= len(CLUSTER_COLORS):
        return CLUSTER_COLORS[:n]
    reps = int(np.ceil(float(n) / float(len(CLUSTER_COLORS))))
    pal = (CLUSTER_COLORS * reps)[:n]
    return pal


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


def _save_zmean_heatmap(df: pd.DataFrame, cluster_col: str, value_cols: list, out_path: Path, title: str):
    z = _zmean(df[value_cols])
    mat = z.join(df[cluster_col]).groupby(cluster_col)[value_cols].mean().sort_index()
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
    ax.set_xlabel("Subject area")
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
    palette = get_cluster_palette(len(zmean))
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

def run(marks_path: Path):
    base = Path(__file__).parent
    clusters_path = base / "diagnostics" / "student cluster labels" / "student_clusters.csv"
    derived_path = base / "diagnostics" / "cluster input features" / "derived_features.csv"
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

    out_profiles = base / "diagnostics" / "gmm profiles BIC"
    out_figs = base / "figures" / "subjectwise by cluster"

    cluster_input_features_dir = base / "diagnostics" / "cluster input features"
    cluster_input_features_dir.mkdir(parents=True, exist_ok=True)
    dfa.to_csv(cluster_input_features_dir / "marks_with_clusters.csv", index=False)

    value_cols = subj_cols

    _save_zmean_heatmap(dfa, cluster_col, value_cols, out_figs / "subject_zmean_by_cluster.png", "Subject-wise z-mean by cluster")

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


if __name__ == "__main__":
    base = Path(__file__).parent
    if len(sys.argv) > 1:
        run(Path(sys.argv[1]))
    else:
        run(base / "data" / "EQTd_DAi_25_cleaned 3_1 for Prince.xlsx")
