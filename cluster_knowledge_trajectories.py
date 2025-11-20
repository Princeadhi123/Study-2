import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

LEGEND_LOC_UPPER_LEFT = "upper left"
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
]


def get_cluster_palette(n: int) -> list:
    if n <= len(CLUSTER_COLORS):
        return CLUSTER_COLORS[:n]
    reps = int(np.ceil(float(n) / float(len(CLUSTER_COLORS))))
    pal = (CLUSTER_COLORS * reps)[:n]
    return pal


def _longest_streak(arr: np.ndarray, value: int = 1) -> int:
    best = 0
    cur = 0
    for v in arr:
        if v == value:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def _consecutive_correct_rate(arr: np.ndarray) -> float:
    n = len(arr)
    if n <= 1:
        return 0.0
    a = arr[:-1]
    b = arr[1:]
    cc = np.sum((a == 1) & (b == 1))
    return float(cc) / float(n - 1)


def compute_student_features(df: pd.DataFrame) -> pd.DataFrame:
    req_cols = {"IDCode", "orig_order", "response", "response_time_sec"}
    missing = req_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    def per_student(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("orig_order")
        resp = g["response"].astype(int).to_numpy()
        rt = g["response_time_sec"].astype(float).to_numpy()

        n = len(g)
        total_correct = int(resp.sum())
        total_incorrect = int(n - total_correct)
        accuracy = float(total_correct) / float(n) if n else 0.0

        avg_rt = float(np.mean(rt)) if n else 0.0
        var_rt = float(np.var(rt, ddof=1)) if n > 1 else 0.0
        std_rt = float(np.sqrt(var_rt)) if var_rt > 0 else 0.0
        rt_cv = float(std_rt / avg_rt) if avg_rt > 0 else 0.0

        longest_correct_streak = int(_longest_streak(resp, 1))
        longest_incorrect_streak = int(_longest_streak(1 - resp, 1))
        consecutive_correct_rate = float(_consecutive_correct_rate(resp))
        response_variance = float(np.var(resp, ddof=1)) if n > 1 else 0.0

        return pd.Series(
            {
                "n_items": n,
                "total_correct": total_correct,
                "total_incorrect": total_incorrect,
                "accuracy": accuracy,
                "avg_rt": avg_rt,
                "var_rt": var_rt,
                "rt_cv": rt_cv,
                "longest_correct_streak": longest_correct_streak,
                "longest_incorrect_streak": longest_incorrect_streak,
                "consecutive_correct_rate": consecutive_correct_rate,
                "response_variance": response_variance,
            }
        )

    feats = df.groupby("IDCode", sort=False).apply(per_student)
    feats.index.name = "IDCode"
    return feats.reset_index()


def _cluster_convex_hull(a: np.ndarray) -> np.ndarray | None:
    """Compute 2D convex hull of points using a monotonic chain algorithm.

    Expects an array of shape (n_samples, 2). Returns an array of hull points
    in order, or None if a hull cannot be formed (e.g., fewer than 3 points).
    """
    pts = np.asarray(a, dtype=float)
    # Drop NaNs
    pts = pts[~np.isnan(pts).any(axis=1)]
    if pts.shape[0] < 3:
        return None
    # Sort by x, then y
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def _cross(o, b, c):
        return (b[0] - o[0]) * (c[1] - o[1]) - (b[1] - o[1]) * (c[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    if hull.shape[0] < 3:
        return None
    return hull


def scale_features(feats: pd.DataFrame, feature_cols: list) -> Tuple[np.ndarray, StandardScaler]:
    X = feats[feature_cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler


def kmeans_sweep(X: np.ndarray, k_range=range(2, 16)) -> Tuple[np.ndarray, Dict]:
    best = {"k": None, "sil": -1.0}
    best_labels = None
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        try:
            sil = silhouette_score(X, labels)
        except Exception:
            sil = -1.0
        if sil > best["sil"]:
            best = {"k": k, "sil": float(sil)}
            best_labels = labels
    return best_labels, best


def _silhouette_curve_kmeans(X: np.ndarray, k_range=range(2, 16)) -> pd.DataFrame:
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        try:
            sil = silhouette_score(X, labels)
        except Exception:
            sil = -1.0
        rows.append({"K": int(k), "silhouette": float(sil)})
    return pd.DataFrame(rows)


def _silhouette_curve_agglomerative(X: np.ndarray, k_range=range(2, 16)) -> pd.DataFrame:
    rows = []
    for k in k_range:
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = agg.fit_predict(X)
        try:
            sil = silhouette_score(X, labels)
        except Exception:
            sil = -1.0
        rows.append({"K": int(k), "silhouette": float(sil)})
    return pd.DataFrame(rows)


def _silhouette_curve_gmm(
    X: np.ndarray,
    k_range=range(2, 16),
    covariance_types=("full", "diag", "tied", "spherical"),
) -> pd.DataFrame:
    rows = []
    for k in k_range:
        best_sil = -1.0
        best_cov = None
        for cov in covariance_types:
            try:
                gm = GaussianMixture(n_components=k, covariance_type=cov, random_state=42, n_init=5)
                gm.fit(X)
                labels = gm.predict(X)
                if len(np.unique(labels)) <= 1:
                    sil = -1.0
                else:
                    sil = silhouette_score(X, labels)
                if sil > best_sil:
                    best_sil = float(sil)
                    best_cov = cov
            except Exception:
                continue
        rows.append({"K": int(k), "silhouette": float(best_sil), "best_covariance_type": best_cov})
    return pd.DataFrame(rows)


def _silhouette_curve_birch(
    X: np.ndarray,
    k_range=range(2, 16),
    thresholds=(0.3, 0.5, 0.7, 0.9),
    branching_factor: int = 50,
) -> pd.DataFrame:
    rows = []
    for k in k_range:
        best_sil = -1.0
        best_thr = None
        for thr in thresholds:
            try:
                br = Birch(n_clusters=int(k), threshold=float(thr), branching_factor=int(branching_factor))
                labels = br.fit_predict(X)
                if len(np.unique(labels)) <= 1:
                    sil = -1.0
                else:
                    sil = silhouette_score(X, labels)
            except Exception:
                sil = -1.0
            if sil > best_sil:
                best_sil = float(sil)
                best_thr = float(thr)
        rows.append({"K": int(k), "silhouette": float(best_sil), "best_threshold": best_thr})
    return pd.DataFrame(rows)


def _save_line_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str, out_path: Path):
    plt.figure(figsize=(7.5, 5.0))
    sns.lineplot(data=df, x=x_col, y=y_col, marker="o")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()



def analyze_pca_loadings(X: np.ndarray, feature_names: list, out_dir: Path):
    """
    Performs PCA to analyze feature loadings and explained variance.
    Saves:
      1. Explained variance bar plot.
      2. Loadings heatmap (Features x PCs).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Fit PCA
    n_components = min(X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X)
    
    # 1. Explained Variance Plot
    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)
    
    plt.figure(figsize=(8, 5))
    x_range = range(1, len(exp_var) + 1)
    plt.bar(x_range, exp_var, alpha=0.6, label='Individual variance')
    plt.step(x_range, cum_var, where='mid', label='Cumulative variance', color='red')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.title('PCA Explained Variance')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(out_dir / "pca_explained_variance.png", dpi=150)
    plt.close()
    
    # 2. Loadings Heatmap
    # components_ is shape (n_components, n_features)
    # We want rows=Features, cols=PCs for easier reading usually, or vice versa.
    # Let's do Features (rows) x Top PCs (cols).
    
    # Limit to top K components that explain e.g. 90% variance, or just top 5
    n_show = np.argmax(cum_var >= 0.90) + 1
    n_show = max(2, min(n_show, 6)) # Show at least 2, at most 6
    
    loadings = pca.components_[:n_show].T # Shape (n_features, n_show)
    
    # Create DataFrame for heatmap
    pc_names = [f"PC{i+1} ({exp_var[i]:.1%})" for i in range(n_show)]
    loadings_df = pd.DataFrame(loadings, index=feature_names, columns=pc_names)
    
    plt.figure(figsize=(8, max(6, len(feature_names) * 0.4)))
    sns.heatmap(loadings_df, annot=True, cmap="RdBu_r", center=0, fmt=".2f")
    plt.title("PCA Loadings (Feature Correlations with PCs)")
    plt.tight_layout()
    plt.savefig(out_dir / "pca_loadings_heatmap.png", dpi=150)
    plt.close()
    
    # Print top loadings for user inspection
    print("\n=== Top PCA Loadings ===")
    for i in range(n_show):
        pc_loadings = loadings_df.iloc[:, i]
        # Sort by absolute value
        top_indices = pc_loadings.abs().sort_values(ascending=False).head(5).index
        print(f"PC{i+1} ({exp_var[i]:.1%} var):")
        for feat in top_indices:
            val = pc_loadings[feat]
            print(f"  - {feat}: {val:.3f}")
    print("========================\n")
    
    return pca
def analyze_lda_loadings(X: np.ndarray, labels: np.ndarray, feature_names: list, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_arr = np.asarray(labels)
    mask = labels_arr >= 0
    if not np.any(mask):
        return None
    X_use = X[mask]
    y_use = labels_arr[mask]
    uniq = np.unique(y_use)
    if len(uniq) < 2:
        return None
    lda = LDA()
    try:
        lda.fit(X_use, y_use)
    except Exception:
        return None
    scalings = getattr(lda, "scalings_", None)
    if scalings is None:
        return lda
    scalings = np.asarray(scalings)
    n_comp = scalings.shape[1]
    exp_var = getattr(lda, "explained_variance_ratio_", None)
    if exp_var is not None:
        exp_var = np.asarray(exp_var)
        n_comp = min(n_comp, exp_var.shape[0])
    n_show = min(n_comp, 6)
    if n_show <= 0:
        return lda
    loadings = scalings[:, :n_show]
    col_names = []
    for i in range(n_show):
        if exp_var is not None and i < len(exp_var):
            col_names.append(f"LD{i+1} ({exp_var[i]:.1%})")
        else:
            col_names.append(f"LD{i+1}")
    loadings_df = pd.DataFrame(loadings, index=feature_names, columns=col_names)
    plt.figure(figsize=(8, max(6, len(feature_names) * 0.4)))
    sns.heatmap(loadings_df, annot=True, cmap="RdBu_r", center=0, fmt=".2f")
    plt.title("LDA Loadings (Feature Contributions to LDs)")
    plt.tight_layout()
    plt.savefig(out_dir / "lda_loadings_heatmap.png", dpi=150)
    plt.close()
    print("\n=== Top LDA Loadings ===")
    for i in range(n_show):
        ld_loadings = loadings_df.iloc[:, i]
        top_indices = ld_loadings.abs().sort_values(ascending=False).head(5).index
        if exp_var is not None and i < len(exp_var):
            ev_str = f"{exp_var[i]:.1%}"
        else:
            ev_str = "n/a"
        print(f"LD{i+1} ({ev_str} discrim.):")
        for feat in top_indices:
            val = ld_loadings[feat]
            print(f"  - {feat}: {val:.3f}")
    print("========================\n")
    return lda


def tsne_scatter(X: np.ndarray, labels: np.ndarray, title: str, out_path: Path, perplexity: float = 30.0):
    n_samples = X.shape[0]
    if n_samples < 2:
        return
    # Clamp perplexity to a sensible range relative to n_samples
    max_perp = max(5.0, min(perplexity, (n_samples - 1) / 3.0))
    tsne = TSNE(n_components=2, random_state=42, perplexity=max_perp, init="random", learning_rate="auto")
    X2 = tsne.fit_transform(X)
    plt.figure(figsize=(7.0, 6.0))
    lbls_num = labels.astype(int)
    levels = [str(i) for i in sorted(np.unique(lbls_num))]
    palette = get_cluster_palette(len(levels))
    sns.scatterplot(
        x=X2[:, 0],
        y=X2[:, 1],
        hue=lbls_num.astype(str),
        hue_order=levels,
        palette=palette,
        s=30,
        linewidth=0,
    )
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc=LEGEND_LOC_UPPER_LEFT)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def umap_scatter(
    X: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: Path,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
):
    try:
        from umap import UMAP
    except ImportError:
        print("UMAP is not installed; skipping UMAP scatter.")
        return
    n_samples = X.shape[0]
    if n_samples < 2:
        return
    umap_model = UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
    )
    X2 = umap_model.fit_transform(X)
    plt.figure(figsize=(7.0, 6.0))
    lbls_num = labels.astype(int)
    levels = [str(i) for i in sorted(np.unique(lbls_num))]
    palette = get_cluster_palette(len(levels))
    sns.scatterplot(
        x=X2[:, 0],
        y=X2[:, 1],
        hue=lbls_num.astype(str),
        hue_order=levels,
        palette=palette,
        s=30,
        linewidth=0,
    )
    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc=LEGEND_LOC_UPPER_LEFT)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _save_cluster_cards(feats: pd.DataFrame, feature_cols: list, labels: np.ndarray, out_dir: Path, label_name: str = "cluster", top_n: int = 5):
    df = feats.copy()
    df[label_name] = labels
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_cols = [c for c in feature_cols if (c != "n_items") and (df[c].nunique() > 1)]
    mu = df[feature_cols].mean()
    sd = df[feature_cols].std(ddof=0).replace(0, np.nan)
    z = (df[feature_cols] - mu) / sd
    zmean = z.join(df[label_name]).groupby(label_name)[feature_cols].mean().sort_index()
    means = df.groupby(label_name)[feature_cols].mean().sort_index()
    counts = df[label_name].value_counts().sort_index()
    equivalent_feature_groups = {
        "accuracy_level": {"total_correct", "total_incorrect", "accuracy"},
        "correctness_dynamics": {"longest_correct_streak", "longest_incorrect_streak", "consecutive_correct_rate"},
        "response_time": {"avg_rt", "var_rt", "rt_cv"},
    }
    feature_to_group = {}
    for g, cols in equivalent_feature_groups.items():
        for c in cols:
            if c in feature_cols:
                feature_to_group[c] = g
    for k in zmean.index:
        ser = zmean.loc[k].abs().sort_values(ascending=False)
        selected = []
        used_groups = set()
        for f in ser.index:
            g = feature_to_group.get(f, f)
            if g in used_groups:
                continue
            selected.append(f)
            used_groups.add(g)
            if len(selected) >= top_n:
                break
        feat = selected
        zvals = zmean.loc[k, feat].sort_values()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(zvals.index, zvals.values, color=["#d62728" if v>0 else "#1f77b4" for v in zmean.loc[k, zvals.index].values])
        for i, f in enumerate(zvals.index):
            ax.text(zvals.values[i], i, f"  {zvals.values[i]:.2f}", va="center")
        ax.set_xlabel("z-mean")
        ax.set_title(f"Cluster {k}  (n={int(counts.get(k, 0))})")
        plt.tight_layout()
        plt.savefig(out_dir / f"cluster_card_{k}.png", dpi=150)
        plt.close()


def _save_accuracy_speed_ellipse(feats: pd.DataFrame, labels: np.ndarray, out_path: Path, label_name: str = "cluster", x: str = "accuracy", y: str = "avg_rt", xlabel: str = "Accuracy", ylabel: str = "Avg RT (s)", title: str = "Accuracy vs Avg RT with cluster ellipses (GMM BIC)"):
    df = feats.copy()
    df[label_name] = labels
    # Slightly larger figure for better readability while keeping the same data scale
    plt.figure(figsize=(10, 7.5))
    ax = plt.gca()
    uniq = sorted(df[label_name].unique())
    palette = get_cluster_palette(len(uniq))
    for i, k in enumerate(uniq):
        sub = df[df[label_name] == k]
        ax.scatter(sub[x], sub[y], s=10, alpha=0.15, color=palette[i])
        mx, my = sub[x].mean(), sub[y].mean()
        # Cluster boundary as convex hull in the (x, y) plane
        try:
            hull = _cluster_convex_hull(sub[[x, y]].to_numpy())
            if hull is not None:
                hx = np.append(hull[:, 0], hull[0, 0])
                hy = np.append(hull[:, 1], hull[0, 1])
                ax.plot(hx, hy, color=palette[i], linewidth=2.0)
        except Exception:
            pass
        ax.scatter([mx], [my], color=palette[i], s=60, label=str(k), edgecolor='black', linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc=LEGEND_LOC_UPPER_LEFT)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def _save_zmean_heatmap(df: pd.DataFrame, feature_cols: list, cluster_col: str, title: str, out_path: Path):
    # Drop constant features (e.g., n_items) from visualization
    feature_cols = [c for c in feature_cols if (c != "n_items") and (df[c].nunique() > 1)]
    mu = df[feature_cols].mean()
    sd = df[feature_cols].std(ddof=0).replace(0, np.nan)
    z = (df[feature_cols] - mu) / sd
    zmean = z.join(df[cluster_col]).groupby(cluster_col)[feature_cols].mean().sort_index()
    # Order features by overall discriminativeness across clusters (avg |z|)
    col_order = zmean.abs().mean(axis=0).sort_values(ascending=False).index.tolist()
    zmean = zmean[col_order]
    vmax = float(np.nanmax(np.abs(zmean.values))) if zmean.size else 0.0
    vmax = max(1.0, min(3.0, vmax))
    vmin = -vmax
    fig_w = max(10.0, 1.3 * len(feature_cols))
    fig_h = max(8.0, 1.2 * max(6, len(zmean)))
    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.heatmap(
        zmean,
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
    ax.set_xlabel("Feature")
    ax.set_ylabel("Cluster")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    # Show cluster counts in y tick labels
    counts = df[cluster_col].value_counts().sort_index()
    ylabels = []
    for i, k in enumerate(zmean.index.tolist()):
        n = int(counts.get(k, 0))
        ylabels.append(f"{k} (n={n})")
    ax.set_yticklabels(ylabels, rotation=0)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    # Improve annotation contrast based on background value
    for text in ax.texts:
        try:
            val = float(text.get_text())
        except Exception:
            continue
        if abs(val) > (0.6 * vmax):
            text.set_color("white")
        else:
            text.set_color("black")
    plt.savefig(out_path, dpi=150)
    plt.close()


def _save_cluster_profiles(feats: pd.DataFrame, feature_cols: list, labels: np.ndarray, label_name: str, profiles_dir: Path, figures_dir: Path):
    df = feats.copy()
    df[label_name] = labels
    profiles_dir.mkdir(parents=True, exist_ok=True)
    counts = df[label_name].value_counts().sort_index().rename_axis(label_name).reset_index(name="count")
    counts["share"] = counts["count"] / counts["count"].sum()
    counts.to_csv(profiles_dir / f"{label_name}_counts.csv", index=False)
    means = df.groupby(label_name)[feature_cols].mean().sort_index()
    means.to_csv(profiles_dir / f"{label_name}_feature_means.csv")
    mu = df[feature_cols].mean()
    sd = df[feature_cols].std(ddof=0).replace(0, np.nan)
    zmeans = ((means - mu) / sd)
    zmeans.to_csv(profiles_dir / f"{label_name}_feature_zmeans.csv")
    _save_zmean_heatmap(df, feature_cols, label_name, f"Z-mean heatmap: {label_name}", figures_dir / f"{label_name}_zmean_heatmap.png")

def _save_line_plot_hue(df: pd.DataFrame, x_col: str, y_col: str, hue_col: str, title: str, out_path: Path):
    plt.figure(figsize=(7.5, 5.0))
    sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col, marker="o")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def estimate_dbscan_elbow_eps(X: np.ndarray, min_samples: int) -> float:
    n = int(min_samples)
    nbrs = NearestNeighbors(n_neighbors=n).fit(X)
    d, _ = nbrs.kneighbors(X)
    if np.allclose(d[:, 0], 0.0):
        nbrs = NearestNeighbors(n_neighbors=n + 1).fit(X)
        d, _ = nbrs.kneighbors(X)
        kdist = d[:, -1]
    else:
        kdist = d[:, -1]
    kdist = np.sort(kdist)[::-1]
    elbow_idx = _find_elbow_index(kdist)
    return float(kdist[elbow_idx])


def dbscan_multi_grid_diagnostics(
    X: np.ndarray,
    eps_values: np.ndarray,
    min_samples_list: list,
) -> Tuple[pd.DataFrame, Dict, Dict]:
    rows = []
    best_sil = {"eps": None, "min_samples": None, "sil": -1.0, "n_clusters": 0, "noise_rate": 0.0}
    best_comb = {"eps": None, "min_samples": None, "sil": -1.0, "n_clusters": 0, "noise_rate": 0.0, "combined": -1.0}
    for ms in min_samples_list:
        for eps in eps_values:
            db = DBSCAN(eps=float(eps), min_samples=int(ms))
            labels = db.fit_predict(X)
            mask = labels != -1
            unique = np.unique(labels[mask]) if mask.any() else np.array([])
            noise_rate = 1.0 - float(np.sum(mask)) / float(len(labels))
            if len(unique) <= 1:
                sil = -1.0
                n_clusters = len(unique)
            else:
                try:
                    sil = silhouette_score(X[mask], labels[mask])
                except Exception:
                    sil = -1.0
                n_clusters = len(unique)
            combined = float(sil) * (1.0 - float(noise_rate))
            rows.append(
                {
                    "eps": float(eps),
                    "min_samples": int(ms),
                    "n_clusters": int(n_clusters),
                    "noise_rate": float(noise_rate),
                    "silhouette_core": float(sil),
                    "combined": float(combined),
                }
            )
            if sil > best_sil["sil"]:
                best_sil = {"eps": float(eps), "min_samples": int(ms), "sil": float(sil), "n_clusters": int(n_clusters), "noise_rate": float(noise_rate)}
            if combined > best_comb.get("combined", -1.0):
                best_comb = {"eps": float(eps), "min_samples": int(ms), "sil": float(sil), "n_clusters": int(n_clusters), "noise_rate": float(noise_rate), "combined": float(combined)}
    return pd.DataFrame(rows), best_sil, best_comb


def gmm_bic_aic_grid(
    X: np.ndarray,
    k_range=range(2, 11),
    covariance_types=("full", "diag", "tied", "spherical"),
) -> Tuple[pd.DataFrame, Dict, Dict]:
    rows = []
    best_bic = {"k": None, "covariance_type": None, "bic": np.inf, "aic": np.inf}
    best_aic = {"k": None, "covariance_type": None, "bic": np.inf, "aic": np.inf}
    for k in k_range:
        for cov in covariance_types:
            try:
                gm = GaussianMixture(n_components=k, covariance_type=cov, random_state=42, n_init=5)
                gm.fit(X)
                bic = float(gm.bic(X))
                aic = float(gm.aic(X))
                rows.append({"K": int(k), "covariance_type": cov, "bic": bic, "aic": aic})
                if bic < best_bic["bic"]:
                    best_bic = {"k": int(k), "covariance_type": cov, "bic": bic, "aic": aic}
                if aic < best_aic["aic"]:
                    best_aic = {"k": int(k), "covariance_type": cov, "bic": bic, "aic": aic}
            except Exception:
                continue
    return pd.DataFrame(rows), best_bic, best_aic


def _find_elbow_index(y: np.ndarray) -> int:
    x = np.arange(len(y))
    p1 = np.array([x[0], y[0]], dtype=float)
    p2 = np.array([x[-1], y[-1]], dtype=float)
    v = p2 - p1
    if np.allclose(v, 0):
        return int(np.argmax(y))
    v = v / np.linalg.norm(v)
    d = np.abs((x - p1[0]) * (-v[1]) + (y - p1[1]) * v[0])
    return int(np.argmax(d))


def dbscan_grid_diagnostics(
    X: np.ndarray, eps_values=None, min_samples: int = 5
) -> Tuple[pd.DataFrame, np.ndarray, Dict, np.ndarray, Dict]:
    if eps_values is None:
        eps_values = np.linspace(0.5, 3.0, 11)
    rows = []
    best_sil = {"eps": None, "min_samples": int(min_samples), "sil": -1.0, "n_clusters": 0, "noise_rate": 0.0}
    best_labels = None
    best_comb = {"eps": None, "min_samples": int(min_samples), "sil": -1.0, "n_clusters": 0, "noise_rate": 0.0, "combined": -1.0}
    best_comb_labels = None
    for eps in eps_values:
        db = DBSCAN(eps=float(eps), min_samples=int(min_samples))
        labels = db.fit_predict(X)
        mask = labels != -1
        unique = np.unique(labels[mask]) if mask.any() else np.array([])
        noise_rate = 1.0 - float(np.sum(mask)) / float(len(labels))
        if len(unique) <= 1:
            sil = -1.0
            n_clusters = len(unique)
        else:
            try:
                sil = silhouette_score(X[mask], labels[mask])
            except Exception:
                sil = -1.0
            n_clusters = len(unique)
        combined = float(sil) * (1.0 - float(noise_rate))
        rows.append(
            {
                "eps": float(eps),
                "min_samples": int(min_samples),
                "n_clusters": int(n_clusters),
                "noise_rate": float(noise_rate),
                "silhouette_core": float(sil),
                "combined": float(combined),
            }
        )
        if sil > best_sil["sil"]:
            best_sil = {"eps": float(eps), "min_samples": int(min_samples), "sil": float(sil), "n_clusters": int(n_clusters), "noise_rate": float(noise_rate)}
            best_labels = labels
        if combined > best_comb.get("combined", -1.0):
            best_comb = {"eps": float(eps), "min_samples": int(min_samples), "sil": float(sil), "n_clusters": int(n_clusters), "noise_rate": float(noise_rate), "combined": float(combined)}
            best_comb_labels = labels
    return pd.DataFrame(rows), best_labels, best_sil, best_comb_labels, best_comb


def dbscan_k_distance_plot(
    X: np.ndarray, min_samples: int, out_path: Path, mark_eps_list=None
):
    n = int(min_samples)
    nbrs = NearestNeighbors(n_neighbors=n).fit(X)
    d, _ = nbrs.kneighbors(X)
    if np.allclose(d[:, 0], 0.0):
        nbrs = NearestNeighbors(n_neighbors=n + 1).fit(X)
        d, _ = nbrs.kneighbors(X)
        kdist = d[:, -1]
    else:
        kdist = d[:, -1]
    kdist = np.sort(kdist)[::-1]
    elbow_idx = _find_elbow_index(kdist)
    elbow_eps = float(kdist[elbow_idx])
    plt.figure(figsize=(7.5, 5.0))
    plt.plot(np.arange(len(kdist)), kdist, lw=1.5)
    plt.scatter([elbow_idx], [kdist[elbow_idx]], color="red", s=25)
    if mark_eps_list:
        for me in mark_eps_list:
            plt.axhline(me, color="orange", ls="--", lw=1)
    plt.axhline(elbow_eps, color="red", ls=":", lw=1)
    plt.title(f"DBSCAN {n}-distance plot (elbow≈{elbow_eps:.3f})")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    labels_arr = np.asarray(labels)
    if labels_arr.shape[0] != X.shape[0]:
        return np.nan
    uniq = np.unique(labels_arr)
    if len(uniq) < 2:
        return np.nan
    D = pairwise_distances(X)
    np.fill_diagonal(D, np.inf)
    intra = 0.0
    for c in uniq:
        idx = np.where(labels_arr == c)[0]
        if len(idx) <= 1:
            continue
        d_within = D[np.ix_(idx, idx)]
        if d_within.size == 0:
            continue
        intra = max(intra, float(np.nanmax(d_within)))
    if intra <= 0.0 or not np.isfinite(intra):
        return np.nan
    inter = np.inf
    for i, ci in enumerate(uniq):
        idx_i = np.where(labels_arr == ci)[0]
        for cj in uniq[i + 1 :]:
            idx_j = np.where(labels_arr == cj)[0]
            if len(idx_i) == 0 or len(idx_j) == 0:
                continue
            d_between = D[np.ix_(idx_i, idx_j)]
            if d_between.size == 0:
                continue
            inter = min(inter, float(np.nanmin(d_between)))
    if inter <= 0.0 or not np.isfinite(inter):
        return np.nan
    return float(inter / intra)


def compute_internal_validity(X: np.ndarray, labels: np.ndarray) -> Dict:
    result = {
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
        "dunn": None,
    }
    if labels is None:
        return result
    labels_arr = np.asarray(labels)
    if labels_arr.shape[0] != X.shape[0]:
        return result
    mask = labels_arr >= 0
    if not np.any(mask):
        return result
    X_use = X[mask]
    y_use = labels_arr[mask]
    uniq = np.unique(y_use)
    if len(uniq) < 2:
        return result
    try:
        result["silhouette"] = float(silhouette_score(X_use, y_use))
    except Exception:
        pass
    try:
        result["davies_bouldin"] = float(davies_bouldin_score(X_use, y_use))
    except Exception:
        pass
    try:
        result["calinski_harabasz"] = float(calinski_harabasz_score(X_use, y_use))
    except Exception:
        pass
    try:
        result["dunn"] = float(_dunn_index(X_use, y_use))
    except Exception:
        pass
    return result


def compute_external_validity(true_labels: np.ndarray, cluster_labels: np.ndarray) -> Dict:
    result = {
        "rand_index": None,
        "jaccard": None,
    }
    if true_labels is None or cluster_labels is None:
        return result
    y_true = np.asarray(true_labels)
    y_pred = np.asarray(cluster_labels)
    if y_true.shape[0] != y_pred.shape[0]:
        return result
    mask = y_pred >= 0
    if not np.any(mask):
        return result
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = y_true.shape[0]
    if n <= 1:
        return result
    ct = pd.crosstab(pd.Series(y_true, name="true"), pd.Series(y_pred, name="pred"))
    m = ct.to_numpy(dtype=float)

    def _comb2(x: np.ndarray) -> float:
        return float(((x * (x - 1.0)) / 2.0).sum())

    a = _comb2(m)  # pairs in same cluster in both
    row_sums = m.sum(axis=1)
    col_sums = m.sum(axis=0)
    b = _comb2(row_sums) - a  # same in true, different in pred
    c = _comb2(col_sums) - a  # same in pred, different in true
    N = n * (n - 1.0) / 2.0
    d = N - a - b - c  # different in both
    if N > 0.0:
        result["rand_index"] = float((a + d) / N)
    denom_j = a + b + c
    if denom_j > 0.0:
        result["jaccard"] = float(a / denom_j)
    return result


def agglomerative_sweep(X: np.ndarray, k_range=range(2, 11)) -> Tuple[np.ndarray, Dict]:
    best = {"k": None, "sil": -1.0}
    best_labels = None
    for k in k_range:
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = agg.fit_predict(X)
        try:
            sil = silhouette_score(X, labels)
        except Exception:
            sil = -1.0
        if sil > best["sil"]:
            best = {"k": k, "sil": float(sil)}
            best_labels = labels
    return best_labels, best


def gmm_sweep(
    X: np.ndarray,
    k_range=range(2, 16),
    covariance_types=("full", "diag", "tied", "spherical"),
) -> Tuple[np.ndarray, Dict]:
    best = {"k": None, "covariance_type": None, "sil": -1.0}
    best_labels = None
    for k in k_range:
        for cov in covariance_types:
            try:
                gm = GaussianMixture(n_components=k, covariance_type=cov, random_state=42, n_init=5)
                gm.fit(X)
                labels = gm.predict(X)
                if len(np.unique(labels)) <= 1:
                    sil = -1.0
                else:
                    sil = silhouette_score(X, labels)
                if sil > best["sil"]:
                    best = {"k": int(k), "covariance_type": cov, "sil": float(sil)}
                    best_labels = labels
            except Exception:
                continue
    return best_labels, best


def birch_sweep(
    X: np.ndarray,
    k_range=range(2, 16),
    thresholds=(0.3, 0.5, 0.7, 0.9),
    branching_factor: int = 50,
) -> Tuple[np.ndarray, Dict]:
    best = {"k": None, "threshold": None, "branching_factor": int(branching_factor), "sil": -1.0}
    best_labels = None
    for k in k_range:
        for thr in thresholds:
            try:
                br = Birch(n_clusters=int(k), threshold=float(thr), branching_factor=int(branching_factor))
                labels = br.fit_predict(X)
                if len(np.unique(labels)) <= 1:
                    sil = -1.0
                else:
                    sil = silhouette_score(X, labels)
            except Exception:
                sil = -1.0
            if sil > best["sil"]:
                best = {
                    "k": int(k),
                    "threshold": float(thr),
                    "branching_factor": int(branching_factor),
                    "sil": float(sil),
                }
                best_labels = labels
    return best_labels, best


def dbscan_grid(X: np.ndarray, eps_values=None, min_samples: int = 5) -> Tuple[np.ndarray, Dict]:
    if eps_values is None:
        eps_values = np.linspace(0.5, 3.0, 11)
    best = {"eps": None, "min_samples": int(min_samples), "sil": -1.0, "n_clusters": 0}
    best_labels = None
    for eps in eps_values:
        db = DBSCAN(eps=float(eps), min_samples=int(min_samples))
        labels = db.fit_predict(X)
        mask = labels != -1
        unique = np.unique(labels[mask]) if mask.any() else np.array([])
        if len(unique) <= 1:
            sil = -1.0
            n_clusters = len(unique)
        else:
            try:
                sil = silhouette_score(X[mask], labels[mask])
            except Exception:
                sil = -1.0
            n_clusters = len(unique)
        if sil > best["sil"]:
            best = {"eps": float(eps), "min_samples": int(min_samples), "sil": float(sil), "n_clusters": int(n_clusters)}
            best_labels = labels
    return best_labels, best


def pca_scatter(X: np.ndarray, labels: np.ndarray, title: str, out_path: Path):
    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X)
    else:
        X2 = X
    plt.figure(figsize=(7.0, 6.0))
    # Ensure consistent colors and ordered legend (0..K-1)
    lbls_num = labels.astype(int)
    levels = [str(i) for i in sorted(np.unique(lbls_num))]
    palette = get_cluster_palette(len(levels))
    sns.scatterplot(
        x=X2[:, 0],
        y=X2[:, 1],
        hue=lbls_num.astype(str),
        hue_order=levels,
        palette=palette,
        s=30,
        linewidth=0,
    )
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc=LEGEND_LOC_UPPER_LEFT)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def lda_scatter(X: np.ndarray, labels: np.ndarray, title: str, out_path: Path):
    lbls_num = labels.astype(int)
    uniq = np.unique(lbls_num)
    # LDA requires at least 2 classes
    if len(uniq) < 2:
        return
    n_components = min(2, len(uniq) - 1)
    lda = LDA(n_components=n_components)
    try:
        X_lda = lda.fit_transform(X, lbls_num)
    except Exception:
        # In case of numerical issues or singular covariance, skip plot
        return
    if n_components == 1:
        X2 = np.column_stack([X_lda[:, 0], np.zeros_like(X_lda[:, 0])])
    else:
        X2 = X_lda
    plt.figure(figsize=(7.0, 6.0))
    levels = [str(i) for i in sorted(uniq)]
    palette = get_cluster_palette(len(levels))
    sns.scatterplot(
        x=X2[:, 0],
        y=X2[:, 1],
        hue=lbls_num.astype(str),
        hue_order=levels,
        palette=palette,
        s=30,
        linewidth=0,
    )
    plt.title(title)
    plt.xlabel("LD1")
    plt.ylabel("LD2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc=LEGEND_LOC_UPPER_LEFT)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_report(report_path: Path, summary: Dict):
    lines = []
    lines.append("Clustering Performance Report\n")
    lines.append("============================\n\n")
    lines.append(f"N students: {summary.get('n_students')}\n")
    lines.append(f"Features used: {', '.join(summary.get('feature_cols', []))}\n\n")
    km = summary.get("kmeans", {})
    ag = summary.get("agglomerative", {})
    gm = summary.get("gmm", {})
    br = summary.get("birch", {})
    db = summary.get("dbscan", {})
    db_comb = summary.get("dbscan_combined", {})
    db_elbow = summary.get("dbscan_elbow_eps", None)
    db_sel = summary.get("dbscan_selected", {})
    db_noise_thr = summary.get("dbscan_noise_threshold", None)
    gm_bic = summary.get("gmm_bic_best", {})
    gm_aic = summary.get("gmm_aic_best", {})
    best_overall = summary.get("best_overall", {})
    lines.append("KMeans\n")
    lines.append(f"  best_k: {km.get('k')}\n")
    lines.append(f"  silhouette: {km.get('sil')}\n\n")
    lines.append("Agglomerative\n")
    lines.append(f"  best_k: {ag.get('k')}\n")
    lines.append(f"  silhouette: {ag.get('sil')}\n\n")
    lines.append("Birch\n")
    lines.append(f"  best_k: {br.get('k')}\n")
    lines.append(f"  silhouette: {br.get('sil')}\n\n")
    lines.append("GMM\n")
    lines.append(f"  best_k: {gm.get('k')}\n")
    lines.append(f"  covariance_type: {gm.get('covariance_type')}\n")
    lines.append(f"  silhouette: {gm.get('sil')}\n\n")
    lines.append("DBSCAN (core points only)\n")
    lines.append(f"  best_eps_by_silhouette: {db.get('eps')}\n")
    lines.append(f"  min_samples: {db.get('min_samples')}\n")
    lines.append(f"  n_clusters: {db.get('n_clusters')}\n")
    lines.append(f"  noise_rate: {db.get('noise_rate')}\n")
    lines.append(f"  silhouette_core: {db.get('sil')}\n\n")
    if db_comb:
        lines.append("  best_eps_by_combined (silhouette*(1-noise))\n")
        lines.append(f"    eps: {db_comb.get('eps')}\n")
        lines.append(f"    n_clusters: {db_comb.get('n_clusters')}\n")
        lines.append(f"    noise_rate: {db_comb.get('noise_rate')}\n")
        lines.append(f"    silhouette_core: {db_comb.get('sil')}\n")
        lines.append(f"    combined: {db_comb.get('combined')}\n\n")
    if db_elbow is not None:
        lines.append(f"  elbow_eps_estimate (k-distance): {db_elbow}\n\n")
    if db_sel:
        lines.append("DBSCAN auto-selection (threshold on noise rate)\n")
        if db_noise_thr is not None:
            lines.append(f"  noise_rate_threshold: {db_noise_thr}\n")
        lines.append(f"  selected_eps: {db_sel.get('eps')}\n")
        lines.append(f"  selected_min_samples: {db_sel.get('min_samples')}\n")
        lines.append(f"  n_clusters: {db_sel.get('n_clusters')}\n")
        lines.append(f"  noise_rate: {db_sel.get('noise_rate')}\n")
        lines.append(f"  silhouette_core: {db_sel.get('sil')}\n")
        if 'combined' in db_sel:
            lines.append(f"  combined: {db_sel.get('combined')}\n")
        lines.append("\n")

    if gm_bic or gm_aic:
        lines.append("GMM model selection (information criteria)\n")
        if gm_bic:
            lines.append(f"  best_by_BIC: k={gm_bic.get('k')}, cov={gm_bic.get('covariance_type')}, BIC={gm_bic.get('bic')}, AIC={gm_bic.get('aic')}\n")
        if gm_aic:
            lines.append(f"  best_by_AIC: k={gm_aic.get('k')}, cov={gm_aic.get('covariance_type')}, BIC={gm_aic.get('bic')}, AIC={gm_aic.get('aic')}\n")
        lines.append("\n")
    if best_overall:
        lines.append("Best overall clustering (by silhouette)\n")
        lines.append(f"  algorithm: {best_overall.get('algorithm')}\n")
        lines.append(f"  silhouette: {best_overall.get('sil')}\n")
        if best_overall.get("k") is not None:
            lines.append(f"  k: {best_overall.get('k')}\n")
        if best_overall.get("covariance_type") is not None:
            lines.append(f"  covariance_type: {best_overall.get('covariance_type')}\n")
        if best_overall.get("threshold") is not None:
            lines.append(f"  threshold: {best_overall.get('threshold')}\n")
        if best_overall.get("eps") is not None:
            lines.append(f"  eps: {best_overall.get('eps')}\n")
        if best_overall.get("min_samples") is not None:
            lines.append(f"  min_samples: {best_overall.get('min_samples')}\n")
        lines.append("\n")
    report_path.write_text("".join(lines), encoding="utf-8")


def main():
    if len(sys.argv) > 1:
        data_path = Path(sys.argv[1])
    else:
        data_path = Path(__file__).parent / "DigiArvi_25_itemwise.csv"

    out_dir = data_path.parent
    figures_dir = out_dir / "figures"

    df = pd.read_csv(data_path)
    feats = compute_student_features(df)

    feature_cols = [
        "n_items",
        "total_correct",
        "total_incorrect",
        "accuracy",
        "avg_rt",
        "var_rt",
        "rt_cv",
        "longest_correct_streak",
        "longest_incorrect_streak",
        "consecutive_correct_rate",
        "response_variance",
    ]

    Xs, scaler = scale_features(feats, feature_cols)

    # --- NEW: PCA Loadings Analysis ---
    pca_dir = figures_dir / "gmm" / "BIC" / "pca"
    analyze_pca_loadings(Xs, feature_cols, pca_dir)
    # ----------------------------------


    km_labels, km_best = kmeans_sweep(Xs, k_range=range(2, 16))
    ag_labels, ag_best = agglomerative_sweep(Xs, k_range=range(2, 16))
    gm_labels, gm_best = gmm_sweep(Xs, k_range=range(2, 16))
    br_labels, br_best = birch_sweep(Xs, k_range=range(2, 16))

    diagnostics_dir = out_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # Per-K silhouette curves (CSV + plots)
    km_curve = _silhouette_curve_kmeans(Xs, k_range=range(2, 16))
    ag_curve = _silhouette_curve_agglomerative(Xs, k_range=range(2, 16))
    gm_curve = _silhouette_curve_gmm(Xs, k_range=range(2, 16))
    br_curve = _silhouette_curve_birch(Xs, k_range=range(2, 16))
    km_curve.to_csv(diagnostics_dir / "kmeans_silhouette_vs_k.csv", index=False)
    ag_curve.to_csv(diagnostics_dir / "agglomerative_silhouette_vs_k.csv", index=False)
    gm_curve.to_csv(diagnostics_dir / "gmm_silhouette_vs_k.csv", index=False)
    br_curve.to_csv(diagnostics_dir / "birch_silhouette_vs_k.csv", index=False)
    _save_line_plot(km_curve, "K", "silhouette", "KMeans silhouette vs K", figures_dir / "kmeans" / "kmeans_silhouette_vs_k.png")
    _save_line_plot(ag_curve, "K", "silhouette", "Agglomerative silhouette vs K", figures_dir / "agglomerative" / "agglomerative_silhouette_vs_k.png")
    _save_line_plot(gm_curve, "K", "silhouette", "GMM silhouette vs K (best cov)", figures_dir / "gmm" / "gmm_silhouette_vs_k.png")
    _save_line_plot(br_curve, "K", "silhouette", "Birch silhouette vs K", figures_dir / "birch" / "birch_silhouette_vs_k.png")

    # GMM information-criteria diagnostics (BIC/AIC)
    gm_bic_df, gm_bic_best, gm_aic_best = gmm_bic_aic_grid(Xs, k_range=range(2, 16))
    gm_bic_df.to_csv(diagnostics_dir / "gmm_bic_aic.csv", index=False)
    _save_line_plot_hue(gm_bic_df, "K", "bic", "covariance_type", "GMM BIC vs K", figures_dir / "gmm" / "BIC" / "gmm_bic_vs_k.png")
    _save_line_plot_hue(gm_bic_df, "K", "aic", "covariance_type", "GMM AIC vs K", figures_dir / "gmm" / "AIC" / "gmm_aic_vs_k.png")


    # Prepare labels for GMM models chosen by best BIC and best AIC
    try:
        _bic_k = gm_bic_best.get("k")
        _bic_cov = gm_bic_best.get("covariance_type")
        if _bic_k is not None and _bic_cov is not None:
            _gm_bic_model = GaussianMixture(n_components=int(_bic_k), covariance_type=_bic_cov, random_state=42, n_init=5)
            _gm_bic_model.fit(Xs)
            gmm_bic_best_labels = _gm_bic_model.predict(Xs)
        else:
            gmm_bic_best_labels = np.full(len(Xs), -1)
    except Exception:
        gmm_bic_best_labels = np.full(len(Xs), -1)

    # LDA loadings based on GMM best-by-BIC clusters
    try:
        lda_dir = figures_dir / "gmm" / "BIC" / "lda"
        analyze_lda_loadings(Xs, gmm_bic_best_labels, feature_cols, lda_dir)
    except Exception:
        pass

    try:
        _aic_k = gm_aic_best.get("k")
        _aic_cov = gm_aic_best.get("covariance_type")
        if _aic_k is not None and _aic_cov is not None:
            _gm_aic_model = GaussianMixture(n_components=int(_aic_k), covariance_type=_aic_cov, random_state=42, n_init=5)
            _gm_aic_model.fit(Xs)
            gmm_aic_best_labels = _gm_aic_model.predict(Xs)
        else:
            gmm_aic_best_labels = np.full(len(Xs), -1)
    except Exception:
        gmm_aic_best_labels = np.full(len(Xs), -1)

    # DBSCAN diagnostics: sweep over eps with noise penalty + k-distance elbow
    db_diag_df, db_labels_sil, db_best, db_labels_comb, db_best_comb = dbscan_grid_diagnostics(
        Xs, eps_values=np.linspace(0.5, 3.0, 11), min_samples=5
    )
    db_diag_df.to_csv(diagnostics_dir / "dbscan_sweep.csv", index=False)
    elbow_eps = estimate_dbscan_elbow_eps(Xs, min_samples=5)
    # Expanded DBSCAN multi-parameter grid and auto-selection
    eps_values_expanded = np.linspace(0.3, 4.0, 38)
    min_samples_list = [3, 5, 8, 10]
    db_multi_df, db_multi_best_sil, db_multi_best_comb = dbscan_multi_grid_diagnostics(
        Xs, eps_values=eps_values_expanded, min_samples_list=min_samples_list
    )
    db_multi_df.to_csv(diagnostics_dir / "dbscan_multi_sweep.csv", index=False)
    _save_line_plot_hue(db_multi_df, "eps", "silhouette_core", "min_samples", "DBSCAN silhouette_core vs eps", figures_dir / "dbscan" / "dbscan_silhouette_vs_eps.png")
    _save_line_plot_hue(db_multi_df, "eps", "combined", "min_samples", "DBSCAN combined vs eps", figures_dir / "dbscan" / "dbscan_combined_vs_eps.png")
    _save_line_plot_hue(db_multi_df, "eps", "noise_rate", "min_samples", "DBSCAN noise_rate vs eps", figures_dir / "dbscan" / "dbscan_noise_vs_eps.png")

    noise_threshold = 0.20
    if db_multi_best_sil.get("noise_rate", 1.0) <= noise_threshold:
        db_selected = db_multi_best_sil
    else:
        db_selected = db_multi_best_comb
    db_selected_labels = DBSCAN(eps=float(db_selected.get("eps")), min_samples=int(db_selected.get("min_samples"))).fit_predict(Xs)
    dbscan_k_distance_plot(
        Xs,
        5,
        figures_dir / "dbscan" / "dbscan_kdistance.png",
        mark_eps_list=[db_best.get("eps"), db_best_comb.get("eps"), elbow_eps, db_selected.get("eps")],
    )
    db_labels = db_labels_sil

    # Determine best overall clustering by silhouette across algorithms
    best_overall_algo = None
    best_overall_sil = -1.0
    best_overall_labels = None
    best_overall_params = {}

    candidates = [
        ("kmeans", km_best.get("sil"), km_labels, {"k": km_best.get("k")}),
        ("agglomerative", ag_best.get("sil"), ag_labels, {"k": ag_best.get("k")}),
        ("birch", br_best.get("sil"), br_labels, {"k": br_best.get("k"), "threshold": br_best.get("threshold"), "branching_factor": br_best.get("branching_factor")}),
        ("gmm", gm_best.get("sil"), gm_labels, {"k": gm_best.get("k"), "covariance_type": gm_best.get("covariance_type")}),
        ("dbscan", db_best.get("sil"), db_labels, {"eps": db_best.get("eps"), "min_samples": db_best.get("min_samples"), "n_clusters": db_best.get("n_clusters")}),
    ]

    for algo, sil, labels_arr, params in candidates:
        try:
            s_val = float(sil) if sil is not None else -1.0
        except Exception:
            s_val = -1.0
        if labels_arr is None:
            continue
        if s_val > best_overall_sil:
            best_overall_sil = s_val
            best_overall_algo = algo
            best_overall_labels = labels_arr
            best_overall_params = params or {}
    internal_rows = []

    def _add_internal_row(name: str, labels_arr, params: Dict | None = None):
        if labels_arr is None:
            return
        metrics = compute_internal_validity(Xs, labels_arr)
        row = {"algorithm": name}
        if params:
            row.update(params)
        row.update(metrics)
        internal_rows.append(row)

    _add_internal_row("kmeans", km_labels, {"k": km_best.get("k")})
    _add_internal_row("agglomerative", ag_labels, {"k": ag_best.get("k")})
    _add_internal_row(
        "birch",
        br_labels,
        {
            "k": br_best.get("k"),
            "threshold": br_best.get("threshold"),
            "branching_factor": br_best.get("branching_factor"),
        },
    )
    _add_internal_row(
        "gmm",
        gm_labels,
        {
            "k": gm_best.get("k"),
            "covariance_type": gm_best.get("covariance_type"),
        },
    )
    _add_internal_row(
        "dbscan",
        db_labels,
        {
            "eps": db_best.get("eps"),
            "min_samples": db_best.get("min_samples"),
            "n_clusters": db_best.get("n_clusters"),
        },
    )
    _add_internal_row(
        "dbscan_combined",
        db_labels_comb,
        {
            "eps": db_best_comb.get("eps"),
            "min_samples": db_best_comb.get("min_samples"),
            "n_clusters": db_best_comb.get("n_clusters"),
        },
    )
    _add_internal_row(
        "dbscan_selected",
        db_selected_labels,
        {
            "eps": db_selected.get("eps"),
            "min_samples": db_selected.get("min_samples"),
            "n_clusters": db_selected.get("n_clusters"),
        },
    )
    _add_internal_row(
        "gmm_bic_best",
        gmm_bic_best_labels,
        {
            "k": gm_bic_best.get("k"),
            "covariance_type": gm_bic_best.get("covariance_type"),
        },
    )
    _add_internal_row(
        "gmm_aic_best",
        gmm_aic_best_labels,
        {
            "k": gm_aic_best.get("k"),
            "covariance_type": gm_aic_best.get("covariance_type"),
        },
    )
    if best_overall_labels is not None:
        _add_internal_row("best_overall", best_overall_labels, {"base_algorithm": best_overall_algo})

    if internal_rows:
        internal_df = pd.DataFrame(internal_rows)
        internal_df.to_csv(diagnostics_dir / "cluster_internal_validity.csv", index=False)

    clusters_dict = {
        "IDCode": feats["IDCode"],
        "kmeans_label": km_labels,
        "agglomerative_label": ag_labels,
        "birch_label": br_labels,
        "gmm_label": gm_labels,
        "gmm_bic_best_label": gmm_bic_best_labels,
        "gmm_aic_best_label": gmm_aic_best_labels,
        "dbscan_label": db_labels,
        "dbscan_combined_label": db_labels_comb,
        "dbscan_selected_label": db_selected_labels,
    }
    if best_overall_labels is not None:
        clusters_dict["best_overall_label"] = best_overall_labels
    else:
        clusters_dict["best_overall_label"] = np.full(len(feats), -1)

    clusters = pd.DataFrame(clusters_dict)

    if "sex" in df.columns:
        sex_by_id = df.groupby("IDCode")["sex"].first()
        true_sex = sex_by_id.reindex(clusters["IDCode"])
        external_rows = []
        label_cols = [
            "kmeans_label",
            "agglomerative_label",
            "birch_label",
            "gmm_label",
            "gmm_bic_best_label",
            "gmm_aic_best_label",
            "dbscan_label",
            "dbscan_combined_label",
            "dbscan_selected_label",
            "best_overall_label",
        ]
        for col in label_cols:
            if col not in clusters.columns:
                continue
            metrics = compute_external_validity(true_sex.to_numpy(), clusters[col].to_numpy())
            row = {
                "algorithm": col.replace("_label", ""),
                "label_column": col,
            }
            row.update(metrics)
            external_rows.append(row)
        if external_rows:
            external_df = pd.DataFrame(external_rows)
            external_df.to_csv(diagnostics_dir / "cluster_external_validity.csv", index=False)

    feats_out = out_dir / "derived_features.csv"
    clusters_out = out_dir / "student_clusters.csv"
    report_out = out_dir / "_clustering_report.txt"

    feats.to_csv(feats_out, index=False)
    clusters.to_csv(clusters_out, index=False)

    try:
        pca_scatter(Xs, km_labels, f"KMeans (k={km_best.get('k')})", figures_dir / "kmeans" / "kmeans_pca.png")
    except Exception:
        pass
    try:
        pca_scatter(Xs, ag_labels, f"Agglomerative (k={ag_best.get('k')})", figures_dir / "agglomerative" / "agglomerative_pca.png")
    except Exception:
        pass
    try:
        pca_scatter(Xs, br_labels, f"Birch (k={br_best.get('k')})", figures_dir / "birch" / "birch_pca.png")
    except Exception:
        pass
    try:
        pca_scatter(
            Xs,
            gm_labels,
            f"GMM (k={gm_best.get('k')}, cov={gm_best.get('covariance_type')})",
            figures_dir / "gmm" / "gmm_pca.png",
        )
    except Exception:
        pass
    # Additional GMM plots for models selected by information criteria
    try:
        bic_k = gm_bic_best.get("k")
        bic_cov = gm_bic_best.get("covariance_type")
        if bic_k is not None and bic_cov is not None:
            pca_scatter(
                Xs,
                gmm_bic_best_labels,
                f"GMM (best by BIC: k={bic_k}, cov={bic_cov})",
                figures_dir / "gmm" / "BIC" / "gmm_bic_best_pca.png",
            )
            lda_scatter(
                Xs,
                gmm_bic_best_labels,
                f"LDA (GMM best by BIC: k={bic_k}, cov={bic_cov})",
                figures_dir / "gmm" / "BIC" / "gmm_bic_best_lda.png",
            )

    except Exception:
        pass
    try:
        aic_k = gm_aic_best.get("k")
        aic_cov = gm_aic_best.get("covariance_type")
        if aic_k is not None and aic_cov is not None:
            pca_scatter(
                Xs,
                gmm_aic_best_labels,
                f"GMM (best by AIC: k={aic_k}, cov={aic_cov})",
                figures_dir / "gmm" / "AIC" / "gmm_aic_best_pca.png",
            )

    except Exception:
        pass
    # t-SNE and UMAP for GMM best-by-BIC clusters
    try:
        bic_k = gm_bic_best.get("k")
        bic_cov = gm_bic_best.get("covariance_type")
        if bic_k is not None and bic_cov is not None and gmm_bic_best_labels is not None:
            tsne_scatter(
                Xs,
                gmm_bic_best_labels,
                f"t-SNE (GMM best by BIC: k={bic_k}, cov={bic_cov})",
                figures_dir / "gmm" / "BIC" / "gmm_bic_best_tsne.png",
            )

    except Exception:
        pass
    try:
        bic_k = gm_bic_best.get("k")
        bic_cov = gm_bic_best.get("covariance_type")
        if bic_k is not None and bic_cov is not None and gmm_bic_best_labels is not None:
            umap_scatter(
                Xs,
                gmm_bic_best_labels,
                f"UMAP (GMM best by BIC: k={bic_k}, cov={bic_cov})",
                figures_dir / "gmm" / "BIC" / "gmm_bic_best_umap.png",
            )

    except Exception:
        pass
    try:
        pca_scatter(Xs, db_labels, f"DBSCAN (best by silhouette, eps={db_best.get('eps')})", figures_dir / "dbscan" / "dbscan_pca.png")
    except Exception:
        pass
    try:
        pca_scatter(Xs, db_labels_comb, f"DBSCAN (best by combined, eps={db_best_comb.get('eps')})", figures_dir / "dbscan" / "dbscan_combined_pca.png")
    except Exception:
        pass

    try:
        pca_scatter(
            Xs,
            db_selected_labels,
            f"DBSCAN (selected, eps={db_selected.get('eps')}, min_samples={db_selected.get('min_samples')})",
            figures_dir / "dbscan" / "dbscan_selected_pca.png",
        )
    except Exception:
        pass

    # Profiles for interpretation: GMM best-by-BIC
    try:
        _save_cluster_profiles(
            feats,
            feature_cols,
            gmm_bic_best_labels,
            "gmm_bic_best_label",
            out_dir / "profiles",
            figures_dir / "gmm" / "BIC",
        )

    except Exception:
        pass

    # Additional interpretability visuals for GMM BIC clusters
    try:
        _save_zmean_heatmap(
            feats,
            feature_cols,
            "gmm_bic_best_label",
            "Z-mean of technical features by cluster (GMM BIC)",
            figures_dir / "gmm" / "BIC" / "gmm_bic_feature_zmean_by_cluster.png",
        )

    except Exception:
        pass

    
    # Accuracy vs RT with covariance ellipses per cluster
    try:
        _save_accuracy_speed_ellipse(
            feats,
            gmm_bic_best_labels,
            figures_dir / "gmm" / "BIC" / "gmm_bic_accuracy_vs_rt_ellipses.png",
            label_name="gmm_bic_best_label",
            x="accuracy",
            y="avg_rt",
            xlabel="Accuracy",
            ylabel="Avg RT (s)",
            title="Accuracy vs Avg RT with cluster ellipses (GMM BIC)"
        )
        # NEW: Plot based on PCA findings (Accuracy vs Var RT)
        _save_accuracy_speed_ellipse(
            feats,
            gmm_bic_best_labels,
            figures_dir / "gmm" / "BIC" / "gmm_bic_accuracy_vs_var_rt.png",
            label_name="gmm_bic_best_label",
            x="accuracy",
            y="var_rt",
            xlabel="Accuracy",
            ylabel="Variance of RT",
            title="Accuracy vs Variance of RT (GMM BIC)"
        )
        # NEW: Plot based on LDA loadings (Total correct vs Var RT)
        _save_accuracy_speed_ellipse(
            feats,
            gmm_bic_best_labels,
            figures_dir / "gmm" / "BIC" / "gmm_bic_total_correct_vs_var_rt.png",
            label_name="gmm_bic_best_label",
            x="total_correct",
            y="var_rt",
            xlabel="Total correct",
            ylabel="Variance of RT",
            title="Total correct vs Variance of RT (GMM BIC)"
        )
    except Exception:
        pass


    # Cluster cards (top 5 features per cluster with raw means)
    try:
        _save_cluster_cards(
            feats,
            feature_cols,
            gmm_bic_best_labels,
            figures_dir / "gmm" / "BIC" / "cluster_cards",
            label_name="gmm_bic_best_label",
            top_n=6,
        )

    except Exception:
        pass

 

    # Ridgeline plots for selected features


    # t-SNE visualization for the best-overall clustering
    try:
        if best_overall_algo is not None and best_overall_labels is not None:
            if best_overall_algo == "kmeans":
                tsne_path = figures_dir / "kmeans" / "kmeans_tsne.png"
            elif best_overall_algo == "agglomerative":
                tsne_path = figures_dir / "agglomerative" / "agglomerative_tsne.png"
            elif best_overall_algo == "birch":
                tsne_path = figures_dir / "birch" / "birch_tsne.png"
            elif best_overall_algo == "gmm":
                tsne_path = figures_dir / "gmm" / "gmm_tsne.png"
            elif best_overall_algo == "dbscan":
                tsne_path = figures_dir / "dbscan" / "dbscan_tsne.png"
            else:
                tsne_path = figures_dir / "tsne_best_overall.png"
            tsne_scatter(Xs, best_overall_labels, f"t-SNE best overall ({best_overall_algo})", tsne_path)
    except Exception:
        pass

    best_overall_summary = {
        "algorithm": best_overall_algo,
        "sil": float(best_overall_sil) if best_overall_sil is not None else None,
    }
    best_overall_summary.update(best_overall_params)

    summary = {
        "n_students": int(len(feats)),
        "feature_cols": feature_cols,
        "kmeans": km_best,
        "agglomerative": ag_best,
        "birch": br_best,
        "gmm": gm_best,
        "dbscan": db_best,
        "dbscan_combined": db_best_comb,
        "dbscan_elbow_eps": elbow_eps,
        "dbscan_selected": db_selected,
        "dbscan_noise_threshold": noise_threshold,
        "gmm_bic_best": gm_bic_best,
        "gmm_aic_best": gm_aic_best,
        "best_overall": best_overall_summary,
    }
    write_report(report_out, summary)

    print("Derived features written to:", feats_out)
    print("Cluster labels written to:", clusters_out)
    print("Report written to:", report_out)
    print("Figures in:", figures_dir)


if __name__ == "__main__":
    main()
