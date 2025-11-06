import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
from matplotlib.patches import Ellipse

warnings.filterwarnings("ignore")


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


def scale_features(feats: pd.DataFrame, feature_cols: list) -> Tuple[np.ndarray, StandardScaler]:
    X = feats[feature_cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler


def kmeans_sweep(X: np.ndarray, k_range=range(2, 11)) -> Tuple[np.ndarray, Dict]:
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


def _silhouette_curve_kmeans(X: np.ndarray, k_range=range(2, 11)) -> pd.DataFrame:
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


def _silhouette_curve_agglomerative(X: np.ndarray, k_range=range(2, 11)) -> pd.DataFrame:
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
    k_range=range(2, 11),
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


def _save_line_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str, out_path: Path):
    plt.figure(figsize=(7.5, 5.0))
    sns.lineplot(data=df, x=x_col, y=y_col, marker="o")
    plt.title(title)
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
    for k in zmean.index:
        ser = zmean.loc[k].abs().sort_values(ascending=False)
        feat = ser.index[:top_n]
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

 

def _save_ridgeline_plots(feats: pd.DataFrame, labels: np.ndarray, out_dir: Path, features=None, label_name: str = "cluster"):
    df = feats.copy()
    df[label_name] = labels
    out_dir.mkdir(parents=True, exist_ok=True)
    if features is None:
        features = ["accuracy", "avg_rt", "rt_cv"]
    for col in features:
        d = pd.DataFrame({col: df[col], label_name: df[label_name]})
        g = sns.FacetGrid(d, row=label_name, hue=label_name, aspect=6, height=1.0, palette="tab10", sharex=True, sharey=False)
        g.map(sns.kdeplot, col, fill=True, alpha=0.8, linewidth=1)
        g.map(plt.axhline, y=0, lw=1, clip_on=False)
        g.set(yticks=[], ylabel="")
        g.set_titles(row_template="{row_name}")
        g.fig.subplots_adjust(hspace=-0.5)
        g.fig.suptitle(f"Ridgeline: {col}", y=1.02)
        plt.tight_layout()
        g.savefig(out_dir / f"ridgeline_{col}.png", dpi=150)
        plt.close(g.fig)
def _compute_effect_sizes(feats: pd.DataFrame, feature_cols: list, labels: np.ndarray, label_name: str = "cluster") -> pd.DataFrame:
    df = feats.copy()
    df[label_name] = labels
    # Drop constant features
    feature_cols = [c for c in feature_cols if df[c].nunique() > 1]
    effects = {}
    for k in sorted(df[label_name].unique()):
        g = df[df[label_name] == k][feature_cols]
        r = df[df[label_name] != k][feature_cols]
        n1, n2 = len(g), len(r)
        if n1 < 2 or n2 < 2:
            es = pd.Series({c: np.nan for c in feature_cols}, name=k)
        else:
            mu1, mu2 = g.mean(), r.mean()
            s1, s2 = g.std(ddof=1), r.std(ddof=1)
            pooled = np.sqrt(((n1 - 1) * (s1 ** 2) + (n2 - 1) * (s2 ** 2)) / max(n1 + n2 - 2, 1))
            es = (mu1 - mu2) / pooled.replace(0, np.nan)
            es.name = k
        effects[k] = es
    return pd.DataFrame(effects).T

def _save_effect_size_heatmap(effect_df: pd.DataFrame, out_path: Path, title: str = "Cohen's d: cluster vs rest"):
    # Order columns by overall discriminativeness
    order = effect_df.abs().mean(axis=0).sort_values(ascending=False).index
    plt.figure(figsize=(1.2 * len(order), 0.6 * max(6, len(effect_df))))
    sns.heatmap(effect_df[order], cmap="coolwarm", center=0, annot=False, cbar=True)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def _save_accuracy_speed_ellipse(feats: pd.DataFrame, labels: np.ndarray, out_path: Path, label_name: str = "cluster", x: str = "accuracy", y: str = "avg_rt"):
    df = feats.copy()
    df[label_name] = labels
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    uniq = sorted(df[label_name].unique())
    palette = sns.color_palette("tab10", n_colors=len(uniq))
    for i, k in enumerate(uniq):
        sub = df[df[label_name] == k]
        ax.scatter(sub[x], sub[y], s=10, alpha=0.15, color=palette[i])
        mx, my = sub[x].mean(), sub[y].mean()
        if len(sub) > 2:
            cov = np.cov(sub[x], sub[y])
            if np.isfinite(cov).all():
                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]
                theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
                width, height = 2 * 1.5 * np.sqrt(np.maximum(vals, 1e-12))
                e = Ellipse((mx, my), width, height, angle=theta, edgecolor=palette[i], facecolor='none', lw=2)
                ax.add_patch(e)
        ax.scatter([mx], [my], color=palette[i], s=60, label=str(k), edgecolor='black', linewidth=0.5)
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Avg RT (s)")
    ax.set_title("Accuracy vs Avg RT with cluster ellipses (GMM BIC)")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def _save_radar_all_clusters(feats: pd.DataFrame, feature_cols: list, labels: np.ndarray, out_path: Path, label_name: str = "cluster"):
    df = feats.copy()
    df[label_name] = labels
    # Drop constant features (e.g., n_items) from visualization
    feature_cols = [c for c in feature_cols if (c != "n_items") and (df[c].nunique() > 1)]
    mu = df[feature_cols].mean()
    sd = df[feature_cols].std(ddof=0).replace(0, np.nan)
    z = (df[feature_cols] - mu) / sd
    zmean = z.join(df[label_name]).groupby(label_name)[feature_cols].mean().sort_index()
    cats = feature_cols
    N = len(cats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    plt.figure(figsize=(9, 7))
    ax = plt.subplot(111, polar=True)
    palette = sns.color_palette("tab10", n_colors=len(zmean))
    for i, k in enumerate(zmean.index.tolist()):
        vals = zmean.loc[k, :].to_numpy().astype(float)
        vals = np.clip(vals, -3.0, 3.0)
        vals = vals.tolist() + [vals[0]]
        ax.plot(angles, vals, color=palette[i], linewidth=2, label=str(k))
        ax.fill(angles, vals, color=palette[i], alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats)
    ax.set_yticks([-3, -1.5, 0, 1.5, 3])
    ax.set_ylim(-3, 3)
    ax.set_title("Cluster z-mean radar (GMM BIC)")
    ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def _save_feature_boxplots(feats: pd.DataFrame, feature_cols: list, labels: np.ndarray, label_name: str, out_dir: Path):
    df = feats.copy()
    df[label_name] = labels
    out_dir.mkdir(parents=True, exist_ok=True)
    # Drop constant features (e.g., n_items) from visualization
    feature_cols = [c for c in feature_cols if (c != "n_items") and (df[c].nunique() > 1)]
    for col in feature_cols:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x=label_name, y=col, showfliers=False)
        sns.stripplot(data=df, x=label_name, y=col, size=2, color="black", alpha=0.2)
        plt.title(f"{col} by {label_name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{col}_by_{label_name}.png", dpi=150)
        plt.close()

def _save_parallel_coordinates(feats: pd.DataFrame, feature_cols: list, labels: np.ndarray, out_path: Path, label_name: str = "cluster"):
    df = feats.copy()
    df[label_name] = labels
    mu = df[feature_cols].mean()
    sd = df[feature_cols].std(ddof=0).replace(0, np.nan)
    z = (df[feature_cols] - mu) / sd
    plot_df = z.copy()
    plot_df[label_name] = df[label_name].astype(str)
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    parallel_coordinates(plot_df, class_column=label_name, cols=feature_cols, ax=ax, color=sns.color_palette("tab10"))
    for ln in ax.lines:
        ln.set_alpha(0.12)
    # Overlay cluster medians for clarity
    med = plot_df.groupby(label_name)[feature_cols].median().sort_index()
    palette = sns.color_palette("tab10", n_colors=len(med))
    angles = np.arange(len(feature_cols))
    for i, (k, row) in enumerate(med.iterrows()):
        ax.plot(angles, row.values, color=palette[i], linewidth=2.5)
    ax.set_title("Parallel coordinates (z-scored), with cluster medians")
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _save_zmean_heatmap(df: pd.DataFrame, feature_cols: list, cluster_col: str, title: str, out_path: Path):
    # Drop constant features (e.g., n_items) from visualization
    feature_cols = [c for c in feature_cols if (c != "n_items") and (df[c].nunique() > 1)]
    mu = df[feature_cols].mean()
    sd = df[feature_cols].std(ddof=0).replace(0, np.nan)
    z = (df[feature_cols] - mu) / sd
    zmean = z.join(df[cluster_col]).groupby(cluster_col)[feature_cols].mean().sort_index()
    plt.figure(figsize=(1.2 * len(feature_cols), 0.6 * max(6, len(zmean))))
    sns.heatmap(zmean, cmap="coolwarm", center=0, annot=False, cbar=True)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
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

def _save_gmm_bic_composite(feats: pd.DataFrame, feature_cols: list, labels: np.ndarray, X: np.ndarray, out_path: Path):
    df = feats.copy()
    df["cluster"] = labels
    # Drop constant features (e.g., n_items) from visualization
    feature_cols = [c for c in feature_cols if (c != "n_items") and (df[c].nunique() > 1)]
    # Z-mean profiles
    mu = df[feature_cols].mean()
    sd = df[feature_cols].std(ddof=0).replace(0, np.nan)
    z = (df[feature_cols] - mu) / sd
    zmean = z.join(df["cluster"]).groupby("cluster")[feature_cols].mean().sort_index()
    # Counts
    counts = df["cluster"].value_counts().sort_index()
    # PCA projection
    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X)
    else:
        X2 = X
    # Figure layout
    plt.figure(figsize=(14, 8))
    gs = plt.GridSpec(2, 2, width_ratios=[1.1, 1.6], height_ratios=[2.0, 1.0])
    ax_heat = plt.subplot(gs[0, 0])
    sns.heatmap(zmean, cmap="coolwarm", center=0, annot=False, cbar=True, ax=ax_heat)
    ax_heat.set_title("GMM (BIC) z-mean feature profiles")
    ax_bar = plt.subplot(gs[1, 0])
    ax_bar.bar(counts.index.astype(str), counts.values, color="#69b3a2")
    ax_bar.set_title("Cluster sizes")
    ax_bar.set_xlabel("Cluster")
    ax_bar.set_ylabel("Count")
    ax_sc = plt.subplot(gs[:, 1])
    sc = ax_sc.scatter(X2[:, 0], X2[:, 1], c=labels, cmap="tab10", s=25)
    ax_sc.set_title("PCA scatter by cluster (GMM BIC)")
    ax_sc.set_xlabel("PC1")
    ax_sc.set_ylabel("PC2")
    handles, _ = sc.legend_elements(prop="colors", alpha=0.6)
    ax_sc.legend(handles, [str(i) for i in sorted(counts.index)], title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

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
    best_sil_labels = None
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
            best_sil_labels = labels
        if combined > best_comb.get("combined", -1.0):
            best_comb = {"eps": float(eps), "min_samples": int(min_samples), "sil": float(sil), "n_clusters": int(n_clusters), "noise_rate": float(noise_rate), "combined": float(combined)}
            best_comb_labels = labels
    return pd.DataFrame(rows), best_sil_labels, best_sil, best_comb_labels, best_comb


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
    k_range=range(2, 11),
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
    palette = sns.color_palette("tab10", n_colors=len(levels))
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
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
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
    db = summary.get("dbscan", {})
    db_comb = summary.get("dbscan_combined", {})
    db_elbow = summary.get("dbscan_elbow_eps", None)
    db_sel = summary.get("dbscan_selected", {})
    db_noise_thr = summary.get("dbscan_noise_threshold", None)
    gm_bic = summary.get("gmm_bic_best", {})
    gm_aic = summary.get("gmm_aic_best", {})
    lines.append("KMeans\n")
    lines.append(f"  best_k: {km.get('k')}\n")
    lines.append(f"  silhouette: {km.get('sil')}\n\n")
    lines.append("Agglomerative\n")
    lines.append(f"  best_k: {ag.get('k')}\n")
    lines.append(f"  silhouette: {ag.get('sil')}\n\n")
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

    km_labels, km_best = kmeans_sweep(Xs, k_range=range(2, 11))
    ag_labels, ag_best = agglomerative_sweep(Xs, k_range=range(2, 11))
    gmm_labels, gmm_best = gmm_sweep(Xs, k_range=range(2, 11))

    diagnostics_dir = out_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # Per-K silhouette curves (CSV + plots)
    km_curve = _silhouette_curve_kmeans(Xs, k_range=range(2, 11))
    ag_curve = _silhouette_curve_agglomerative(Xs, k_range=range(2, 11))
    gm_curve = _silhouette_curve_gmm(Xs, k_range=range(2, 11))
    km_curve.to_csv(diagnostics_dir / "kmeans_silhouette_vs_k.csv", index=False)
    ag_curve.to_csv(diagnostics_dir / "agglomerative_silhouette_vs_k.csv", index=False)
    gm_curve.to_csv(diagnostics_dir / "gmm_silhouette_vs_k.csv", index=False)
    _save_line_plot(km_curve, "K", "silhouette", "KMeans silhouette vs K", figures_dir / "kmeans_silhouette_vs_k.png")
    _save_line_plot(ag_curve, "K", "silhouette", "Agglomerative silhouette vs K", figures_dir / "agglomerative_silhouette_vs_k.png")
    _save_line_plot(gm_curve, "K", "silhouette", "GMM silhouette vs K (best cov)", figures_dir / "gmm_silhouette_vs_k.png")

    # GMM information-criteria diagnostics (BIC/AIC)
    gm_bic_df, gm_bic_best, gm_aic_best = gmm_bic_aic_grid(Xs, k_range=range(2, 11))
    gm_bic_df.to_csv(diagnostics_dir / "gmm_bic_aic.csv", index=False)
    _save_line_plot_hue(gm_bic_df, "K", "bic", "covariance_type", "GMM BIC vs K", figures_dir / "gmm_bic_vs_k.png")
    _save_line_plot_hue(gm_bic_df, "K", "aic", "covariance_type", "GMM AIC vs K", figures_dir / "gmm_aic_vs_k.png")

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
    _save_line_plot_hue(db_multi_df, "eps", "silhouette_core", "min_samples", "DBSCAN silhouette_core vs eps", figures_dir / "dbscan_silhouette_vs_eps.png")
    _save_line_plot_hue(db_multi_df, "eps", "combined", "min_samples", "DBSCAN combined vs eps", figures_dir / "dbscan_combined_vs_eps.png")
    _save_line_plot_hue(db_multi_df, "eps", "noise_rate", "min_samples", "DBSCAN noise_rate vs eps", figures_dir / "dbscan_noise_vs_eps.png")

    noise_threshold = 0.20
    if db_multi_best_sil.get("noise_rate", 1.0) <= noise_threshold:
        db_selected = db_multi_best_sil
    else:
        db_selected = db_multi_best_comb
    db_selected_labels = DBSCAN(eps=float(db_selected.get("eps")), min_samples=int(db_selected.get("min_samples"))).fit_predict(Xs)
    dbscan_k_distance_plot(
        Xs,
        5,
        figures_dir / "dbscan_kdistance.png",
        mark_eps_list=[db_best.get("eps"), db_best_comb.get("eps"), elbow_eps, db_selected.get("eps")],
    )
    db_labels = db_labels_sil

    clusters = pd.DataFrame(
        {
            "IDCode": feats["IDCode"],
            "kmeans_label": km_labels,
            "agglomerative_label": ag_labels,
            "gmm_label": gmm_labels,
            "gmm_bic_best_label": gmm_bic_best_labels,
            "gmm_aic_best_label": gmm_aic_best_labels,
            "dbscan_label": db_labels,
            "dbscan_combined_label": db_labels_comb,
            "dbscan_selected_label": db_selected_labels,
        }
    )

    feats_out = out_dir / "derived_features.csv"
    clusters_out = out_dir / "student_clusters.csv"
    report_out = out_dir / "_clustering_report.txt"

    feats.to_csv(feats_out, index=False)
    clusters.to_csv(clusters_out, index=False)

    try:
        pca_scatter(Xs, km_labels, f"KMeans (k={km_best.get('k')})", figures_dir / "kmeans_pca.png")
    except Exception:
        pass
    try:
        pca_scatter(Xs, ag_labels, f"Agglomerative (k={ag_best.get('k')})", figures_dir / "agglomerative_pca.png")
    except Exception:
        pass
    try:
        pca_scatter(
            Xs,
            gmm_labels,
            f"GMM (k={gmm_best.get('k')}, cov={gmm_best.get('covariance_type')})",
            figures_dir / "gmm_pca.png",
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
                figures_dir / "gmm_bic_best_pca.png",
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
                figures_dir / "gmm_aic_best_pca.png",
            )
    except Exception:
        pass
    try:
        pca_scatter(Xs, db_labels, f"DBSCAN (best by silhouette, eps={db_best.get('eps')})", figures_dir / "dbscan_pca.png")
    except Exception:
        pass
    try:
        pca_scatter(Xs, db_labels_comb, f"DBSCAN (best by combined, eps={db_best_comb.get('eps')})", figures_dir / "dbscan_combined_pca.png")
    except Exception:
        pass

    try:
        pca_scatter(
            Xs,
            db_selected_labels,
            f"DBSCAN (selected, eps={db_selected.get('eps')}, min_samples={db_selected.get('min_samples')})",
            figures_dir / "dbscan_selected_pca.png",
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
            figures_dir,
        )
    except Exception:
        pass

    # Composite summary figure: heatmap + sizes + PCA scatter
    try:
        _save_gmm_bic_composite(
            feats,
            feature_cols,
            gmm_bic_best_labels,
            Xs,
            figures_dir / "gmm_bic_cluster_summary.png",
        )
    except Exception:
        pass

    # Additional interpretability visuals for GMM BIC clusters
    try:
        _save_radar_all_clusters(
            feats,
            feature_cols,
            gmm_bic_best_labels,
            figures_dir / "gmm_bic_radar_all.png",
            label_name="gmm_bic_best_label",
        )
    except Exception:
        pass
    try:
        _save_feature_boxplots(
            feats,
            feature_cols,
            gmm_bic_best_labels,
            label_name="gmm_bic_best_label",
            out_dir=figures_dir / "boxplots",
        )
    except Exception:
        pass
    
    # Accuracy vs RT with covariance ellipses per cluster
    try:
        _save_accuracy_speed_ellipse(
            feats,
            gmm_bic_best_labels,
            figures_dir / "gmm_bic_accuracy_vs_rt_ellipses.png",
            label_name="gmm_bic_best_label",
            x="accuracy",
            y="avg_rt",
        )
    except Exception:
        pass

    # Cluster cards (top 5 features per cluster with raw means)
    try:
        _save_cluster_cards(
            feats,
            feature_cols,
            gmm_bic_best_labels,
            figures_dir / "cluster_cards",
            label_name="gmm_bic_best_label",
            top_n=6,
        )
    except Exception:
        pass

 

    # Ridgeline plots for selected features
    try:
        _save_ridgeline_plots(
            feats,
            gmm_bic_best_labels,
            figures_dir / "ridgelines",
            features=["accuracy", "avg_rt", "rt_cv"],
            label_name="gmm_bic_best_label",
        )
    except Exception:
        pass

    summary = {
        "n_students": int(len(feats)),
        "feature_cols": feature_cols,
        "kmeans": km_best,
        "agglomerative": ag_best,
        "gmm": gmm_best,
        "dbscan": db_best,
        "dbscan_combined": db_best_comb,
        "dbscan_elbow_eps": elbow_eps,
        "dbscan_selected": db_selected,
        "dbscan_noise_threshold": noise_threshold,
        "gmm_bic_best": gm_bic_best,
        "gmm_aic_best": gm_aic_best,
    }
    write_report(report_out, summary)

    print("Derived features written to:", feats_out)
    print("Cluster labels written to:", clusters_out)
    print("Report written to:", report_out)
    print("Figures in:", figures_dir)


if __name__ == "__main__":
    main()
