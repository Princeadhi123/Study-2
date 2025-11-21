# Study 2 – Student Knowledge Trajectories and Cluster Diagnostics

This repository contains code to:

- Derive per-student behavioural features from item-level response data.
- Run multiple clustering algorithms over these features.
- Compare and select clustering solutions using internal and external validity metrics.
- Generate rich visual diagnostics (PCA/LDA loadings, 2D embeddings, cluster profiles, etc.).
- Link cluster memberships to subject-wise marks and visualise subject profiles by cluster.

The analysis is organised around two main scripts:

- `cluster_knowledge_trajectories.py`
- `subjectwise_by_cluster.py`

---

## 1. Repository structure

At the top level of `Study-2` you will typically have:

- **`cluster_knowledge_trajectories.py`**  
  Main pipeline. Reads an itemwise CSV, computes per-student features, runs clustering, evaluates models, and writes diagnostics + figures.

- **`subjectwise_by_cluster.py`**  
  Post-hoc analysis. Merges student cluster labels with an Excel file of subject-wise marks and produces cluster-by-subject profiles.

- **`data/`**  
  - `DigiArvi_25_itemwise.csv` – default input for `cluster_knowledge_trajectories.py` (item-level responses).  
  - `EQTd_DAi_25_cleaned 3_1 for Prince.xlsx` – default input for `subjectwise_by_cluster.py` (subject marks).

- **`diagnostics/`** *(created by the scripts)*  
  - `cluster input features/` – derived per-student features and merged marks+clusters.  
  - `student cluster labels/` – cluster labels per student for each clustering method.  
  - `model results/` – numeric diagnostics for clustering model sweeps (e.g. silhouette vs K, GMM BIC/AIC grid, DBSCAN sweeps).  
  - `cluster validity/` – internal and external cluster validity metrics.

- **`figures/`** *(created by the scripts)*  
  Subfolders for each algorithm and analysis, for example:  
  - `kmeans/`, `agglomerative/`, `birch/`, `gmm/`, `dbscan/`  
  - `gmm/BIC/`, `gmm/AIC/`  
  - `cluster validity/`  
  - `subjectwise by cluster/`

- **`.gitignore`** – Git ignore rules.
- **`README.md`** – this file.

Your exact file names may differ, but the scripts assume the default layout above unless you pass custom paths on the command line.

---

## 2. Dependencies

Tested with **Python 3.10+**.

Core Python packages:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `umap-learn` *(optional, used only if installed for UMAP plots)*
- `openpyxl` *(or another Excel engine; used by `pandas.read_excel`)*

### 2.1. Install with `pip`

From the `Study-2` directory, create an environment and install dependencies, for example:

```bash
python -m venv .venv
.venv\Scripts\activate  # on Windows

pip install numpy pandas scikit-learn matplotlib seaborn umap-learn openpyxl
```

If you do not need UMAP plots, you can omit `umap-learn`.

---

## 3. Input data formats

### 3.1. Itemwise response file (`DigiArvi_25_itemwise.csv`)

The clustering pipeline expects a **long-format** CSV with at least the following columns:

- **`IDCode`** – student identifier.
- **`orig_order`** – original item order *within each student*. This is used to compute streak-related and consecutive-correct features.
- **`response`** – binary correctness indicator (0/1 or equivalent) for each item.
- **`response_time_sec`** – response time for each item in **seconds**.

Optional columns:

- **`sex`** – used for external validity checks (Rand index and Jaccard vs true labels). If absent, external validity is skipped.
- Any other item-level metadata columns are ignored by the core feature computation.

The script `compute_student_features` groups by `IDCode`, sorts by `orig_order`, and computes per-student summary features. If any of the required columns are missing, the script will raise an error.

### 3.2. Subject-wise marks file (`EQTd_DAi_25_cleaned 3_1 for Prince.xlsx`)

`subjectwise_by_cluster.py` expects an Excel file with:

- A student ID column (e.g. `IDCode`, `ID`, `StudentID`).  
  The script auto-detects an ID column via `_detect_id_col`.

- One or more subject columns named like `s1`, `s2`, ..., `sK` (case-insensitive).  
  These are detected by `_detect_subject_cols`.

- Optionally, a **total** or **overall** column, detected by `_detect_total_col` (e.g. containing `total`, `sum`, `overall`).

The script keeps only the detected ID column, all subject columns, and the optional total column.

---

## 4. Running the clustering pipeline

### 4.1. Default run (using repository data)

From the `Study-2` directory:

```bash
python cluster_knowledge_trajectories.py
```

This uses:

- **Input**: `data/DigiArvi_25_itemwise.csv`
- **Output root**: the project root (`Study-2/`)

It will:

1. Read the itemwise CSV.  
2. Compute per-student features via `compute_student_features`:
   - `n_items`
   - `total_correct`, `total_incorrect`, `accuracy`
   - `avg_rt`, `var_rt`, `rt_cv`
   - `longest_correct_streak`, `longest_incorrect_streak`
   - `consecutive_correct_rate`
   - `response_variance`
3. Standardise features (`StandardScaler`).
4. Run multiple clustering algorithms:
   - KMeans
   - Agglomerative (Ward linkage)
   - Birch
   - Gaussian Mixture Models (GMM) with multiple covariance types
   - DBSCAN with grid search over `eps` and `min_samples`
5. Compute internal validity metrics where possible:
   - Silhouette
   - Davies–Bouldin
   - Calinski–Harabasz
   - Average intra-cluster distance
6. If `sex` is present, compute external validity vs `sex` labels:
   - Rand index
   - Jaccard coefficient
7. Perform model selection and “best overall” selection based on silhouette and DBSCAN noise penalties.
8. Generate figures and summary reports.

### 4.2. Custom input path

You can supply a different itemwise CSV:

```bash
python cluster_knowledge_trajectories.py path\to\your_itemwise.csv
```

- **Input**: `path\to\your_itemwise.csv`
- **Output root**: `parent directory of your_itemwise.csv`

All `diagnostics/` and `figures/` folders will be created under that output root.

---

## 5. Outputs from `cluster_knowledge_trajectories.py`

### 5.1. Derived features and cluster labels

Under `diagnostics/` (relative to the chosen output root):

- **`cluster input features/derived_features.csv`**  
  Per-student feature matrix with one row per `IDCode`.

- **`student cluster labels/student_clusters.csv`**  
  One row per `IDCode`, containing cluster labels for each method, for example:
  - `kmeans_label`
  - `agglomerative_label`
  - `birch_label`
  - `gmm_label`
  - `gmm_bic_best_label`
  - `gmm_aic_best_label`
  - `dbscan_label`
  - `dbscan_combined_label`
  - `dbscan_selected_label`
  - `best_overall_label` (if defined)

### 5.2. Model sweeps and diagnostics (`diagnostics/model results/`)

- **Per-K silhouette curves** (CSV + figures):
  - `kmeans_silhouette_vs_k.csv`
  - `agglomerative_silhouette_vs_k.csv`
  - `gmm_silhouette_vs_k.csv`
  - `birch_silhouette_vs_k.csv`

- **GMM information criteria grid**:
  - `gmm_bic_aic.csv` – BIC/AIC values for combinations of `K` and covariance types.

- **DBSCAN sweeps**:
  - `dbscan_sweep.csv` – single-parameter eps sweep (silhouette on core points).
  - `dbscan_multi_sweep.csv` – expanded grid over `eps` and `min_samples`, including:
    - `n_clusters`
    - `noise_rate`
    - `silhouette_core`
    - `combined = silhouette_core * (1 - noise_rate)`

### 5.3. Cluster validity summaries (`diagnostics/cluster validity/`)

- **`cluster_internal_validity.csv`**  
  One row per algorithm with internal metrics.

- **`cluster_external_validity.csv`** *(if `sex` is available)*  
  Rand and Jaccard vs `sex` for each label column.

### 5.4. Figures (`figures/`)

Some key figures (non-exhaustive):

- **PCA and LDA**:
  - `gmm/BIC/pca_explained_variance.png`
  - `gmm/BIC/pca_loadings_heatmap.png`
  - `gmm/BIC/lda/lda_loadings_heatmap.png`

- **Silhouette vs K**:
  - `kmeans/kmeans_silhouette_vs_k.png`
  - `agglomerative/agglomerative_silhouette_vs_k.png`
  - `gmm/gmm_silhouette_vs_k.png`
  - `birch/birch_silhouette_vs_k.png`

- **GMM BIC/AIC curves**:
  - `gmm/BIC/gmm_bic_vs_k.png`
  - `gmm/AIC/gmm_aic_vs_k.png`

- **DBSCAN diagnostics**:
  - `dbscan/dbscan_silhouette_vs_eps.png`
  - `dbscan/dbscan_combined_vs_eps.png`
  - `dbscan/dbscan_noise_vs_eps.png`
  - `dbscan/dbscan_kdistance.png` – k-distance plot with elbow and selected `eps` lines.

- **2D embeddings and scatter plots**:
  - PCA scatter plots for each algorithm, e.g. `kmeans/kmeans_pca.png`, `gmm/gmm_pca.png`, etc.
  - Additional PCA/LDA/TSNE/UMAP plots for the GMM best-by-BIC model:
    - `gmm/BIC/gmm_bic_best_pca.png`
    - `gmm/BIC/gmm_bic_best_lda.png`
    - `gmm/BIC/gmm_bic_best_tsne.png`
    - `gmm/BIC/gmm_bic_best_umap.png` *(UMAP only if `umap-learn` is installed)*

- **Cluster validity heatmaps**:
  - `cluster validity/internal_validity_summary.png`
  - `cluster validity/internal_validity_heatmap.png`
  - `cluster validity/external_validity_summary.png`
  - `cluster validity/external_validity_heatmap.png`

- **Cluster profiles & interpretability** (for GMM best-by-BIC):
  - `gmm/BIC/gmm_bic_feature_zmean_by_cluster.png` – z-mean heatmap of features by cluster.
  - `gmm/BIC/cluster_cards/cluster_card_*.png` – per-cluster “cards” showing top features (by z-score).
  - Ellipse-based views:
    - `gmm/BIC/gmm_bic_accuracy_vs_rt_ellipses.png`
    - `gmm/BIC/gmm_bic_accuracy_vs_var_rt.png`
    - `gmm/BIC/gmm_bic_total_correct_vs_var_rt.png`

- **Best-overall clustering t-SNE**:
  - e.g. `kmeans/kmeans_tsne.png`, `gmm/gmm_tsne.png`, or `tsne_best_overall.png` depending on which algorithm wins.

### 5.5. Text report

- **`diagnostics/_clustering_report.txt`**  
  Human-readable summary with:
  - Best K and silhouette for KMeans, Agglomerative, Birch, GMM.  
  - DBSCAN best eps/min_samples (by silhouette and by combined score).  
  - GMM BIC/AIC best models.  
  - Overall best algorithm and its key parameters.

---

## 6. Subject-wise analysis by cluster

After running `cluster_knowledge_trajectories.py` (so that `diagnostics/student cluster labels/student_clusters.csv` exists), you can analyse subject-wise marks by cluster.

### 6.1. Default run

From `Study-2`:

```bash
python subjectwise_by_cluster.py
```

This uses:

- **Cluster labels**: `diagnostics/student cluster labels/student_clusters.csv`  
  (expects column `gmm_bic_best_label` and an ID column such as `IDCode`).

- **Subject-wise marks**: `data/EQTd_DAi_25_cleaned 3_1 for Prince.xlsx`

It will:

1. Read student cluster labels and detect the ID column.  
2. Read the Excel marks file, detect ID and subject columns (`s1`, `s2`, ...), and an optional total column.  
3. Merge marks with cluster labels on the ID.  
4. Write a merged CSV to:
   - `diagnostics/cluster input features/marks_with_clusters.csv`
5. Produce subject-by-cluster visualisations in:
   - `figures/subjectwise by cluster/`

### 6.2. Custom marks file

You can provide a custom Excel path:

```bash
python subjectwise_by_cluster.py path\to\your_marks.xlsx
```

The script will:

- Use the same cluster labels CSV (`diagnostics/student cluster labels/student_clusters.csv`).
- Read `your_marks.xlsx`, auto-detect ID and subject columns, merge, and regenerate the figures.

### 6.3. Subject-level outputs

Under `figures/subjectwise by cluster/` you will get, for example:

- **`subject_zmean_by_cluster.png`**  
  Heatmap of z-standardised subject scores by cluster. Each row is a cluster; each column is a subject.

- **`subject_radar_all_clusters.png`**  
  A single radar plot comparing subject z-means across all clusters simultaneously.

These help interpret which subject areas distinguish clusters.

---

## 7. Typical workflow

1. **Prepare itemwise response data** in the required long format and place it under `data/` (or another folder).
2. **Run the clustering pipeline**:
   - `python cluster_knowledge_trajectories.py`  
     or  
   - `python cluster_knowledge_trajectories.py path\to\your_itemwise.csv`
3. Inspect:
   - `diagnostics/cluster input features/derived_features.csv`
   - `diagnostics/student cluster labels/student_clusters.csv`
   - `diagnostics/cluster validity/*.csv`
   - `figures/*` for visual diagnostics.
4. **(Optional) Link to subject marks**:
   - Ensure `diagnostics/student cluster labels/student_clusters.csv` is present.  
   - Run `python subjectwise_by_cluster.py` (or provide your own marks file path).
   - Inspect subject-level heatmaps and radar plots.
5. Use the combination of numeric diagnostics and figures to interpret and select meaningful student knowledge trajectory clusters.

---

## 8. Customisation notes

You can customise the analysis by editing the Python scripts:

- **Feature set**:  
  Modify `feature_cols` in `main()` of `cluster_knowledge_trajectories.py` to include or exclude features.

- **K ranges for clustering**:  
  Change the `k_range` arguments in functions like `kmeans_sweep`, `agglomerative_sweep`, `gmm_sweep`, and `birch_sweep`.

- **DBSCAN search grid**:  
  Adjust `eps_values` and `min_samples_list` in the `dbscan_grid_diagnostics` / `dbscan_multi_grid_diagnostics` calls.

- **DBSCAN noise threshold**:  
  `noise_threshold` (default `0.20`) controls whether the best-silhouette or best-combined DBSCAN configuration is chosen.

- **External validity label**:  
  By default, `sex` is used if available. You can modify `compute_external_validity` and the call sites to use different true labels (e.g. grade bands, school type).

Whenever you change the scripts, re-run the pipelines to regenerate diagnostics and figures.
