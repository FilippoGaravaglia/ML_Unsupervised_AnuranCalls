# ML Unsupervised — Anuran Calls

Unsupervised Machine Learning project based on the **UCI Anuran Calls (MFCCs)** dataset.

The project implements a **custom K-Means clustering algorithm from scratch in Python**, evaluates the quality of the resulting clusters, selects the number of clusters through internal clustering metrics, and compares the custom implementation with both **scikit-learn** and **WEKA**.

---

## Project Goal

The goal of this project is to identify natural groups in acoustic frog-call data using an **unsupervised learning** approach.

Unlike classification or regression, no target variable is used during clustering.

The model receives only the acoustic features:

- `MFCCs_1`
- `MFCCs_2`
- ...
- `MFCCs_22`

and tries to group observations according to their similarity.

The biological labels:

- `Family`
- `Genus`
- `Species`

are excluded from the clustering process and are used only afterwards for **post-hoc interpretation** of the discovered clusters.

---

## Dataset

The project uses the **Anuran Calls (MFCCs)** dataset from the UCI Machine Learning Repository.

Dataset characteristics:

- **7,195 observations**
- **22 numerical MFCC features**
- **4 biological families**
- **8 genera**
- **10 species**
- **No missing values**
- **No duplicated rows**

Each observation represents a frog-call syllable through a vector of **Mel-Frequency Cepstral Coefficients (MFCCs)**.

MFCCs are numerical descriptors extracted from audio signals and can be considered a compact acoustic representation of each frog call.

The original dataset also contains:

- `Family`
- `Genus`
- `Species`
- `RecordID`

These columns are not used as input for K-Means.

---

## Repository Structure

```text
ML_Unsupervised_AnuranCalls/
├── data/
│   ├── raw/
│   │   ├── Frogs_MFCCs.csv
│   │   └── Readme.txt
│   │
│   └── processed/
│       ├── features_standardized.csv
│       ├── features_standardized.arff
│       ├── labels.csv
│       └── preprocessing_metadata.json
│
├── results/
│   ├── eda/
│   │   ├── mfcc_correlation_matrix.png
│   │   ├── mfcc_feature_boxplots.png
│   │   └── mfcc_feature_variance.png
│   │
│   ├── cluster_selection/
│   │   ├── elbow_curve.png
│   │   ├── silhouette_curve.png
│   │   └── k_selection_results.csv
│   │
│   ├── final_clustering/
│   │   ├── centroids.csv
│   │   ├── cluster_assignments.csv
│   │   ├── cluster_vs_family.csv
│   │   ├── cluster_vs_genus.csv
│   │   └── cluster_vs_species.csv
│   │
│   ├── sklearn/
│   │   ├── centroids.csv
│   │   ├── cluster_assignments.csv
│   │   ├── cluster_vs_family.csv
│   │   └── metrics.csv
│   │
│   └── weka/
│
├── scripts/
│   ├── analyze_features.py
│   ├── evaluate_final_clustering.py
│   ├── evaluate_sklearn_clustering.py
│   ├── explore_dataset.py
│   ├── inspect_kmeans_convergence.py
│   ├── plot_feature_analysis.py
│   ├── prepare_dataset.py
│   ├── run_kmeans_model.py
│   └── select_number_of_clusters.py
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── preprocessing.py
│   │   └── schema.py
│   │
│   ├── evaluation/
│   │   └── cluster_selection.py
│   │
│   └── models/
│       └── kmeans.py
│
├── tests/
│   └── test_kmeans.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Exploratory Data Analysis

Before implementing K-Means, the 22 MFCC features were analyzed to understand their statistical properties.

The analysis included:

- mean
- standard deviation
- variance
- minimum and maximum values
- detection of constant features
- feature correlation analysis
- feature distribution visualization

### Main Findings

No feature has zero variance.

`MFCCs_1` has the lowest variance and is almost constant for many observations, but it still contains some variability and was therefore retained.

Only one feature pair exceeded the selected absolute correlation threshold of `0.90`:

```text
MFCCs_13 <-> MFCCs_15 = -0.9046
```

This indicates a strong inverse linear relationship between the two features.

Since no widespread redundancy was found, all 22 MFCC features were retained.

The feature distributions showed different levels of dispersion, which is particularly important for K-Means because the algorithm relies on distances between observations.

---

## Preprocessing

The clustering input was separated from biological metadata.

The following columns were excluded from K-Means:

```text
Family
Genus
Species
RecordID
```

Only the 22 MFCC features were used for clustering.

All features were standardized using `StandardScaler`.

After preprocessing, every feature has approximately:

```text
Mean = 0
Standard deviation = 1
```

This prevents features with naturally larger dispersion from dominating the Euclidean distance calculation.

The processed clustering matrix contains:

```text
7,195 observations
22 standardized features
```

The biological labels were stored separately and used only for post-hoc analysis.

---

## Custom K-Means Implementation

K-Means was implemented from scratch in:

```text
src/models/kmeans.py
```

The algorithm performs the following steps:

1. Randomly select `K` observations as initial centroids.
2. Compute the Euclidean distance between every observation and every centroid.
3. Assign each observation to its nearest centroid.
4. Recalculate each centroid as the mean of the observations assigned to that cluster.
5. Measure how much the centroids moved.
6. Repeat until convergence or until the maximum number of iterations is reached.

The implementation exposes:

```text
centroids_
labels_
inertia_
n_iterations_
converged_
centroid_shift_history_
```

### Convergence

Convergence is detected by measuring the maximum centroid movement between consecutive iterations.

If:

```text
maximum centroid shift <= tolerance
```

the algorithm is considered converged.

A maximum number of iterations is also enforced as a safety mechanism.

---

## Empty Cluster Handling

If a cluster receives no observations during an iteration, its centroid cannot be computed as an average.

Instead of reducing the number of clusters, the implementation reinitializes that centroid using another observation from the dataset.

This keeps the requested value of `K` unchanged throughout the algorithm.

---

## Initial Smoke Test

Before applying K-Means to the real dataset, the implementation was tested on a simple synthetic dataset containing two clearly separated groups.

The custom implementation correctly identified:

```text
Cluster 1 centered around [1, 1]
Cluster 2 centered around [8, 8]
```

with:

```text
Iterations: 2
Inertia: approximately 0.08
```

This validated the basic assignment, centroid update, and convergence logic.

---

## Unit Tests

The custom implementation is covered by **10 unit tests**.

The tests verify:

- identification of clearly separated clusters
- correct centroid computation
- convergence
- non-negative inertia
- error when predicting before fitting
- invalid cluster count handling
- cluster count greater than number of samples
- invalid input dimensionality
- non-finite input handling
- prediction of cluster labels for new observations

Test result:

```text
10 passed
```

---

## Selecting the Number of Clusters

The number of clusters was not chosen using the biological labels.

Instead, values of:

```text
K = 2 ... 10
```

were evaluated using only the standardized MFCC features.

For each value of `K`, the custom K-Means algorithm was executed with:

```text
10 random initializations
```

and the solution with the lowest inertia was retained.

This resulted in:

```text
9 candidate values of K
×
10 random initializations
=
90 custom K-Means runs
```

Two internal clustering metrics were used:

- Inertia
- Silhouette Score

### Results

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 108667.951258 | 0.334677 |
| 3 | 94587.598255 | **0.360221** |
| 4 | 84679.208108 | 0.355365 |
| 5 | 76362.032032 | 0.262988 |
| 6 | 70837.978697 | 0.267440 |
| 7 | 61997.964380 | 0.289797 |
| 8 | 59741.847942 | 0.306792 |
| 9 | 54855.078374 | 0.302366 |
| 10 | 51995.326577 | 0.258147 |

The inertia curve did not show a uniquely clear elbow.

The highest silhouette score was obtained with:

```text
K = 3
Silhouette ≈ 0.3602
```

Therefore, `K = 3` was selected for the final clustering.

The biological labels were not used during this selection.

---

## Final Custom K-Means Model

The final custom clustering was executed with:

```text
K = 3
Random initializations = 10
Maximum iterations = 300
Tolerance = 1e-4
```

The best final run produced:

```text
Best seed: 4
Iterations: 7
Converged: True
Inertia: 94587.598255
Silhouette score: 0.356451
```

### Final Cluster Sizes

```text
Cluster 0: 626 observations
Cluster 1: 3574 observations
Cluster 2: 2995 observations
```

The cluster numbers are arbitrary labels and do not represent an intrinsic ordering.

---

## Post-Hoc Biological Interpretation

After the clustering was completed, the discovered clusters were compared with the biological `Family` labels.

These labels were never used to train or select the model.

### Cluster vs Family

| Cluster | Bufonidae | Dendrobatidae | Hylidae | Leptodactylidae |
|---:|---:|---:|---:|---:|
| 0 | 5 | 0 | 594 | 27 |
| 1 | 0 | 0 | 111 | 3463 |
| 2 | 63 | 542 | 1460 | 930 |

The result shows that:

- one cluster is strongly associated with `Hylidae`
- one cluster is strongly associated with `Leptodactylidae`
- the third cluster contains a mixture of multiple biological families

This indicates that the acoustic MFCC features capture part of the biological structure of the dataset, but the natural geometry of the acoustic data does not perfectly reproduce the biological taxonomy.

### External Agreement Metrics

```text
Adjusted Rand Index: 0.402223
Normalized Mutual Information: 0.385627
```

These values indicate a partial but not perfect correspondence between the unsupervised clusters and the real family labels.

---

## Comparison with scikit-learn

The custom implementation was compared with:

sklearn.cluster.KMeans

The scikit-learn benchmark used:

```text
K = 3
init = random
10 random initializations
max_iter = 300
tol = 1e-4
algorithm = Lloyd
```

The same standardized dataset was used.

### scikit-learn Results

```text
Best seed: 1
Iterations: 5
Inertia: 94587.598255
Silhouette score: 0.356451

Family ARI: 0.402223
Family NMI: 0.385627
```

Cluster sizes:

```text
3574
626
2995
```

The ordering of the cluster IDs differs from the custom implementation, but this is irrelevant because cluster labels are arbitrary.

### Custom vs scikit-learn Agreement

The Adjusted Rand Index between the custom cluster assignments and scikit-learn cluster assignments is:

```text
ARI = 1.000000
```

This indicates a perfect agreement between the two clustering solutions, independently of cluster numbering.

---

## Comparison with WEKA

The clustering was also replicated using:

```text
WEKA SimpleKMeans
```

The following configuration was used:

```text
Number of clusters: 3
Initialization: Random
Distance: Euclidean
Maximum iterations: 300
Seeds tested: 0 ... 9
```

Because the features had already been standardized in Python, WEKA's internal distance normalization was disabled:

```text
dontNormalize = True
```

This ensured that WEKA used the same standardized feature space as the custom implementation and scikit-learn.

Ten independent random initializations were evaluated and the solution with the smallest within-cluster sum of squared errors was retained.

The best WEKA solution produced approximately:

```text
Within-cluster SSE: 94587.5981
Cluster sizes:
3574
626
2995
```

This is effectively equivalent to:

```text
Custom inertia:       94587.598255
scikit-learn inertia: 94587.598255
WEKA SSE:             94587.598149
```

The extremely small numerical difference is attributable to implementation-level floating-point details.

The final cluster sizes are the same across the three implementations, except for arbitrary cluster ID ordering.

---

## Final Comparison

| Implementation | K | Inertia / SSE | Silhouette | Cluster Sizes |
|---|---:|---:|---:|---|
| Custom K-Means | 3 | 94587.598255 | 0.356451 | 626 / 3574 / 2995 |
| scikit-learn | 3 | 94587.598255 | 0.356451 | 3574 / 626 / 2995 |
| WEKA SimpleKMeans | 3 | ~94587.598149 | — | 3574 / 626 / 2995 |

The custom and scikit-learn cluster assignments have:

```text
Adjusted Rand Index = 1.0
```

The three implementations therefore converge to the same underlying clustering structure.

---

## Main Conclusions

The project demonstrates the complete workflow of an unsupervised clustering problem:

```text
Raw dataset
↓
Dataset inspection
↓
Feature analysis
↓
Removal of labels and identifier from clustering input
↓
Feature standardization
↓
Custom K-Means implementation
↓
Convergence validation
↓
Unit testing
↓
Selection of K using inertia and silhouette
↓
Final clustering with K = 3
↓
Post-hoc biological interpretation
↓
Comparison with scikit-learn
↓
Comparison with WEKA
```

The main conclusions are:

1. The 22 MFCC features contain meaningful acoustic structure.
2. The feature scales and dispersions justify standardization before distance-based clustering.
3. The Elbow Method alone does not provide a clearly identifiable optimal value of `K`.
4. Silhouette analysis identifies `K = 3` as the strongest candidate among the tested values.
5. The final three clusters show meaningful but incomplete correspondence with frog biological families.
6. Two clusters are strongly associated with individual families, while the third contains a mixture of families.
7. The custom K-Means implementation produces the same final clustering as scikit-learn.
8. WEKA SimpleKMeans produces essentially the same clustering and within-cluster squared error.
9. The agreement with established implementations validates the custom algorithm.

---

## Technologies

- Python 3.14
- NumPy
- pandas
- matplotlib
- scikit-learn
- pytest
- WEKA 3.8.7

---

## Installation

Create and activate a Python virtual environment.

Example:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

---

## Reproducing the Project

### 1. Inspect the raw dataset

```bash
python -m scripts.explore_dataset
```

### 2. Analyze the MFCC features

```bash
python -m scripts.analyze_features
```

### 3. Generate EDA plots

```bash
python -m scripts.plot_feature_analysis
```

### 4. Prepare the clustering dataset

```bash
python -m scripts.prepare_dataset
```

### 5. Run the custom K-Means smoke test

```bash
python -m scripts.run_kmeans_model
```

### 6. Inspect convergence

```bash
python -m scripts.inspect_kmeans_convergence
```

### 7. Run unit tests

```bash
pytest -v
```

Expected result:

```text
10 passed
```

### 8. Select the number of clusters

```bash
python -m scripts.select_number_of_clusters
```

### 9. Run the final custom clustering

```bash
python -m scripts.evaluate_final_clustering
```

### 10. Run the scikit-learn benchmark

```bash
python -m scripts.evaluate_sklearn_clustering
```

### 11. Run the WEKA benchmark

Load:

```text
data/processed/features_standardized.arff
```

in WEKA Explorer and configure:

```text
Clusterer: SimpleKMeans
K: 3
Initialization: Random
Distance: EuclideanDistance
dontNormalize: True
maxIterations: 300
Seeds: 0 ... 9
```

Retain the run with the minimum within-cluster sum of squared errors.

---

## Methodological Notes

The biological labels are deliberately excluded from:

- preprocessing decisions related to clustering structure
- K-Means training
- selection of the number of clusters

They are only introduced after the final clustering for external interpretation.

This preserves the unsupervised nature of the experiment.

The number of clusters is not inferred from the known number of biological families.

Although the dataset contains four `Family` values, the selected value is:

```text
K = 3
```

because it is derived from the internal structure of the standardized MFCC features rather than from the known taxonomy.

---

## Author

**Filippo Garavaglia**

Machine Learning project developed for academic study and experimental comparison of unsupervised clustering implementations.