# Unsupervised Clustering Module

**Paper:** *Evidence-Grounded LLM Summarization for Actionable Student Feedback Analysis*

This folder contains the **unsupervised clustering component** of the paper.
It covers data augmentation, multi-encoder embedding fusion, density-based clustering,
and all associated visualizations (Figure 4, Table 4).

> **Scope:** This module is responsible for the unsupervised analysis of student feedback.
> The supervised classification pipeline (attention fusion + KMeans classifier) is maintained
> by co-authors in a separate module.

---

## Pipeline

```
dataset.csv  (959 samples, 10 categories)
     │
     ├─► [§2.4]   Data Augmentation         3 methods → 671 → 2013 training texts
     │
     ├─► [§3.3.1] Embedding Generation      5 sentence encoders → pre-computed NPZ files
     │
     ├─► [§3.3.2] Embedding Fusion          4 strategies (mean / weighted / concat / concat+PCA)
     │
     ├─► [§3.3.3] HDBSCAN Clustering        density-based, no cluster count required
     │
     └─► [§4.2]   Evaluation & Visualization Table 4 (metrics) + Figure 4 (t-SNE)
```

---

## Repository Layout

```
clustering/
├── unsupervised_clustering_pipeline.ipynb  ← main walkthrough (start here)
└── src/
    ├── augmentation.py   — data augmentation generation + semantic fidelity validation
    ├── embeddings.py     — 5-model embedding generation and NPZ loading utilities
    ├── fusion.py         — 4 embedding fusion strategies
    ├── clustering.py     — HDBSCAN clustering + Table 4 metrics
    └── visualization.py  — t-SNE dimensionality reduction + Figure 4 (SVG/PNG)
```

> **Data files** (not included in this repo due to size/privacy): place `dataset.csv` and
> the pre-computed `tsne_stage3_Test_individual.npz` / `tsne_stage3_Train_individual.npz`
> archives in a `data/embeddings/` folder relative to this directory.

---

## Methods

### §2.4 — Data Augmentation

The training set (671 samples) was expanded 3× using three complementary methods
to improve model robustness and address class imbalance:

| Method | Implementation | aug_p |
|---|---|---|
| **Synonym Replacement** | WordNet lexical substitution | 0.10 |
| **Back-Translation** | Word-order swap (syntactic proxy) | 0.10 |
| **LLM Paraphrasing** | Higher-rate synonym substitution | 0.20 |

**Semantic fidelity** was validated using `sentence-transformers/multi-qa-mpnet-base-dot-v1`
on 80 randomly selected feedback texts (cosine similarity threshold = 0.75):

| Method | Mean sim. | Std | % > 0.75 |
|---|---|---|---|
| Synonym Replacement | 0.93 | ±0.10 | 91.2% |
| Back-Translation | 0.91 | ±0.08 | 95.0% |
| LLM Paraphrasing | 0.88 | ±0.13 | 86.2% |

### §3.3.1 — Five Sentence Encoder Models

| Model | Dim | HuggingFace ID |
|---|---|---|
| GTE-Large | 1024 | `thenlper/gte-large` |
| BGE-Large | 1024 | `BAAI/bge-large-en-v1.5` |
| MXBai-Embed-Large | 1024 | `mixedbread-ai/mxbai-embed-large-v1` |
| Instructor-XL | 768 | `hkunlp/instructor-xl` |
| Multi-QA-MPNet | 768 | `sentence-transformers/multi-qa-mpnet-base-dot-v1` |

All embeddings are ℓ2-normalised. Pre-computed archives are provided in `data/embeddings/`.

### §3.3.2 — Embedding Fusion Strategies

| Strategy | Description | Output dim |
|---|---|---|
| **Mean Ensemble** | Element-wise average of zero-padded embeddings | 1024 |
| **Weighted Ensemble** | Performance-weighted average (weights 1.3, 1.2, 1.2, 1.0, 1.0) | 1024 |
| **Concatenation** | Direct feature-level concatenation | 4608 |
| **Concat + PCA** | Concatenation + PCA retaining ≥ 66.6% cumulative variance | 47–53 |

PCA criterion: `sklearn.PCA(n_components=0.666)` automatically selects the minimum number
of components to retain 66.6% of explained variance (47 components on the full dataset).

### §3.3.3 — HDBSCAN Clustering

HDBSCAN discovers intrinsic semantic structure without requiring the number of clusters
to be specified in advance. Configured with:

```python
min_cluster_size = 10
min_samples      = 5
metric           = 'euclidean'
```

HDBSCAN is fully **deterministic** — given identical hyperparameters and embeddings,
it produces identical results on every run. Variance estimates are not applicable.

### §4.2 — Results (Table 4 + Figure 4)

**Table 4 — Clustering metrics on the full 959-sample dataset:**

| Fusion Strategy | Silhouette↑ | Davies–Bouldin↓ | Calinski–Harabasz↑ | Clusters | Noise |
|---|---|---|---|---|---|
| Average Ensemble | 0.238 | 1.369 | 24.12 | 8 | 612 (63.7%) |
| Weighted Ensemble | 0.259 | 1.287 | 25.67 | 9 | 564 (58.7%) |
| Concatenation | 0.219 | 1.521 | 22.34 | 11 | 721 (75.1%) |
| **Concat + PCA** | **0.271** | **1.241** | **26.94** | **10** | **412 (42.9%)** |

†HDBSCAN is deterministic given fixed hyperparameters and embedding representations;
variance estimates are therefore not applicable for clustering metrics.

**Concat+PCA** is the best-performing strategy across all metrics, recovering all 10 semantic
feedback categories as distinct clusters in the t-SNE visualization (Figure 4).

---

## Setup

```bash
# From the repository root
pip install -r requirements.txt

# Download NLTK WordNet data (one-time)
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## How to Run

### Option A — Notebook (recommended)

Open `unsupervised_clustering_pipeline.ipynb` for a complete end-to-end walkthrough.

```bash
jupyter notebook clustering/unsupervised_clustering_pipeline.ipynb
```

### Option B — Python modules

Each `src/` module can be run directly as a standalone script:

```bash
# Inspect pre-computed embedding files
python clustering/src/embeddings.py

# Run embedding fusion + show output shapes
python clustering/src/fusion.py

# Run HDBSCAN + print Table 4
python clustering/src/clustering.py

# Validate augmentation semantic fidelity
python clustering/src/augmentation.py

# Regenerate Figure 4 SVG/PNG files (~5 min)
python clustering/src/visualization.py
```

> **Note:** `src/visualization.py` loads all 959 samples, runs HDBSCAN, and then t-SNE.
> Expect ~5–10 minutes on CPU.

---

## Dependencies

| Package | Role |
|---|---|
| `hdbscan` | Density-based unsupervised clustering |
| `scikit-learn` | PCA, t-SNE, clustering metrics |
| `sentence-transformers` | Augmentation validation encoder |
| `matplotlib` | Figure 4 generation |
| `nltk` | WordNet synonym replacement |
| `numpy`, `pandas` | Numerical and data utilities |
