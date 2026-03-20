# EduFeedback-RAG

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/repo-EduFeedback--RAG-181717?logo=github)](https://github.com/nubcico/EduFeedback)

---

## Abstract

Analyzing large-scale student feedback is critical for higher education quality assurance, yet manual analysis is inefficient and subjective. This paper proposes an integrated framework — **EduFeedback-RAG** — that unifies supervised classification, unsupervised clustering, and Retrieval-Augmented Generation (RAG) to produce evidence-grounded and actionable insights. Ensemble-based supervised models perform thematic classification, while multi-encoder embedding fusion enables unsupervised discovery of coherent feedback clusters. A multi-stage RAG module integrates category predictions and cluster structure to retrieve representative evidence and generate transparent summaries with citation traceability.

The framework is evaluated on student feedback collected from a Central Asian university and two public benchmarks, EduRABSA and Coursera course reviews, covering seven thematic categories. The supervised ensemble achieves **83.1% accuracy** and **0.82 Macro-F1**, while unsupervised clustering attains a **silhouette score of 0.271** (Concat+PCA). Cross-dataset evaluation demonstrates robust generalization, with ensemble accuracy exceeding 80% on both external benchmarks.

---

## Repository Structure

```
EduFeedback/
├── README.md                                    ← you are here
├── requirements.txt                             ← unified dependency list
│
├── clustering/
│   ├── README.md                                ← unsupervised module documentation
│   ├── unsupervised_clustering_pipeline.ipynb   ← end-to-end clustering walkthrough
│   └── src/
│       ├── __init__.py
│       ├── augmentation.py    — text augmentation (synonym replace, back-translation, LLM para)
│       ├── embeddings.py      — 5-encoder sentence embedding generation and NPZ utilities
│       ├── fusion.py          — 4 embedding fusion strategies (mean / weighted / concat / concat+PCA)
│       ├── clustering.py      — HDBSCAN + Table 4 evaluation metrics
│       └── visualization.py   — t-SNE reduction + Figure 4 (SVG/PNG output)
│
└── rag_evaluation/
    ├── README.md                                ← RAG evaluation documentation
    └── rag_evaluation.ipynb                     ← faithfulness & cluster coverage ablation
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/nubcico/EduFeedback.git
cd EduFeedback

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Download NLTK WordNet data (one-time, required for augmentation module)
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## Quick Start

| Notebook | Description |
|---|---|
| [`clustering/unsupervised_clustering_pipeline.ipynb`](clustering/unsupervised_clustering_pipeline.ipynb) | Full unsupervised pipeline: augmentation → embeddings → fusion → HDBSCAN → t-SNE |
| [`rag_evaluation/rag_evaluation.ipynb`](rag_evaluation/rag_evaluation.ipynb) | RAG ablation study: Baseline vs Supervised vs Supervised+Unsupervised RAG |

```bash
# Launch Jupyter and open either notebook
jupyter notebook
```

---

## Module Descriptions

### Clustering Module (`clustering/`)

The clustering module implements the full unsupervised analysis pipeline from the paper:

| Component | Paper Section | File |
|---|---|---|
| Data Augmentation | §2.4 | `src/augmentation.py` |
| Sentence Embedding Generation | §3.3.1 | `src/embeddings.py` |
| Embedding Fusion (4 strategies) | §3.3.2 | `src/fusion.py` |
| HDBSCAN Clustering + Metrics | §3.3.3 | `src/clustering.py` |
| t-SNE Visualization (Figure 4) | §4.2 | `src/visualization.py` |

**Five sentence encoders:** GTE-Large, BGE-Large, MXBai-Embed-Large, Instructor-XL, Multi-QA-MPNet

**Four fusion strategies:** Mean Ensemble (1024d), Weighted Ensemble (1024d), Concatenation (4608d), Concat+PCA (47d)

**Best strategy:** Concat+PCA — Silhouette 0.271, Davies-Bouldin 1.241, 10 clusters, 42.9% noise

### RAG Evaluation Module (`rag_evaluation/`)

The RAG evaluation notebook benchmarks three retrieval configurations using two
evidence-grounded metrics (no ground-truth summaries required):

- **Faithfulness** — proportion of generated statements supported by retrieved evidence
- **Cluster Coverage** — proportion of HDBSCAN clusters represented in retrieved evidence

---

## Results

### Unsupervised Clustering (Table 4)

| Fusion Strategy | Silhouette↑ | Davies–Bouldin↓ | Calinski–Harabasz↑ | Clusters | Noise |
|---|---|---|---|---|---|
| Average Ensemble | 0.238 | 1.369 | 24.12 | 8 | 612 (63.7%) |
| Weighted Ensemble | 0.259 | 1.287 | 25.67 | 9 | 564 (58.7%) |
| Concatenation | 0.219 | 1.521 | 22.34 | 11 | 721 (75.1%) |
| **Concat + PCA** | **0.271** | **1.241** | **26.94** | **10** | **412 (42.9%)** |

### RAG Summarization Ablation (Table 1)

| Model | Faithfulness ↑ | Cluster Coverage ↑ |
|---|---|---|
| Baseline RAG | 0.72 | 0.40 |
| Supervised RAG | 0.80 | 0.56 |
| **Supervised + Unsupervised RAG (Ours)** | **0.87** | **0.76** |

### Supervised Classification (Table 2)

| Model | Primary Acc | EduRABSA Acc | Coursera Acc |
|---|---|---|---|
| TF-IDF + LR/SVM | 77.1% | 73.4% | 74.8% |
| SetFit | 81.3% | 78.2% | 79.5% |
| DeBERTa-LoRA | 83.3% | 80.1% | 81.4% |
| **Ensemble (Ours)** | **83.1%** | **80.8%** | **82.1%** |

*Mean ± std over 5-fold cross-validation. Full table with F1 scores in the paper.*

---

## Data

> **Note:** The student feedback dataset is not included in this repository due to privacy and institutional considerations. We plan to release a de-identified version in a future update.
>
> Pre-computed embedding archives (`tsne_stage3_Test_individual.npz`, `tsne_stage3_Train_individual.npz`) covering all 959 samples across five encoders will be made available separately.

---

## Citation

If you use this code or find this work useful, please cite:

```bibtex
@article{baimukanova2026edufeedback,
  title   = {Evidence-Grounded LLM Summarization for Actionable Student Feedback Analysis},
  author  = {Baimukanova, Zhanerke and Saparbekov, Yerassyl and Ha, Hyesong and Lee, Minho},
  journal = {Information},
  year    = {2026},
  note    = {Manuscript ID: information-4180445}
}
```

---

## Contact

Department of Computer Science, Nazarbayev University, Astana, Kazakhstan
