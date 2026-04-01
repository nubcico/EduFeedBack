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

## Table 5. Evaluation of RAG summarization strategies (ablation study)

| Model | Faithfulness ↑ | Cluster Coverage ↑ |
|-------|---------------|-------------------|
| Baseline RAG | 0.71 | 0.59 |
| Supervised RAG | 0.89 | 0.60 |
| **Supervised + Unsupervised RAG (Ours)** | **0.93** | **0.64** |

---

## Table 6. Comparison of RAG summaries under high and moderate supervised–unsupervised agreement

| Component | Example 1: High agreement | Example 2: Split clusters |
|-----------|--------------------------|--------------------------|
| **Category** | Teaching Quality (supervised ensemble) | General Comments & Satisfaction (supervised ensemble) |
| **Samples analyzed** | 105 total → 10 via RAG (top-k) | 210 total → 10 via RAG (top-k) |
| **Cluster mapping** | 84% → C0 (96.9% purity) | 41% → C1 (90.8% purity); 32% → C3; 27% noise |
| **Agreement level** | High (C0 predominantly Teaching Quality) | Moderate (category spans multiple clusters) |
| **Top keywords** | *lectures, clarity, examples, pacing, difficult, fast* | *overall, satisfied, workload, balance, experience* |
| **Generated summary** | Lecture clarity and pacing issues dominate, with **68% negative sentiment**. Difficulty following explanations (**84%**), rapid pacing (**76%**), and insufficient worked examples (**62%**) were frequent. Actions include **3–4 worked examples per lecture**, **pause intervals**, **scaffolding materials**, and **reduced content density**. | Feedback is heterogeneous: **58% overall satisfaction**, with helpfulness (45%) and challenging-but-rewarding perceptions (38%). Concerns include workload/resources (32%) and work–life balance (27%). Outputs include both actionable items and diagnostic flags for manual review. |
| **Statistics provided** | 4 indicators (68%, 84%, 76%, 62%) | 6 indicators (58%, 45%, 38%, 32%, 42%, 35%) |
| **Actionability** | 4 quantified recommendations | Mixed: actionable + diagnostic |
| **Traceability** | Category: Teaching Quality; Cluster: C0; IDs: FB_0023, FB_0087, FB_0134, FB_0156, FB_0189, FB_0203, FB_0267, FB_0289, FB_0312, FB_0378 | Category: General Comments; Clusters: C1, C3, noise; IDs: FB_0156, FB_0189, FB_0234, FB_0278, FB_0301, FB_0345, FB_0389, FB_0412, FB_0456, FB_0489 |

---

## Table 3. Supervised learning performance across datasets (mean ± std over 5-fold CV)

| Model | Primary Acc | Primary F1 | EduRABSA Acc | EduRABSA F1 | Coursera Acc | Coursera F1 |
|-------|------------|-----------|-------------|------------|-------------|------------|
| *Classical Models* | | | | | | |
| TF-IDF + LR/SVM | 80.0 (±2.7) | 0.798 | 72.9 (±0.9) | 0.700 | 48.3 (±1.6) | 0.492 |
| Hybrid (TF-IDF + Emb) | 82.2 (±2.2) | 0.821 | 74.4 (±1.1) | 0.715 | 49.4 (±0.9) | 0.519 |
| *Neural Models* | | | | | | |
| DTLP | 74.5 (±3.4) | 0.746 | 59.4 (±0.7) | 0.546 | 47.5 (±1.0) | 0.402 |
| MHAF | 71.6 (±1.9) | 0.714 | 61.5 (±0.7) | 0.547 | 48.8 (±0.8) | 0.384 |
| *Transformer Models* | | | | | | |
| SetFit | 84.5 (±2.5) | 0.846 | 80.2 (±1.1) | 0.774 | 49.3 (±0.7) | 0.469 |
| DeBERTa-LoRA | 83.3 (±1.2) | 0.820 | 73.2 (±1.1) | 0.699 | 47.1 (±1.2) | 0.420 |
| MPNet FT | 79.2 (±2.5) | 0.799 | 72.9 (±0.5) | 0.688 | 48.4 (±1.9) | 0.438 |
| **Ensemble (Ours)** | **83.0 (±2.0)** | **0.829** | **81.1 (±1.4)** | **0.778** | **49.8 (±0.7)** | **0.534** |


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
