# RAG Evaluation

**Paper:** *Evidence-Grounded LLM Summarization for Actionable Student Feedback Analysis*

This folder contains the quantitative evaluation of the RAG summarization module,
corresponding to the ablation study added in response to Reviewer 1.

---

## Contents

| File | Description |
|---|---|
| `rag_evaluation.ipynb` | End-to-end RAG ablation: computes Faithfulness and Cluster Coverage for three RAG configurations |

---

## Evaluation Design

Because the student feedback dataset does not contain ground-truth summaries, standard
reference-based metrics (ROUGE, BERTScore) are not applicable. Instead, two
**evidence-grounded** metrics are used:

### Faithfulness
Measures the proportion of generated summary statements that are explicitly supported
by the retrieved evidence:

```
Faithfulness = N_supported / N_total
```

Each sentence-level statement is assessed by an automated LLM judge (GPT-based) with
the prompt: *"Is the following statement supported by the evidence? answer Supported / Not Supported"*

### Cluster Coverage
Measures how broadly the retrieved evidence spans the HDBSCAN-discovered semantic clusters:

```
Cluster Coverage = C_represented / C_retrieved
```

where `C_represented` is the number of distinct HDBSCAN clusters represented in the
retrieved set and `C_retrieved` is the total number of retrieved items.

---

## Results (Table 1 in the paper)

| Model | Faithfulness ↑ | Cluster Coverage ↑ |
|---|---|---|
| Baseline RAG | 0.72 | 0.40 |
| Supervised RAG | 0.80 | 0.56 |
| **Supervised + Unsupervised RAG (Ours)** | **0.87** | **0.76** |

The integrated framework achieves the highest Faithfulness (0.87) and Cluster Coverage (0.76),
demonstrating that incorporating both supervised category labels and unsupervised cluster
identifiers improves grounding and semantic diversity of generated summaries.

---

## How to Run

```bash
# From repo root, activate the environment first
pip install -r requirements.txt

jupyter notebook rag_evaluation/rag_evaluation.ipynb
```

> **Note:** The evaluation cells that call the LLM judge require a valid OpenAI API key
> set as the `OPENAI_API_KEY` environment variable.
