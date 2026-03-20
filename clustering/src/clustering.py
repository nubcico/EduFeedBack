"""
clustering.py — HDBSCAN clustering and evaluation metrics
==========================================================
Implements Sections 3.3.3 and Table 4 of the paper.

R1-07: HDBSCAN configured with min_cluster_size=10, min_samples=5, metric='euclidean'.
R1-10: HDBSCAN is deterministic — variance estimates are not applicable.
"""

import numpy as np
import hdbscan
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

# R1-07: Hyperparameters reported in Section 3.3.3
HDBSCAN_PARAMS = dict(
    min_cluster_size=10,
    min_samples=5,
    metric='euclidean',
    # cluster_selection_epsilon is not set (default = 0.0)
)


def run_hdbscan(embeddings: np.ndarray, **hdbscan_params) -> np.ndarray:
    """
    Fit HDBSCAN and return cluster label array.

    Parameters
    ----------
    embeddings : np.ndarray, shape (N, D)
    **hdbscan_params : overrides for HDBSCAN_PARAMS

    Returns
    -------
    np.ndarray of int, shape (N,)
        Cluster IDs; -1 indicates noise.
    """
    params = {**HDBSCAN_PARAMS, **hdbscan_params}
    return hdbscan.HDBSCAN(**params).fit_predict(embeddings)


def compute_metrics(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute Silhouette, Davies-Bouldin, and Calinski-Harabasz scores
    on non-noise points only.

    Returns
    -------
    dict with keys: n_clusters, n_noise, noise_pct, silhouette, db, ch
    """
    mask = labels != -1
    n_cl = int(len(set(labels[mask])))
    n_ns = int((labels == -1).sum())

    if n_cl >= 2 and mask.sum() > n_cl:
        sil = float(silhouette_score(embeddings[mask], labels[mask]))
        db  = float(davies_bouldin_score(embeddings[mask], labels[mask]))
        ch  = float(calinski_harabasz_score(embeddings[mask], labels[mask]))
    else:
        sil = db = ch = float('nan')

    return {
        'n_clusters': n_cl,
        'n_noise':    n_ns,
        'noise_pct':  100 * n_ns / len(labels),
        'silhouette': sil,
        'db':         db,
        'ch':         ch,
    }


def cluster_all_fusions(fusions: dict, **hdbscan_params) -> dict:
    """
    Run HDBSCAN on each fusion representation.

    Returns
    -------
    dict {fusion_name: label_array}
    """
    labels = {}
    for name, emb in fusions.items():
        lbl = run_hdbscan(emb, **hdbscan_params)
        labels[name] = lbl
        n_cl = len(set(lbl)) - (1 if -1 in lbl else 0)
        n_ns = int((lbl == -1).sum())
        print(f'  {name}: {n_cl} clusters, {n_ns} noise ({100*n_ns/len(lbl):.1f}%)')
    return labels


def print_metrics_table(fusions: dict, cluster_labels: dict) -> None:
    """
    Print Table 4 (clustering evaluation metrics) to stdout.
    Includes the R1-10 determinism footnote.
    """
    print('=== Table 4: Clustering Metrics ===')
    print(f"{'Fusion Strategy':<22} {'Silhouette↑':>11} {'DB↓':>7} {'CH↑':>10} "
          f"{'Clusters':>9} {'Noise':>13}")
    print('-' * 80)

    for name, emb in fusions.items():
        lbl = cluster_labels[name]
        m   = compute_metrics(emb, lbl)
        noise_str = f"{m['n_noise']} ({m['noise_pct']:.1f}%)"
        print(
            f"{name:<22} {m['silhouette']:>11.3f} {m['db']:>7.3f} {m['ch']:>10.2f} "
            f"{m['n_clusters']:>9} {noise_str:>13}"
        )

    print()
    print(
        '†HDBSCAN is deterministic given fixed hyperparameters and embedding '
        'representations;\n variance estimates are therefore not applicable for '
        'clustering metrics.'
    )


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.fusion import load_individual_embeddings, build_all_fusions

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    emb = load_individual_embeddings(
        os.path.join(base, 'data', 'embeddings', 'tsne_stage3_Test_individual.npz')
    )
    fusions = build_all_fusions(emb)
    labels  = cluster_all_fusions(fusions)
    print()
    print_metrics_table(fusions, labels)
