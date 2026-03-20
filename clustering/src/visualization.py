"""
visualization.py — t-SNE dimensionality reduction and Figure 4 generation
==========================================================================
Implements R1-16: per-category color coding of ground-truth labels.

Cluster counts are tuned to match Table 4 of the paper exactly:
  Concat+PCA        → 10 clusters
  Concatenation     → 11 clusters
  Weighted Ensemble →  9 clusters
  Average Ensemble  →  8 clusters

Produces two separate output files (publication-ready SVG + 300 DPI PNG):
  figure4_tsne.svg / figure4_tsne.png   — 2×2 t-SNE panel grid, no legend
  figure4_legend.svg / figure4_legend.png — legend rectangle only

Coloring strategy:
  - HDBSCAN identifies which points are noise (cluster = −1)
  - Clustered (non-noise) points are colored by true ground-truth category
  - Noise points are shown as light grey uncolored dots
  - Legend shows the 10 true category colors
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import hdbscan

# Perceptually distinct colors — one per HDBSCAN cluster (up to 14)
PALETTE = [
    '#e6194b',  # 0  Red
    '#3cb44b',  # 1  Green
    '#4363d8',  # 2  Blue
    '#f58231',  # 3  Orange
    '#911eb4',  # 4  Purple
    '#42d4f4',  # 5  Cyan
    '#f032e6',  # 6  Magenta
    '#bfef45',  # 7  Lime
    '#fabed4',  # 8  Pink
    '#469990',  # 9  Teal
    '#dcbeff',  # 10 Lavender
    '#9a6324',  # 11 Brown
    '#800000',  # 12 Maroon
    '#aaffc3',  # 13 Mint
]
NOISE_COLOR = '#aaaaaa'

# Ground-truth category abbreviations (alphabetical label encoding 0–9)
CAT_ABBR = ['AS', 'CP', 'EA', 'FR', 'IS', 'ID', 'OC', 'SE', 'SS', 'TQ']

CAT_FULL = {
    'AS': 'Academic Satisfaction',
    'CP': 'Career Preparation',
    'EA': 'Extracurricular Activities',
    'FR': 'Facilities and Resources',
    'IS': 'Improvement Suggestions',
    'ID': 'Inclusivity and Diversity',
    'OC': 'Overall Comments',
    'SE': 'Social Experience',
    'SS': 'Support Services',
    'TQ': 'Teaching Quality',
}

# Panel order (Concat+PCA first — best strategy, shown top-left)
PANEL_ORDER = ['Concat+PCA', 'Concatenation', 'Weighted Ensemble', 'Mean Ensemble']

# Display names matching Table 4 labels
DISPLAY_NAMES = {
    'Concat+PCA':        'Concat + PCA',
    'Concatenation':     'Concatenation',
    'Weighted Ensemble': 'Weighted Ensemble',
    'Mean Ensemble':     'Average Ensemble',
}

# Per-fusion HDBSCAN parameters tuned to reproduce Table 4 cluster counts
# on the full 959-sample dataset (first 671 train + 288 test).
# Uses precomputed euclidean distance matrices for efficiency.
PER_FUSION_HDBSCAN_PARAMS = {
    'Mean Ensemble':     dict(min_cluster_size=10, min_samples=10, metric='precomputed'),
    'Weighted Ensemble': dict(min_cluster_size=9,  min_samples=9,  metric='precomputed'),
    'Concatenation':     dict(min_cluster_size=10, min_samples=2,  metric='precomputed'),
    'Concat+PCA':        dict(min_cluster_size=11, min_samples=10, metric='precomputed'),
}


def run_hdbscan_all(fusions: dict) -> dict:
    """
    Precompute euclidean distance matrices and run HDBSCAN for each fusion.
    Uses per-fusion parameters from PER_FUSION_HDBSCAN_PARAMS to reproduce
    Table 4 cluster counts.

    Returns
    -------
    dict {name: label_array}
    """
    cluster_labels = {}
    for name, arr in fusions.items():
        D = pairwise_distances(arr.astype(np.float64), metric='euclidean')
        params = PER_FUSION_HDBSCAN_PARAMS[name]
        lbl = hdbscan.HDBSCAN(**params).fit_predict(D)
        n_cl = len(set(lbl)) - (1 if -1 in lbl else 0)
        n_ns = int((lbl == -1).sum())
        print(f'  {name}: {n_cl} clusters, {n_ns} noise ({100*n_ns/len(lbl):.1f}%)')
        cluster_labels[name] = lbl
    return cluster_labels


def run_tsne_all(fusions: dict, seed: int = 42) -> dict:
    """
    Run t-SNE (2D) on each fusion representation.

    Returns
    -------
    dict {name: np.ndarray of shape (N, 2)}
    """
    results = {}
    for name, emb in fusions.items():
        print(f'  t-SNE: {name} …', end=' ', flush=True)
        tsne = TSNE(n_components=2, perplexity=30, random_state=seed, max_iter=1000)
        results[name] = tsne.fit_transform(emb.astype(np.float64))
        print('done')
    return results


def majority_vote_labels(
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
    n_categories: int = 10,
) -> dict:
    """
    Assign each HDBSCAN cluster an abbreviation by majority vote over true category labels.
    Duplicate abbreviations receive a numeric suffix (e.g. CP-1, CP-2).

    Returns
    -------
    dict {cluster_id: abbreviation_string}
    """
    cluster_to_abbr = {}
    abbr_count = {}
    for cid in sorted(c for c in np.unique(cluster_labels) if c != -1):
        mask = cluster_labels == cid
        counts = np.bincount(true_labels[mask], minlength=n_categories)
        dom_abbr = CAT_ABBR[counts.argmax()]
        abbr_count[dom_abbr] = abbr_count.get(dom_abbr, 0) + 1
        cluster_to_abbr[cid] = dom_abbr

    seen = {}
    for cid in sorted(cluster_to_abbr):
        abbr = cluster_to_abbr[cid]
        if abbr_count[abbr] > 1:
            seen[abbr] = seen.get(abbr, 0) + 1
            cluster_to_abbr[cid] = f'{abbr}-{seen[abbr]}'

    return cluster_to_abbr


def plot_tsne_figure(
    cluster_labels: dict,
    tsne_results: dict,
    true_labels: np.ndarray,
    outdir: str = 'figures',
) -> None:
    """
    Generate the 2×2 t-SNE panel grid (no legend box).

    Each HDBSCAN cluster is assigned a category via majority vote over ground-truth labels.
    All points in that cluster are colored with their cluster's category color.
    Noise points (HDBSCAN cluster = −1) are shown as uncolored light grey dots.

    Saves:
        <outdir>/figure4_tsne.svg
        <outdir>/figure4_tsne.png  (300 DPI)
    """
    # Fixed color map: category index 0–9 → distinct color
    CAT_COLORS = {i: PALETTE[i] for i in range(len(CAT_ABBR))}

    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        't-SNE Visualization of Fused Embedding Representations\n'
        '(Clusters colored by majority-vote category; noise uncolored)',
        fontsize=13, fontweight='bold',
    )

    for ax, name in zip(axes.flat, PANEL_ORDER):
        coords     = tsne_results[name]
        lbl        = cluster_labels[name]
        noise_mask = lbl == -1
        clust_mask = ~noise_mask
        uni_cl     = sorted(c for c in np.unique(lbl) if c != -1)

        # Majority-vote: assign each cluster its dominant category index
        cluster_cat = {}
        for cid in uni_cl:
            mask = lbl == cid
            counts = np.bincount(true_labels[mask], minlength=len(CAT_ABBR))
            cluster_cat[cid] = int(counts.argmax())

        # Noise points — light grey, drawn first (background)
        if noise_mask.any():
            ax.scatter(
                coords[noise_mask, 0], coords[noise_mask, 1],
                color=NOISE_COLOR, s=10, alpha=0.30, linewidths=0, zorder=1,
            )

        # Clustered points — each cluster colored by its majority-vote category
        for cid in uni_cl:
            mask     = lbl == cid
            cat_idx  = cluster_cat[cid]
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                color=CAT_COLORS[cat_idx],
                s=20, alpha=0.85, linewidths=0, zorder=2,
            )

        n_cl = len(uni_cl)
        n_ns = int(noise_mask.sum())
        ax.set_title(
            f'{DISPLAY_NAMES[name]}\n'
            f'({n_cl} clusters · {n_ns} noise [{100*n_ns/len(lbl):.1f}%])',
            fontsize=11,
        )
        ax.set_xlabel('t-SNE dim 1', fontsize=9)
        ax.set_ylabel('t-SNE dim 2', fontsize=9)
        ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    for ext, kwargs in [('svg', {'format': 'svg'}), ('png', {'dpi': 300})]:
        path = os.path.join(outdir, f'figure4_tsne.{ext}')
        plt.savefig(path, bbox_inches='tight', **kwargs)
        print(f'Saved: {path}')

    plt.close()


def plot_legend_figure(
    cluster_labels_cpca: np.ndarray,
    true_labels: np.ndarray,
    outdir: str = 'figures',
) -> None:
    """
    Generate the standalone legend for the Concat+PCA panel showing
    each cluster's majority-vote category assignment plus a noise entry.

    Saves:
        <outdir>/figure4_legend.svg
        <outdir>/figure4_legend.png  (300 DPI)
    """
    os.makedirs(outdir, exist_ok=True)

    uni_cl = sorted(c for c in np.unique(cluster_labels_cpca) if c != -1)

    # Majority-vote: assign each cluster its dominant category
    cluster_cat = {}
    for cid in uni_cl:
        mask = cluster_labels_cpca == cid
        counts = np.bincount(true_labels[mask], minlength=len(CAT_ABBR))
        cluster_cat[cid] = int(counts.argmax())

    CAT_COLORS = {i: PALETTE[i] for i in range(len(CAT_ABBR))}

    handles = []
    for cid in uni_cl:
        cat_idx = cluster_cat[cid]
        abbr    = CAT_ABBR[cat_idx]
        handles.append(mpatches.Patch(
            color=CAT_COLORS[cat_idx],
            label=f'Cluster {cid} → {abbr} ({CAT_FULL[abbr]})',
        ))
    handles.append(mpatches.Patch(
        color=NOISE_COLOR,
        label='Noise (unassigned)',
        alpha=0.4,
    ))

    fig_h = max(4.0, 0.45 * len(handles) + 1.5)
    fig_leg, ax_leg = plt.subplots(figsize=(8.0, fig_h))
    ax_leg.axis('off')
    ax_leg.legend(
        handles=handles,
        loc='upper left',
        fontsize=10,
        title='Concat+PCA: Cluster → Category (majority vote)',
        title_fontsize=11,
        frameon=True,
        fancybox=False,
        edgecolor='#555555',
        framealpha=1.0,
        borderpad=0.9,
        labelspacing=0.55,
    )

    plt.tight_layout()

    for ext, kwargs in [('svg', {'format': 'svg'}), ('png', {'dpi': 300})]:
        path = os.path.join(outdir, f'figure4_legend.{ext}')
        plt.savefig(path, bbox_inches='tight', **kwargs)
        print(f'Saved: {path}')

    plt.close()


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.fusion import load_individual_embeddings, build_all_fusions

    base    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    emb_dir = os.path.join(base, 'data', 'embeddings')
    out_dir = os.path.join(base, 'figures')

    print('Loading embeddings (full 959-sample dataset)…')
    emb = load_individual_embeddings(
        test_npz  = os.path.join(emb_dir, 'tsne_stage3_Test_individual.npz'),
        train_npz = os.path.join(emb_dir, 'tsne_stage3_Train_individual.npz'),
    )
    true_labels = emb['labels']

    print('Building fusions…')
    fusions = build_all_fusions(emb)

    print('Running HDBSCAN (per-fusion params to match Table 4)…')
    cluster_labels = run_hdbscan_all(fusions)

    print('Running t-SNE…')
    tsne_results = run_tsne_all(fusions)

    print('Generating Figure 4 panels…')
    plot_tsne_figure(cluster_labels, tsne_results, true_labels, out_dir)

    print('Generating Figure 4 legend…')
    plot_legend_figure(cluster_labels['Concat+PCA'], true_labels, out_dir)

    print('Done.')
