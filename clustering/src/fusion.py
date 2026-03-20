"""
fusion.py — Embedding loading and fusion strategies
=====================================================
Implements the four fusion methods from Section 3.3.2 of the paper:
  1. Mean Ensemble   — element-wise average of ℓ2-normalised embeddings
  2. Weighted Ensemble — performance-weighted average
  3. Concatenation   — direct feature-level concatenation (4608 dims)
  4. Concat+PCA      — concatenation + PCA retaining 66.6% cumulative variance (R1-08)
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# Five sentence encoder models used in the paper
MODEL_KEYS = [
    'thenlper/gte-large',
    'BAAI/bge-large-en-v1.5',
    'mixedbread-ai/mxbai-embed-large-v1',
    'hkunlp/instructor-xl',
    'sentence-transformers/multi-qa-mpnet-base-dot-v1',
]

# Encoder performance weights (from paper, Section 3.3.2)
DEFAULT_WEIGHTS = {
    'thenlper/gte-large':                                  1.2,
    'BAAI/bge-large-en-v1.5':                             1.3,
    'mixedbread-ai/mxbai-embed-large-v1':                 1.2,
    'hkunlp/instructor-xl':                               1.0,
    'sentence-transformers/multi-qa-mpnet-base-dot-v1':   1.0,
}


def load_individual_embeddings(test_npz: str, train_npz: str = None) -> dict:
    """
    Load and ℓ2-normalise per-model embeddings.

    Parameters
    ----------
    test_npz : str
        Path to tsne_stage3_Test_individual.npz (288 test samples).
    train_npz : str, optional
        Path to tsne_stage3_Train_individual.npz.  When provided the function
        reconstructs the full 959-sample original dataset by prepending the
        first 671 rows of the training file (original samples only; rows
        672–2013 are augmented copies).

    Returns
    -------
    dict  {model_key: np.ndarray of shape (N, D)}
          Also includes key 'labels' with integer category IDs.
    """
    ind_test = np.load(test_npz, allow_pickle=True)

    if train_npz is not None:
        ind_train = np.load(train_npz, allow_pickle=True)
        embeddings = {}
        for k in MODEL_KEYS:
            full = np.concatenate(
                [ind_train[k][:671], ind_test[k]], axis=0
            ).astype(np.float32)
            embeddings[k] = normalize(full, norm='l2')
        embeddings['labels'] = np.concatenate(
            [ind_train['labels'][:671], ind_test['labels']]
        )
    else:
        embeddings = {}
        for k in MODEL_KEYS:
            embeddings[k] = normalize(ind_test[k].astype(np.float32), norm='l2')
        embeddings['labels'] = ind_test['labels']

    n = embeddings[MODEL_KEYS[0]].shape[0]
    for k in MODEL_KEYS:
        print(f'  {k}: {embeddings[k].shape}')
    print(f'Total samples: {n}')
    return embeddings


def build_mean_fusion(embeddings: dict) -> np.ndarray:
    """
    Mean Ensemble: zero-pad each embedding to max_dim, then average and re-normalise.
    """
    max_dim = max(embeddings[k].shape[1] for k in MODEL_KEYS)
    padded = []
    for k in MODEL_KEYS:
        e = embeddings[k]
        if e.shape[1] < max_dim:
            pad = np.zeros((e.shape[0], max_dim - e.shape[1]), dtype=np.float32)
            e = np.concatenate([e, pad], axis=1)
        padded.append(e)
    mean = np.mean(np.stack(padded, axis=0), axis=0).astype(np.float32)
    return normalize(mean, norm='l2')


def build_weighted_fusion(embeddings: dict, weights: dict = None) -> np.ndarray:
    """
    Weighted Ensemble: performance-weighted average (zero-padded), then re-normalised.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    max_dim = max(embeddings[k].shape[1] for k in MODEL_KEYS)
    total_w = sum(weights[k] for k in MODEL_KEYS)
    n = embeddings[MODEL_KEYS[0]].shape[0]
    wsum = np.zeros((n, max_dim), dtype=np.float32)
    for k in MODEL_KEYS:
        e = embeddings[k]
        if e.shape[1] < max_dim:
            pad = np.zeros((n, max_dim - e.shape[1]), dtype=np.float32)
            e = np.concatenate([e, pad], axis=1)
        wsum += (weights[k] / total_w) * e
    return normalize(wsum, norm='l2')


def build_concat_fusion(embeddings: dict) -> np.ndarray:
    """
    Concatenation: stack all five embeddings along the feature axis → 4608 dims.
    """
    raw = np.concatenate([embeddings[k] for k in MODEL_KEYS], axis=1).astype(np.float32)
    print(f'Concatenation dim: {raw.shape[1]}')
    return normalize(raw, norm='l2')


def build_concat_pca_fusion(
    embeddings: dict,
    n_components: float = 0.666,
    seed: int = 42,
) -> tuple:
    """
    Concat+PCA (R1-08): concatenate then apply PCA retaining n_components cumulative
    variance.  sklearn interprets a float in (0, 1) as a variance threshold.

    Returns
    -------
    (reduced_embeddings, fitted_pca_model)
    """
    raw = np.concatenate([embeddings[k] for k in MODEL_KEYS], axis=1).astype(np.float64)
    pca = PCA(n_components=n_components, random_state=seed)
    reduced = pca.fit_transform(raw).astype(np.float32)
    print(
        f'Concat+PCA → {pca.n_components_} components '
        f'retaining {pca.explained_variance_ratio_.sum() * 100:.1f}% variance'
    )
    return normalize(reduced, norm='l2'), pca


def build_all_fusions(embeddings: dict, seed: int = 42) -> dict:
    """
    Build all four fusion representations and return as a dict.

    Keys: 'Concat+PCA', 'Concatenation', 'Weighted Ensemble', 'Mean Ensemble'
    """
    concat_pca, _ = build_concat_pca_fusion(embeddings, seed=seed)
    return {
        'Concat+PCA':        concat_pca,
        'Concatenation':     build_concat_fusion(embeddings),
        'Weighted Ensemble': build_weighted_fusion(embeddings),
        'Mean Ensemble':     build_mean_fusion(embeddings),
    }


if __name__ == '__main__':
    import os, sys
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    emb = load_individual_embeddings(
        os.path.join(base, 'data', 'embeddings', 'tsne_stage3_Test_individual.npz')
    )
    fusions = build_all_fusions(emb)
    for name, arr in fusions.items():
        print(f'{name}: {arr.shape}')
