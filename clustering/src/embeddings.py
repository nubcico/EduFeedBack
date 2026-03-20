"""
embeddings.py — Sentence encoder models and pre-computed embedding management
==============================================================================
Documents the five sentence encoder models used in the paper (Section 3.3.1)
and provides utilities for loading pre-computed NPZ archives.

Generating embeddings from scratch requires GPU access and ~30 GB RAM.
Pre-computed embeddings are provided in data/embeddings/ as NPZ archives.

NPZ file structure
------------------
Each .npz file contains:
  'labels'                                          → int array (N,)
  'thenlper/gte-large'                              → float32 array (N, 1024)
  'BAAI/bge-large-en-v1.5'                          → float32 array (N, 1024)
  'mixedbread-ai/mxbai-embed-large-v1'              → float32 array (N, 1024)
  'hkunlp/instructor-xl'                            → float32 array (N, 768)
  'sentence-transformers/multi-qa-mpnet-base-dot-v1'→ float32 array (N, 768)

Available files
---------------
  tsne_stage3_Test_individual.npz   — 288 held-out test samples (no augmentation)
  tsne_stage3_Train_individual.npz  — 2013 training samples
                                      rows 0–670   = original 671 training texts
                                      rows 671–2012 = augmented copies (3 methods × 671)
"""

import numpy as np
from sklearn.preprocessing import normalize

# Five sentence encoder models — HuggingFace IDs and embedding dimensions
ENCODER_MODELS = {
    'thenlper/gte-large':                                  1024,
    'BAAI/bge-large-en-v1.5':                             1024,
    'mixedbread-ai/mxbai-embed-large-v1':                 1024,
    'hkunlp/instructor-xl':                               768,
    'sentence-transformers/multi-qa-mpnet-base-dot-v1':   768,
}

MODEL_KEYS = list(ENCODER_MODELS.keys())


def inspect_npz(path: str) -> None:
    """Print shape and dtype information for every array in an NPZ file."""
    data = np.load(path, allow_pickle=True)
    print(f'NPZ: {path}')
    for key in data.files:
        arr = data[key]
        print(f'  {key:<55} {str(arr.shape):<18} {arr.dtype}')
    data.close()


def load_embeddings(npz_path: str, l2_normalize: bool = True) -> dict:
    """
    Load per-model embeddings from a single NPZ file.

    Parameters
    ----------
    npz_path     : str — path to .npz file
    l2_normalize : bool — apply ℓ2 normalisation (default True, matches paper)

    Returns
    -------
    dict with keys: model names + 'labels'
    """
    data = np.load(npz_path, allow_pickle=True)
    result = {'labels': data['labels']}
    for key in MODEL_KEYS:
        arr = data[key].astype(np.float32)
        result[key] = normalize(arr, norm='l2') if l2_normalize else arr
    data.close()
    n = result[MODEL_KEYS[0]].shape[0]
    print(f'Loaded {n} samples from {npz_path}')
    return result


def generate_embeddings(
    texts: list,
    model_id: str,
    batch_size: int = 32,
    device: str = 'cpu',
) -> np.ndarray:
    """
    Generate ℓ2-normalised embeddings for a list of texts using a sentence encoder.

    This function requires `sentence-transformers` and (for large batches) a GPU.
    The paper used GPU instances for all five encoders; expected runtime per model:
      - ~5 min (288 test samples) on CPU
      - ~30 min (2013 train samples) on CPU

    Parameters
    ----------
    texts    : list of str
    model_id : str — one of MODEL_KEYS (or any sentence-transformers model)
    batch_size: int
    device   : str — 'cpu' or 'cuda'

    Returns
    -------
    np.ndarray of shape (N, D), ℓ2-normalised float32
    """
    from sentence_transformers import SentenceTransformer

    print(f'Loading model: {model_id}')
    model = SentenceTransformer(model_id, device=device)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return normalize(emb.astype(np.float32), norm='l2')


def generate_all_embeddings(
    texts: list,
    output_path: str,
    labels: np.ndarray = None,
    batch_size: int = 32,
    device: str = 'cpu',
) -> None:
    """
    Run all five encoder models on a list of texts and save results to an NPZ file.

    This reproduces the pre-computed embedding archives in data/embeddings/.

    Parameters
    ----------
    texts       : list of str — feedback texts to encode
    output_path : str — path to save the NPZ file
    labels      : np.ndarray — integer category IDs (optional)
    batch_size  : int
    device      : str — 'cpu' or 'cuda'
    """
    arrays = {}
    if labels is not None:
        arrays['labels'] = np.array(labels)

    for model_id in MODEL_KEYS:
        print(f'\nEncoding with {model_id}...')
        arrays[model_id] = generate_embeddings(texts, model_id,
                                               batch_size=batch_size, device=device)

    np.savez_compressed(output_path, **arrays)
    print(f'\nSaved to: {output_path}')
    inspect_npz(output_path)


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    base    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    emb_dir = os.path.join(base, 'data', 'embeddings')

    print('=== Inspecting pre-computed embedding files ===\n')
    for fname in ['tsne_stage3_Test_individual.npz', 'tsne_stage3_Train_individual.npz']:
        path = os.path.join(emb_dir, fname)
        if os.path.exists(path):
            inspect_npz(path)
            print()

    print('=== Encoder model reference ===')
    for model_id, dim in ENCODER_MODELS.items():
        print(f'  {dim:>4}d  {model_id}')
