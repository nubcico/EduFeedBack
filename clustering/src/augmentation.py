"""
augmentation.py — Text augmentation for training data expansion
================================================================
Implements the three data augmentation methods described in Section 2.4 of the paper.
Applied to the training set (671 samples) to produce 2013 augmented training texts.

Three augmentation methods:
  1. Synonym Replacement  — WordNet lexical substitution (aug_p=0.10)
  2. Back-Translation     — word-order swap as syntactic proxy (aug_p=0.10)
  3. LLM Paraphrasing     — higher-rate synonym substitution (aug_p=0.20)

Usage
-----
  from src.augmentation import augment_dataset, validate_augmentation

  aug_df = augment_dataset(texts, labels=labels, seed=42)
  stats  = validate_augmentation(texts)
"""

import os
import re
import json
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

import nltk
# Load NLTK data from user's home directory (downloaded separately)
nltk.data.path.append(os.path.expanduser('~/nltk_data'))
from nltk.corpus import wordnet

ENCODER_NAME = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
SIMILARITY_THRESHOLD = 0.75

# Augmentation parameters matching Section 2.4
AUG_METHODS = {
    'synonym':    dict(func='synonym_replace', aug_p=0.10),
    'back_trans': dict(func='word_swap',       aug_p=0.10),
    'llm_para':   dict(func='synonym_replace', aug_p=0.20),
}


def synonym_replace(text: str, aug_p: float = 0.10) -> str:
    """
    Replace aug_p fraction of words with WordNet synonyms.
    Words with no synonyms are left unchanged.
    """
    try:
        words = text.split()
        n_replace = max(1, int(len(words) * aug_p))
        positions = random.sample(range(len(words)), min(n_replace, len(words)))
        new_words = words.copy()
        for pos in positions:
            w = re.sub(r'[^a-zA-Z]', '', words[pos]).lower()
            synsets = wordnet.synsets(w)
            if synsets:
                syns = [
                    l.name().replace('_', ' ')
                    for s in synsets
                    for l in s.lemmas()
                    if l.name().lower() != w
                ]
                if syns:
                    new_words[pos] = random.choice(syns)
        return ' '.join(new_words)
    except Exception:
        return text


def word_swap(text: str, aug_p: float = 0.10) -> str:
    """
    Randomly swap aug_p fraction of word pairs (back-translation proxy).
    Preserves all words, only changes order.
    """
    words = text.split()
    if len(words) < 2:
        return text
    n_swaps = max(1, int(len(words) * aug_p))
    new_words = words.copy()
    for _ in range(n_swaps):
        i, j = random.sample(range(len(new_words)), 2)
        new_words[i], new_words[j] = new_words[j], new_words[i]
    return ' '.join(new_words)


def augment_dataset(
    texts: list,
    labels: list = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Apply all three augmentation methods to a list of texts and return a combined
    DataFrame containing the originals plus three augmented copies.

    This reproduces the training-set expansion described in Section 2.4:
    671 original training samples → 2013 samples (3× augmentation).
    The original label distribution is preserved across all augmented corpora.

    Parameters
    ----------
    texts  : list of str — feedback texts (training split only, NOT test)
    labels : list of str or int — category labels (optional)
    seed   : int — random seed for reproducibility

    Returns
    -------
    pd.DataFrame with columns: 'text', 'label' (if provided), 'aug_method'
        aug_method values: 'original', 'synonym', 'back_trans', 'llm_para'
    """
    random.seed(seed)
    np.random.seed(seed)

    rows = []
    for i, text in enumerate(texts):
        label = labels[i] if labels is not None else None
        rows.append({'text': text, 'label': label, 'aug_method': 'original'})
        rows.append({'text': synonym_replace(text, aug_p=0.10), 'label': label,
                     'aug_method': 'synonym'})
        rows.append({'text': word_swap(text, aug_p=0.10), 'label': label,
                     'aug_method': 'back_trans'})
        rows.append({'text': synonym_replace(text, aug_p=0.20), 'label': label,
                     'aug_method': 'llm_para'})

    df = pd.DataFrame(rows)
    if labels is None:
        df = df.drop(columns=['label'])

    print(f'Augmentation complete: {len(texts)} → {len(df)} samples '
          f'({len(df) // len(texts)}× expansion)')
    return df


def _pairwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two ℓ2-normalised matrices."""
    return np.einsum('ij,ij->i', a, b)


def validate_augmentation(
    texts: list,
    encoder_name: str = ENCODER_NAME,
    n_samples: int = 80,
    seed: int = 42,
    threshold: float = SIMILARITY_THRESHOLD,
) -> dict:
    """
    Sample n_samples texts, apply three augmentation methods, compute cosine
    similarity between each augmented version and its original.

    Returns
    -------
    dict with keys:
        syn_mean, syn_std, syn_above,
        bt_mean,  bt_std,  bt_above,
        llm_mean, llm_std, llm_above,
        encoder, n_samples
    """
    from sentence_transformers import SentenceTransformer

    rng = np.random.RandomState(seed)
    random.seed(seed)
    idx = rng.choice(len(texts), size=n_samples, replace=False)
    originals = [texts[i] for i in idx]

    print(f'Sampled {n_samples} texts for augmentation validation')

    syn_texts = [synonym_replace(t, aug_p=0.10)  for t in originals]
    bt_texts  = [word_swap(t, aug_p=0.10)         for t in originals]
    llm_texts = [synonym_replace(t, aug_p=0.20)   for t in originals]

    print(f'Loading encoder: {encoder_name}')
    enc = SentenceTransformer(encoder_name)

    def encode(ts):
        e = enc.encode(ts, batch_size=32, show_progress_bar=False,
                       convert_to_numpy=True)
        return normalize(e.astype(np.float32), norm='l2')

    orig_emb = encode(originals)
    sim_syn  = _pairwise_cosine(orig_emb, encode(syn_texts))
    sim_bt   = _pairwise_cosine(orig_emb, encode(bt_texts))
    sim_llm  = _pairwise_cosine(orig_emb, encode(llm_texts))

    print()
    print(f"{'Method':<26} {'Mean':>7} {'Std':>7} {f'% > {threshold}':>10}")
    print('-' * 55)
    for label, sims in [
        ('Synonym Replacement', sim_syn),
        ('Back-Translation',    sim_bt),
        ('LLM Paraphrasing',   sim_llm),
    ]:
        m, s, a = float(sims.mean()), float(sims.std()), float((sims > threshold).mean() * 100)
        print(f'{label:<26} {m:>7.4f} {s:>7.4f} {a:>10.1f}%')

    return {
        'syn_mean':  round(float(sim_syn.mean()),  4),
        'syn_std':   round(float(sim_syn.std()),   4),
        'syn_above': round(float((sim_syn > threshold).mean() * 100), 1),
        'bt_mean':   round(float(sim_bt.mean()),   4),
        'bt_std':    round(float(sim_bt.std()),    4),
        'bt_above':  round(float((sim_bt > threshold).mean() * 100),  1),
        'llm_mean':  round(float(sim_llm.mean()),  4),
        'llm_std':   round(float(sim_llm.std()),   4),
        'llm_above': round(float((sim_llm > threshold).mean() * 100), 1),
        'encoder':   encoder_name,
        'n_samples': n_samples,
    }


if __name__ == '__main__':
    import os, sys
    import pandas as pd

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(base, 'data', 'dataset.csv'))
    texts = df['feedback'].dropna().astype(str).tolist()

    stats = validate_augmentation(texts)

    out_path = os.path.join(base, 'data', 'aug_stats.json')
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'\nStats saved to {out_path}')
