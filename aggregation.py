import numpy as np


def compute_frame_scores(embeddings):
    """
    Compute per-frame temporal variation score.
    Inspired by STOP Eq. 8-10: score based on L2 distance
    between adjacent frame embeddings.
    """
    N = len(embeddings)
    if N == 1:
        return np.array([1.0])

    deltas = embeddings[1:] - embeddings[:-1]        # (N-1, D)
    diff_scores = np.sum(deltas ** 2, axis=1)        # (N-1,)

    frame_scores = np.zeros(N)
    frame_scores[0] = diff_scores[0]
    frame_scores[-1] = diff_scores[-1]
    for i in range(1, N - 1):
        frame_scores[i] = (diff_scores[i - 1] + diff_scores[i]) / 2.0

    return frame_scores


def uniform_pooling(embeddings):
    """Baseline: simple mean of all frame embeddings."""
    return embeddings.mean(axis=0)


def tv_weighted_pooling(embeddings):
    """
    Ours: weight each frame by its temporal variation score,
    then take a weighted sum. Frames with more motion change
    contribute more to the final representation.
    """
    scores = compute_frame_scores(embeddings)
    weights = scores / (scores.sum() + 1e-8)          # normalize
    return (embeddings * weights[:, None]).sum(axis=0)


def topk_pooling(embeddings, k=4):
    """
    Ablation: select top-K highest-variation frames
    and mean pool only those.
    """
    N = len(embeddings)
    if N <= k:
        return embeddings.mean(axis=0)

    scores = compute_frame_scores(embeddings)
    topk_idx = np.argsort(scores)[-k:]
    return embeddings[topk_idx].mean(axis=0)