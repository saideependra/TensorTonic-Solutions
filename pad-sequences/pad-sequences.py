import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if len(seqs) == 0:
        return np.zeros((0,0), dtype=int)

    if max_len is None:
        max_len = max(len(s) for s in seqs)

    out = np.full((len(seqs),max_len),pad_value,dtype=int)
    for i, s in enumerate(seqs):
        keep = s[:max_len]
        out[i,:len(keep)] = keep

    return out