import numpy as np

def temporal_score(frame_scores):
    diffs = np.diff(frame_scores)
    instability = np.mean(np.abs(diffs))

    # Higher instability → more likely AI
    score = min(instability * 5, 1.0)
    return round(score, 3)
