def fuse_scores(cnn, fft, temporal=None, metadata=0.5):
    score = (
        0.45 * cnn +
        0.30 * fft +
        0.15 * (temporal if temporal is not None else cnn) +
        0.10 * metadata
    )
    
    return min(max(score, 0.0), 1.0)
