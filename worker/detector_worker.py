# ==============================
# AI IMAGE DETECTOR – CHECKPOINT 1
# Multi-iteration ensemble version
# ==============================

import sys
import os
import json
import numpy as np
from PIL import Image
import io

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.cnn_infer import cnn_score


def metadata_score(image_bytes: bytes) -> float:
    img = Image.open(io.BytesIO(image_bytes))
    exif = img.getexif()

    score = 0.0
    if not exif or len(exif) == 0:
        score += 0.4
    if exif.get(271) is None:
        score += 0.2

    return min(score, 1.0)


def fft_score_once(image_bytes: bytes) -> float:
    gray = Image.open(io.BytesIO(image_bytes)).convert("L")
    arr = np.array(gray)

    # Minor numerical jitter via float conversion
    arr = arr.astype(np.float32) + np.random.normal(0, 0.5, arr.shape)

    f = np.fft.fftshift(np.fft.fft2(arr))
    fft = np.log(np.abs(f) + 1)
    return min(fft.mean() / 10.0, 1.0)


def fft_score(image_bytes: bytes, runs: int = 20) -> dict:
    scores = [fft_score_once(image_bytes) for _ in range(runs)]
    return {
        "mean": round(float(np.mean(scores)), 3),
        "std": round(float(np.std(scores)), 3)
    }


def main():
    inp, out = sys.argv[1], sys.argv[2]

    try:
        image_bytes = open(inp, "rb").read()

        cnn = cnn_score(image_bytes, runs=20)
        fft = fft_score(image_bytes, runs=20)
        meta = metadata_score(image_bytes)

        # CP1 fusion on MEANS
        final = (
            0.45 * fft["mean"] +
            0.35 * cnn["mean"] +
            0.20 * meta
        )

        # Stability penalty (important)
        instability = cnn["std"] + fft["std"]
        final = final - 0.5 * instability

        final = round(float(np.clip(final, 0.0, 1.0)), 3)

        if final >= 0.8:
            verdict = "AI-generated"
        elif final >= 0.6:
            verdict = "Possibly AI-generated"
        elif final >= 0.4:
            verdict = "Uncertain"
        else:
            verdict = "Likely natural"

        result = {
            "type": "image",
            "verdict": verdict,
            "confidence": final,
            "signals": {
                "cnn_mean": cnn["mean"],
                "cnn_std": cnn["std"],
                "fft_mean": fft["mean"],
                "fft_std": fft["std"],
                "metadata": meta,
                "instability": round(instability, 3)
            },
            "status": "completed",
            "note": (
                "CHECKPOINT 1 ensemble detector. "
                "Final decision based on mean signal across multiple stochastic passes. "
                "Instability across runs reduces confidence."
            )
        }

    except Exception as e:
        result = {
            "type": "image",
            "status": "failed",
            "error": str(e)
        }

    with open(out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
