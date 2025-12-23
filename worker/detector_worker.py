import sys
import os
import json

# ---------------- PATH FIX ----------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.cnn_infer import cnn_score

# ---------------- METADATA ----------------
def metadata_score(image_bytes: bytes) -> float:
    from PIL import Image
    import io

    img = Image.open(io.BytesIO(image_bytes))
    exif = img.getexif()
    score = 0.0

    if not exif or len(exif) == 0:
        score += 0.4
    if exif.get(271) is None or exif.get(272) is None:
        score += 0.2

    w, h = img.size
    if w % 64 == 0 and h % 64 == 0:
        score += 0.2

    return min(score, 1.0)


# ---------------- JPEG ----------------
def jpeg_quant_score(image_bytes: bytes) -> float:
    from PIL import Image
    import io
    import numpy as np

    img = Image.open(io.BytesIO(image_bytes))
    if img.format != "JPEG" or not img.quantization:
        return 0.0

    q = np.array(list(img.quantization.values())[0])
    return 0.3 if q.std() < 10 else 0.0


# ---------------- MAIN ----------------
def main():
    inp, out = sys.argv[1], sys.argv[2]

    try:
        import numpy as np
        from PIL import Image
        import io

        image_bytes = open(inp, "rb").read()

        # ---- FFT ----
        gray = Image.open(io.BytesIO(image_bytes)).convert("L")
        img = np.array(gray)
        f = np.fft.fftshift(np.fft.fft2(img))
        fft = np.log(np.abs(f) + 1)
        fft_score = min(fft.mean() / 10.0, 1.0)

        # ---- Other signals ----
        meta = metadata_score(image_bytes)
        jpeg = jpeg_quant_score(image_bytes)
        cnn = cnn_score(image_bytes)

        fft_reliability = 1.0 - jpeg
        fft_effective = fft_score * fft_reliability * (1 - meta * 0.4)

        # ---- Fusion (CNN dominates) ----
        final = (
            0.55 * cnn +
            0.35 * fft_effective +
            0.10 * (fft_score if fft_score > 0.9 else 0)
        )

        final = min(final, 1.0)

        if final >= 0.75:
            verdict = "Highly likely AI-generated"
        elif final >= 0.4:
            verdict = "Possibly AI-generated"
        else:
            verdict = "Uncertain / likely natural"

        result = {
            "type": "image",
            "verdict": verdict,
            "confidence": round(final, 3),
            "signals": {
                "fft": round(fft_score, 3),
                "metadata": round(meta, 3),
                "jpeg_quant": round(jpeg, 3),
                "cnn": round(cnn, 3)
            },
            "status": "completed",
            "note": (
                "Noise-residual CNN detects diffusion smoothness; FFT detects texture artifacts; "
                "metadata and JPEG adjust reliability. Verdict reflects probability, not certainty."
            )
        }

    except Exception as e:
        result = {
            "type": "image",
            "status": "failed",
            "error": str(e)
        }

    with open(out, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
