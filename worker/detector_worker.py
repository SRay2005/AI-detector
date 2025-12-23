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
def jpeg_quantization_score(image_bytes: bytes) -> float:
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

        # ---------------- FFT ----------------
        gray = Image.open(io.BytesIO(image_bytes)).convert("L")
        arr = np.array(gray)
        f = np.fft.fftshift(np.fft.fft2(arr))
        fft = np.log(np.abs(f) + 1)
        fft_score = min(fft.mean() / 10.0, 1.0)

        # ---------------- Signals ----------------
        cnn = cnn_score(image_bytes)
        meta = metadata_score(image_bytes)
        jpeg = jpeg_quantization_score(image_bytes)

        # ---------------- Reliability ----------------
        fft_reliability = 1.0 - jpeg
        fft_effective = fft_score * fft_reliability * (1 - meta * 0.4)

        # ---------------- FINAL FUSION (FIXED) ----------------
        final = (
            0.40 * fft_effective +
            0.40 * cnn +
            0.20 * meta
        )

        final = round(min(final, 1.0), 3)

        # ---------------- MARKERS (FIXED) ----------------
        if final >= 0.65:
            verdict = "Likely AI-generated"
        elif final >= 0.45:
            verdict = "Possibly AI-generated"
        else:
            verdict = "Likely natural / uncertain"

        result = {
            "type": "image",
            "verdict": verdict,
            "confidence": final,
            "signals": {
                "fft": round(fft_score, 3),
                "cnn": round(cnn, 3),
                "metadata": round(meta, 3),
                "jpeg_quant": round(jpeg, 3)
            },
            "status": "completed",
            "note": (
                "CNN feature variance provides a weak semantic cue; "
                "FFT captures texture artifacts; metadata and JPEG "
                "adjust reliability. Decision is probabilistic."
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
