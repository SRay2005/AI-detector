import sys
import os
import json

# ==========================================================
# FORCE PROJECT ROOT INTO PYTHON PATH (CRITICAL)
# ==========================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import AFTER path fix
from models.cnn_infer import cnn_score


# ==========================================================
# METADATA SIGNAL
# ==========================================================
def metadata_score(image_bytes: bytes) -> float:
    """
    Higher = weaker camera provenance.
    Used only as a mild dampener.
    """
    from PIL import Image
    import io

    score = 0.0
    img = Image.open(io.BytesIO(image_bytes))
    exif = img.getexif()

    if not exif or len(exif) == 0:
        score += 0.4
    else:
        make = exif.get(271)
        model = exif.get(272)
        software = exif.get(305)

        if make is None or model is None:
            score += 0.2

        if software:
            s = str(software).lower()
            if any(x in s for x in [
                "diffusion", "midjourney", "dall",
                "ai", "generated", "automatic1111"
            ]):
                score += 0.5

    w, h = img.size
    if w % 64 == 0 and h % 64 == 0:
        score += 0.2

    return min(score, 1.0)


# ==========================================================
# JPEG QUANTIZATION (FFT RELIABILITY ESTIMATOR)
# ==========================================================
def jpeg_quantization_score(image_bytes: bytes) -> float:
    """
    Measures recompression strength.
    This reduces FFT reliability; it is NOT an AI signal.
    """
    from PIL import Image
    import io
    import numpy as np

    img = Image.open(io.BytesIO(image_bytes))

    if img.format != "JPEG":
        return 0.0

    qtables = img.quantization
    if not qtables:
        return 0.3

    score = 0.0
    for table in qtables.values():
        table = np.array(table)

        if table.std() < 10:
            score += 0.4

        if len(set(table)) < 20:
            score += 0.3

    return min(score, 1.0)


# ==========================================================
# MAIN WORKER
# ==========================================================
def main():
    input_path = os.path.abspath(sys.argv[1])
    output_path = os.path.abspath(sys.argv[2])

    # ---- crash-safe initial write ----
    with open(output_path, "w") as f:
        json.dump({
            "type": "image",
            "verdict": "Uncertain",
            "confidence": 0.0,
            "signals": {},
            "status": "started"
        }, f)
        f.flush()
        os.fsync(f.fileno())

    try:
        import numpy as np
        from PIL import Image
        import io

        with open(input_path, "rb") as f:
            image_bytes = f.read()

        # ================= FFT =================
        gray = Image.open(io.BytesIO(image_bytes)).convert("L")
        img = np.array(gray)

        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1)

        h, w = magnitude.shape
        fft_score = min(
            magnitude[h // 4: 3 * h // 4, w // 4: 3 * w // 4].mean() / 10.0,
            1.0
        )

        # ================= OTHER SIGNALS =================
        meta = metadata_score(image_bytes)
        jpeg = jpeg_quantization_score(image_bytes)
        cnn = cnn_score(image_bytes)

        # ================= RELIABILITY GATING =================
        fft_reliability = 1.0 - min(jpeg, 0.7)
        meta_damp = min(meta * 0.4, 0.25)

        fft_effective = fft_score * fft_reliability * (1 - meta_damp)

        # Extreme FFT override ONLY if FFT is reliable
        fft_override = 0.15 if (fft_score >= 0.9 and jpeg < 0.3) else 0.0

        # ================= FINAL FUSION =================
        final_score = (
            0.45 * fft_effective +
            0.35 * cnn +
            0.2 * fft_override
        )

        final_score = min(final_score, 1.0)

        # ================= VERDICT =================
        if final_score >= 0.75:
            verdict = "Highly likely AI-generated"
        elif final_score >= 0.4:
            verdict = "Possibly AI-generated or edited"
        else:
            verdict = "Uncertain / likely natural"

        result = {
            "type": "image",
            "verdict": verdict,
            "confidence": round(final_score, 3),
            "signals": {
                "fft": round(fft_score, 3),
                "metadata": round(meta, 3),
                "jpeg_quant": round(jpeg, 3),
                "fft_reliability": round(fft_reliability, 3),
                "cnn": round(cnn, 3)
            },
            "status": "completed",
            "note": (
                "FFT detects texture artifacts; JPEG estimates FFT reliability; "
                "metadata provides weak provenance; CNN evaluates semantic realism. "
                "No single signal dominates the decision."
            )
        }

    except Exception as e:
        result = {
            "type": "image",
            "verdict": "Uncertain",
            "confidence": 0.0,
            "status": "failed",
            "error": str(e)
        }

    # ---- guaranteed final write ----
    with open(output_path, "w") as f:
        json.dump(result, f)
        f.flush()
        os.fsync(f.fileno())


if __name__ == "__main__":
    main()
