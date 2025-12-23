import sys
import json
import os


# ---------------- METADATA SIGNAL ----------------
def metadata_score(image_bytes: bytes) -> float:
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
                "diffusion", "midjourney", "dall", "ai", "generated"
            ]):
                score += 0.5

    w, h = img.size
    if w % 64 == 0 and h % 64 == 0:
        score += 0.2

    return min(score, 1.0)


# ---------------- JPEG QUANTIZATION SIGNAL ----------------
def jpeg_quantization_score(image_bytes: bytes) -> float:
    """
    Detects unnatural JPEG quantization tables.
    AI images often have:
    - flat / uniform tables
    - non-camera-standard quantization
    """
    from PIL import Image
    import io
    import numpy as np

    img = Image.open(io.BytesIO(image_bytes))

    if img.format != "JPEG":
        return 0.0  # PNG/WebP skip

    qtables = img.quantization
    if not qtables:
        return 0.3

    score = 0.0

    for table in qtables.values():
        table = np.array(table)

        # Very flat tables → suspicious
        if table.std() < 10:
            score += 0.4

        # Many identical values → suspicious
        if len(set(table)) < 20:
            score += 0.3

    return min(score, 1.0)


# ---------------- MAIN WORKER ----------------
def main():
    input_path = os.path.abspath(sys.argv[1])
    output_path = os.path.abspath(sys.argv[2])

    base = {
        "type": "image",
        "verdict": "Uncertain",
        "confidence": 0.0,
        "signals": {},
        "status": "started"
    }

    with open(output_path, "w") as f:
        json.dump(base, f)
        f.flush()
        os.fsync(f.fileno())

    try:
        import numpy as np
        from PIL import Image
        import io

        with open(input_path, "rb") as f:
            image_bytes = f.read()

        # ---------- FFT ----------
        gray = Image.open(io.BytesIO(image_bytes)).convert("L")
        img = np.array(gray)

        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1)

        h, w = magnitude.shape
        hf_energy = magnitude[h//4:3*h//4, w//4:3*w//4].mean()
        fft_score = min(hf_energy / 10.0, 1.0)

        # ---------- METADATA ----------
        meta_score = metadata_score(image_bytes)

        # ---------- JPEG ----------
        jpeg_score = jpeg_quantization_score(image_bytes)

        # ---------- FUSION (GATED, NOT AVERAGED) ----------
        # Metadata reduces trust in FFT
        fft_effective = fft_score * (1 - min(meta_score, 0.7))

        # JPEG boosts confidence only if FFT is already high
        final_score = fft_effective
        if fft_score > 0.6:
            final_score += 0.4 * jpeg_score

        final_score = min(final_score, 1.0)

        # ---------- VERDICT ----------
        if final_score >= 0.7:
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
                "metadata": round(meta_score, 3),
                "jpeg_quant": round(jpeg_score, 3)
            },
            "status": "completed",
            "note": (
                "Decision uses frequency artifacts, metadata provenance, "
                "and JPEG quantization analysis. Verdict reflects likelihood, "
                "not certainty."
            )
        }

    except Exception as e:
        result = {
            "type": "image",
            "verdict": "Uncertain",
            "confidence": 0.0,
            "signals": {},
            "status": "failed",
            "error": str(e)
        }

    with open(output_path, "w") as f:
        json.dump(result, f)
        f.flush()
        os.fsync(f.fileno())


if __name__ == "__main__":
    main()
