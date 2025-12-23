import sys
import json
import os

def main():
    input_path = os.path.abspath(sys.argv[1])
    output_path = os.path.abspath(sys.argv[2])

    # ALWAYS write something first
    base_result = {
        "type": "image",
        "ai_generated": False,
        "confidence": 0.0,
        "signals": {},
        "status": "started"
    }

    with open(output_path, "w") as f:
        json.dump(base_result, f)
        f.flush()
        os.fsync(f.fileno())

    try:
        # Import heavy libs INSIDE try
        import numpy as np
        from PIL import Image
        import io

        with open(input_path, "rb") as f:
            image_bytes = f.read()

        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        img = np.array(image)

        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1)

        h, w = magnitude.shape
        hf_energy = magnitude[h//4:3*h//4, w//4:3*w//4].mean()
        score = float(min(hf_energy / 10.0, 1.0))

        result = {
            "type": "image",
            "ai_generated": score > 0.5,
            "confidence": round(score, 3),
            "signals": {"fft": round(score, 3)},
            "status": "completed"
        }

    except Exception as e:
        result = {
            "type": "image",
            "ai_generated": False,
            "confidence": 0.0,
            "signals": {},
            "status": "failed",
            "error": str(e)
        }

    # ALWAYS overwrite output
    with open(output_path, "w") as f:
        json.dump(result, f)
        f.flush()
        os.fsync(f.fileno())

if __name__ == "__main__":
    main()
