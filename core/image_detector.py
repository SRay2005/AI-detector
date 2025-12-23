from PIL import Image
import io
import random

def detect_image(file_bytes: bytes):
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    # TEMPORARY heuristic (placeholder)
    confidence = round(random.uniform(0.3, 0.9), 2)

    return {
        "type": "image",
        "ai_generated": confidence > 0.5,
        "confidence": confidence
    }
