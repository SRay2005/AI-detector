def detect_image(file_bytes: bytes):
    return {
        "type": "image",
        "ai_generated": False,
        "confidence": 0.0,
        "signals": {}
    }
