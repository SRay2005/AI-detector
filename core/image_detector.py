from models.cnn_detector import cnn_score
from models.fft_model import fft_score

def detect_image(file_bytes: bytes):
    cnn = cnn_score(file_bytes)
    fft = fft_score(file_bytes)

    # Weighted fusion
    final_confidence = 0.6 * cnn + 0.4 * fft

    return {
        "type": "image",
        "ai_generated": final_confidence > 0.5,
        "confidence": round(final_confidence, 3),
        "signals": {
            "cnn": round(cnn, 3),
            "fft": round(fft, 3)
        }
    }
