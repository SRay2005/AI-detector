from utils.preprocessing import preprocess_image
from models.cnn_model import cnn_predict
from models.fft_model import fft_predict
from core.fusion import fuse_scores
from utils.metadata import metadata_score

def detect_image(file):
    
    image = preprocess_image(file)

    cnn_score = cnn_predict(image)
    fft_score = fft_predict(image)
    meta_score = metadata_score(file)

    final = fuse_scores(cnn_score, fft_score, metadata=meta_score)

    return {
        "type": "image",
        "ai_generated": final > 0.5,
        "confidence": round(final, 3),
        "signals": {
            "cnn": cnn_score,
            "fft": fft_score,
            "metadata": meta_score
        }
    }
