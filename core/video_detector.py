from utils.video_utils import extract_frames
from models.cnn_model import cnn_predict
from models.temporal_model import temporal_score
from core.fusion import fuse_scores

def detect_video(file):
    frames = extract_frames(file, fps=2)

    frame_scores = [cnn_predict(f) for f in frames]
    avg_score = sum(frame_scores) / len(frame_scores)

    temporal = temporal_score(frame_scores)

    final = fuse_scores(avg_score, avg_score, temporal)

    return {
        "type": "video",
        "ai_generated": final > 0.5,
        "confidence": round(final, 3),
        "frames_analyzed": len(frames)
    }
