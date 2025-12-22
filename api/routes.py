from fastapi import APIRouter, UploadFile
from core.image_detector import detect_image
from core.video_detector import detect_video

router = APIRouter()

@router.post("/detect")
async def detect(file: UploadFile):
    if file.content_type.startswith("image"):
        return detect_image(file)
    elif file.content_type.startswith("video"):
        return detect_video(file)
    else:
        return {"error": "Unsupported media type"}
