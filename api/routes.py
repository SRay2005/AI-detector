from fastapi import APIRouter, UploadFile, File
from core.image_detector import detect_image

router = APIRouter()

@router.post("/detect")
async def detect(file: UploadFile = File(...)):
    data = await file.read()

    if file.content_type.startswith("image"):
        return detect_image(data)

    return {"error": "Unsupported file type"}
