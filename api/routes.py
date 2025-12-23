from fastapi import APIRouter, UploadFile, File

router = APIRouter()

@router.post("/detect")
async def detect(file: UploadFile = File(...)):
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "message": "API is working"
    }
