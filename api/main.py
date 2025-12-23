from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="AI Image & Video Detector",
    version="1.0.0"
)

app.include_router(router)

@app.get("/")
def health_check():
    return {"status": "ok"}
