from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="AI Image & Video Detector")

app.include_router(router)

