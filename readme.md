### AI Image & Video Detector

### Run
pip install -r requirements.txt
uvicorn api.main:app --reload

### Endpoint
POST /detect
- image/*
- video/*
