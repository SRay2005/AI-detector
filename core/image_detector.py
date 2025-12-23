import subprocess
import json
import uuid
import os
import sys

TEMP_DIR = "temp"

def detect_image(file_bytes: bytes):
    os.makedirs(TEMP_DIR, exist_ok=True)

    uid = str(uuid.uuid4())
    input_path = os.path.abspath(f"{TEMP_DIR}/{uid}.bin")
    output_path = os.path.abspath(f"{TEMP_DIR}/{uid}.json")

    with open(input_path, "wb") as f:
        f.write(file_bytes)

    try:
        subprocess.run(
            [
                sys.executable,
                os.path.abspath("worker/detector_worker.py"),
                input_path,
                output_path
            ],
            timeout=15,
            check=True
        )

        with open(output_path, "r") as f:
            result = json.load(f)

    except Exception as e:
        result = {
            "type": "image",
            "ai_generated": False,
            "confidence": 0.0,
            "signals": {},
            "error": str(e)
        }

    finally:
        for p in (input_path, output_path):
            if os.path.exists(p):
                os.remove(p)

    return result
