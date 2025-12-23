from PIL import Image
import io


def preprocess_image(file):
    content = file.file.read()
    file.file.seek(0)
    image = Image.open(io.BytesIO(content)).convert("RGB")
    return image

