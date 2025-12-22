from PIL import Image
import io


def preprocess_image(file):
    content = file.file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    return image
