from PIL import Image
import io


def metadata_score(file):
    try:
        image = Image.open(io.BytesIO(file.file.read()))
        exif = image.getexif()
        if not exif:
            return 0.8  # suspicious
        return 0.2
    except:
        return 0.7
