from PIL import Image
import io


def metadata_score(file):
    try:
        file.file.seek(0)
        image = Image.open(file.file)
        exif = image.getexif()
        return 0.2 if exif else 0.8
    except:
        return 0.7
