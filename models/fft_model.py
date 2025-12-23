import numpy as np
from PIL import Image
import io

def fft_score(image_bytes: bytes) -> float:
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = np.array(image)

    # FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    h, w = magnitude.shape
    high_freq_energy = magnitude[h//4:3*h//4, w//4:3*w//4].mean()

    score = min(high_freq_energy / 10.0, 1.0)
    return float(score)
