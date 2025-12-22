import numpy as np
import cv2

def fft_predict(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    high_freq_energy = magnitude[magnitude.shape[0]//4:, :].mean()

    # Heuristic normalisation
    score = min(high_freq_energy / 10, 1.0)
    return round(score, 3)
