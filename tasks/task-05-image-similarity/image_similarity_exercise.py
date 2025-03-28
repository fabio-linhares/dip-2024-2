# image_similarity_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `compare_images(i1, i2)` that receives two grayscale images
represented as NumPy arrays (2D arrays of shape (H, W)) and returns a dictionary with the following metrics:

1. Mean Squared Error (MSE)
2. Peak Signal-to-Noise Ratio (PSNR)
3. Structural Similarity Index (SSIM) - simplified version without using external libraries
4. Normalized Pearson Correlation Coefficient (NPCC)

You must implement these functions yourself using only NumPy (no OpenCV, skimage, etc).

Each function should be implemented as a helper function and called inside `compare_images(i1, i2)`.

Function signature:
    def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

The return value should be like:
{
    "mse": float,
    "psnr": float,
    "ssim": float,
    "npcc": float
}

Assume that i1 and i2 are normalized grayscale images (values between 0 and 1).
"""

import numpy as np

def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:
    # Erro Quadrático Médio (MSE)
    def mse(i1, i2):
        return np.mean((i1 - i2) ** 2)
    
    # Função para calcular o Pico de Relação Sinal-Ruído (PSNR)
    def psnr(i1, i2):
        mse_value = mse(i1, i2)
        if mse_value == 0:
            return float('inf')
        max_pixel_value = 1.0 # Assumindo imagens normalizadas [0, 1]
        return 20 * np.log10(max_pixel_value / np.sqrt(mse_value))
    
    # Função simplificada para calcular o Índice de Similaridade Estrutural (SSIM)
    def ssim(i1, i2, C1=1e-4, C2=9e-4):
        mu1 = np.mean(i1)
        mu2 = np.mean(i2)
        sigma1 = np.var(i1)
        sigma2 = np.var(i2)
        sigma12 = np.mean((i1 - mu1) * (i2 - mu2))

        ssim_value = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
        return ssim_value
    
    # Função para calcular o Coeficiente de Correlação de Pearson Normalizado (NPCC)
    def npcc(i1, i2):
        mean1 = np.mean(i1)
        mean2 = np.mean(i2)
        numerator = np.sum((i1 - mean1) * (i2 - mean2))
        denominator = np.sqrt(np.sum((i1 - mean1) ** 2) * np.sum((i2 - mean2) ** 2))
        if denominator == 0:
            return 0  # Define NPCC como 0 caso o denominador seja zero
        return numerator / denominator

    # Retorno do dicionário com todos os cálculos
    return {
        "mse": mse(i1, i2),
        "psnr": psnr(i1, i2),
        "ssim": ssim(i1, i2),
        "npcc": npcc(i1, i2)
    }

# Sample grayscale images (2D arrays)
#i1 = np.array([
#    [0.0, 0.1, 0.2, 0.3, 0.4],
#    [0.5, 0.6, 0.7, 0.8, 0.9],
#    [1.0, 0.9, 0.8, 0.7, 0.6],
#    [0.5, 0.4, 0.3, 0.2, 0.1],
#    [0.0, 0.1, 0.2, 0.3, 0.4]
#])

#i2 = np.array([
#    [0.4, 0.3, 0.2, 0.1, 0.0],
#    [0.6, 0.5, 0.4, 0.3, 0.2],
#    [0.8, 0.7, 0.6, 0.5, 0.4],
#    [0.9, 0.8, 0.7, 0.6, 0.5],
#    [1.0, 0.9, 0.8, 0.7, 0.6]
#])

# Execute the comparison function
#results = compare_images(i1, i2)
#print("Image Similarity Results:")
#print(results)