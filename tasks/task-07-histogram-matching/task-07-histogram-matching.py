# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import cv2 as cv
import numpy as np
import scikitimage as ski

def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
    # Criar array para a imagem resultante
    matched_img = np.zeros_like(source_img)
    
    # Processar cada canal (R, G, B) separadamente
    for i in range(3):  # Para cada canal de cor
        # Extrair o canal atual
        source_channel = source_img[:, :, i]
        reference_channel = reference_img[:, :, i]
        
        # Calcular histogramas normalizados e CDFs
        hist_source, bins = np.histogram(source_channel.flatten(), 256, [0, 256], density=True)
        hist_reference, _ = np.histogram(reference_channel.flatten(), 256, [0, 256], density=True)
        
        # Calcular CDFs (Cumulative Distribution Functions)
        cdf_source = hist_source.cumsum()
        cdf_source = 255 * cdf_source / cdf_source[-1]  # Normalizar para 0-255
        
        cdf_reference = hist_reference.cumsum()
        cdf_reference = 255 * cdf_reference / cdf_reference[-1]  # Normalizar para 0-255
        
        # Criar mapeamento usando interpolação
        # Para cada valor na CDF da fonte, encontre o valor correspondente na CDF de referência
        interp_values = np.interp(cdf_source, cdf_reference, np.arange(256))
        
        # Mapear valores de pixels do canal fonte
        lookup_table = np.uint8(interp_values)
        matched_channel = lookup_table[source_channel]
        
        # Armazenar o canal processado na imagem resultante
        matched_img[:, :, i] = matched_channel
    
    return matched_img.astype(np.uint8)
    # pass
