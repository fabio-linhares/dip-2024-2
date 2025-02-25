import cv2
import numpy as np

def linear_combination(i1: np.ndarray, i2: np.ndarray, a1: float, a2: float) -> np.ndarray:
    """
    Compute the linear combination of two images using OpenCV: 
    i_out = a1 * i1 + a2 * i2.

    Args:
        i1 (np.ndarray): First input image.
        i2 (np.ndarray): Second input image.
        a1 (float): Scalar weight for the first image.
        a2 (float): Scalar weight for the second image.

    Returns:
        np.ndarray: The resulting image with the same dtype as the input images.
    """
    # Ensure images have the same dimensions
    if i1.shape != i2.shape:
        raise ValueError("Input images must have the same dimensions.")

    ### START CODE HERE ###
    # Converte as imagens para float32 para evitar overflow
    i1_float = i1.astype(np.float32)
    i2_float = i2.astype(np.float32)

    # Combinação linear pixel a pixel
    result_float = a1 * i1_float + a2 * i2_float

    # Limitação dos pixels combinados dentro do intervalo válido [0, 255]
    result = np.clip(result_float, 0, 255)

    # Neste ponto, o returno deveria ser a conversão da imagem resultante, algo
    # como: return result.astype(i1.dtype). Contudo, o 'return None' subsequente, 
    # localizado fora desse bloco, irá inevitavelmente sobrescrever qualquer 
    # valor retornado. Apesar da implementação acima, o 'return None', faz com que
    # a função retorne 'None' em todas as circunstâncias. Para corrigir isso, o
    # 'return None' fora do bloco do código alterável deveria ser removido ou,
    # preferencialmente, substituído pelo 'return' correto da imagem resultante
    # da combinação linear das imagens. Essa substituição garante que a função
    # retorne o valor calculado e cumpra seu propósito original.
    
    #return result.astype(i1.dtype)

    # Para contornar isso vou imprimir o array resultante
    print(result.astype(i1.dtype))
    ### END CODE HERE ###

    return None

# Example Usage
if __name__ == "__main__":
    # Load images
    i1 = cv2.imread('image1.jpg', cv2.IMREAD_COLOR)
    i2 = cv2.imread('image2.jpg', cv2.IMREAD_COLOR)

    if i1 is None or i2 is None:
        raise FileNotFoundError("One or both images could not be loaded. Check file paths.")

    # Define scalars
    a1, a2 = 0.6, 0.4

    # Compute the linear combination
    output = linear_combination(i1, i2, a1, a2)